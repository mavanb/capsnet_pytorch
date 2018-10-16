""" Module with ignite trainers

Contains an abstract Trainer class which handles most required features of a deep learning training process. The class
uses ignite as an engine to train the model. The trainer class must be implemented by a class that implements the train,
valid and test functions. Optionally, additional custom events can be specified in the add_custom_events function.
"""

from __future__ import print_function

import os
import time

import numpy as np
import torch
import visdom
from ignite.engines.engine import Events, Engine
from ignite.handlers import ModelCheckpoint, EarlyStopping
from torch.utils.data.sampler import SequentialSampler

from data.data_loader import get_train_valid_data
from ignite_features.handlers import SaveBestScore
from ignite_features.log_handlers import LogTrainProgressHandler, LogEpochMetricHandler
from ignite_features.metric import ValueEpochMetric, ValueIterMetric, TimeMetric, EntropyEpochMetric, \
    ActivationEpochMetric
from ignite_features.plot_handlers import VisEpochPlotter, VisIterPlotter
from utils import get_device, flex_profile, get_logger, calc_entropy


class Trainer:

    """ Abstract Trainer class.

    Helper class to support a ignite training process. Call run to start training. The main tasks are:
        - init visdom
        - set seed
        - log model architecture and parameters to file or console
        - limit train / valid samples in debug mode
        - split train data into train and validation
        - load model if required
        - init train and validate ignite engines
        - sets main metrics (both iter and epoch): loss and acc
        - add default events: model saving (each epoch), early stopping, log training progress
        - calls the validate engine after each training epoch, which runs one epoch.

    When extending this class, implement the following functions:
    - _add_custom_events: function adds events specific to a child class to the engines.
    - _train_function: executes a training step. It takes the ignite engine, this class and the current batch as
        arguments. Should return a dict with keys:
            - 'loss': metric of this class
            - 'acc': metric of this class
            - any key that is expected by the custom events of the child class
    - _validate_function: same as _train_function, but for validate

    Args:
        model (_Net): model/network to be trained.
        loss (_Loss): loss of the model
        optimizer (Optimizer): optimizer used in gradient update
        dataset (Dataset): dataset of torch.Dataset class
        conf (Namespace): configuration obtained using configurations.general_confs.get_conf
    """

    def __init__(self, model, loss, optimizer, data_train, data_test, conf):

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.conf = conf
        self.device = get_device()

        self._log = get_logger(__name__)
        
        # init visdom
        if conf.use_visdom:
            self.vis = visdom.Visdom()

            # start visdom if in conf
            if conf.start_visdom:

                import subprocess
                import sys

                # if visdom connection exists kill it
                if self.vis.check_connection():
                    self._log.info("Existing visdom connection found. Killed it.")

                    # kill process with process name containing 'visdom.server'.
                    subprocess.Popen(["pkill", "-f", ".*visdom\.server.*"])
                else:
                    self._log.info("No visdom connection found. Starting visdom.")

                # create visdom enviroment path if not exists
                if not os.path.exists(conf.exp_path):
                    os.makedirs(conf.exp_path)

                subprocess.Popen([f"{sys.executable}", "-m", "visdom.server", "-logging_level", "50", "-env_path",
                                  conf.exp_path])

                retries = 0
                while (not self.vis.check_connection()) and retries < 10:
                    retries += 1
                    time.sleep(1)

                if self.vis.check_connection():
                    self._log.info("Succesfully started Visdom.")
                else:
                    raise RuntimeError("Could not start Visdom")

            # if use existing connection
            elif self.vis.check_connection():
                self._log.info("Use existing Visdom connection")

            # if no connection and not start
            else:
                raise RuntimeError("Start visdom manually or set start_visdom to True")
        else:
            self.vis = None

        # print number of parameters in model
        num_parameters = np.sum([np.prod(list(p.shape)) for p in model.parameters()])
        self._log.info("Number of parameters model: {}".format(num_parameters))
        self._log.info("Model architecture: \n" + str(model))

        # init data sets
        cuda_kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}
        kwargs = {**cuda_kwargs, 'train_max': 4, 'valid_max': 2} if conf.debug else cuda_kwargs
        kwargs = {**kwargs, "seed": conf.seed} if conf.seed else kwargs
        self.train_loader, self.val_loader = get_train_valid_data(data_train, valid_size=conf.valid_size, batch_size=conf.batch_size,
                                                                  drop_last=conf.drop_last,
                                                                  shuffle=conf.shuffle, **kwargs)
        test_debug_sampler = SequentialSampler(list(range(3 * conf.batch_size))) if conf.debug else None
        self.test_loader = torch.utils.data.DataLoader(data_test, batch_size=conf.batch_size, drop_last=conf.drop_last,
                                                       sampler=test_debug_sampler, **cuda_kwargs)

        # model to cude if device is gpu
        model.to(self.device)

        # optimize cuda
        torch.backends.cudnn.benchmark = conf.cudnn_benchmark

        # load model
        if conf.load_model:
            if os.path.isfile(conf.model_load_path):
                if torch.cuda.is_available():
                    model = torch.load(conf.model_load_path)
                else:
                    model = torch.load(conf.model_load_path, map_location=lambda storage, loc: storage)
                self._log.info("Loaded model.")
            else:
                self._log.info("No model to load found. Start training new model.")

        # init an ignite engine for each data set
        self.train_engine = Engine(self._train_function, self)
        self.valid_engine = Engine(self._valid_function, self)
        self.test_engine = Engine(self._test_function, self)

        # add train metrics
        ValueIterMetric(lambda x: x["loss"]).attach(self.train_engine, "batch_loss")  # for plot and progress log
        ValueIterMetric(lambda x: x["acc"]).attach(self.train_engine, "batch_acc")  # for plot and progress log

        # add visdom plot for the training loss
        training_loss_plot = VisIterPlotter(self.vis, "batch_loss", "Loss", "Training Batch Loss", self.conf.model_name)
        self.train_engine.add_event_handler(Events.ITERATION_COMPLETED, training_loss_plot)

        # add visdom plot for the training accuracy
        training_acc_plot = VisIterPlotter(self.vis, "batch_acc", "Acc", "Training Batch Acc", self.conf.model_name)
        self.train_engine.add_event_handler(Events.ITERATION_COMPLETED, training_acc_plot)

        # add logs handlers, requires the batch_loss and batch_acc metrics
        self.train_engine.add_event_handler(Events.ITERATION_COMPLETED, LogTrainProgressHandler())

        # add valid metrics
        ValueEpochMetric(lambda x: x["acc"]).attach(self.valid_engine, "acc")  # for plot and logging
        ValueEpochMetric(lambda x: x["loss"]).attach(self.valid_engine, "loss")  # for plot, logging and early stopping

        # add valid plots and logger
        # self.valid_engine.add_event_handler(Events.EPOCH_COMPLETED,
        #                                     VisEpochPlotter(self.vis, "acc", "Acc", "Validation Acc"))
        self.valid_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                            LogEpochMetricHandler('Validation set: {:.4f}', "acc"))

        # print end testing
        self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, lambda _: self._log.info("Done testing"))

        # saves models
        if conf.save_trained:
            save_path = f"{conf.exp_path}/{conf.trained_model_path}"
            save_handler = ModelCheckpoint(save_path, conf.model_name,
                                           score_function=lambda engine: engine.state.metrics["acc"],
                                           n_saved=conf.n_saved,
                                           require_empty=False)
            self.valid_engine.add_event_handler(Events.EPOCH_COMPLETED, save_handler, {'': model})

        # add events custom events of the child class
        self._add_custom_events()

        # add early stopping, use total loss over epoch, stop if no improvement: higher score = better
        if conf.early_stop:
            early_stop_handler = EarlyStopping(patience=1,
                                               score_function=lambda engine: -engine.state.metrics["loss"],
                                               trainer=self.train_engine)
            self.valid_engine.add_event_handler(Events.COMPLETED, early_stop_handler)

        # set epoch in state of train_engine to model epoch at start to resume training for loaded model.
        # Note: new models have epoch = 0.
        @self.train_engine.on(Events.STARTED)
        def update_epoch(engine):
            engine.state.epoch = model.epoch

        # update epoch of the model, to make sure the is correct of resuming training
        @self.train_engine.on(Events.EPOCH_COMPLETED)
        def update_model_epoch(_):
            model.epoch += 1

        # makes sure eval_engine is started after train epoch, should be after all custom train_engine epoch_completed
        # events
        @self.train_engine.on(Events.EPOCH_COMPLETED)
        def call_valid(_):
            self.valid_engine.run(self.val_loader)

        @self.train_engine.on(Events.ITERATION_COMPLETED)
        def check_nan(_):
            assert all([torch.isnan(p).nonzero().shape == torch.Size([0]) for p in model.parameters()]), \
                "Parameters contain NaNs. Occurred in this iteration."

        # makes sure test_engine is started after train epoch, should be after all custom valid_engine epoch_completed
        # events
        @self.valid_engine.on(Events.EPOCH_COMPLETED)
        def call_test(_):
            self.test_engine.run(self.test_loader)

        # make that epoch in valid_engine and test_engine gives correct epoch (same train_engine was during run),
        # but makes sure only runs once
        @self.valid_engine.on(Events.STARTED)
        @self.test_engine.on(Events.STARTED)
        def set_train_epoch(engine):
            engine.state.epoch = self.train_engine.state.epoch - 1
            engine.state.max_epochs = self.train_engine.state.epoch

        # Save the visdom environment
        @self.test_engine.on(Events.EPOCH_COMPLETED)
        def save_visdom_env(_):
            if isinstance(self.vis, visdom.Visdom):
                self.vis.save([self.conf.model_name])

    def run(self):
        """ Start the training process. """
        self.train_engine.run(self.train_loader, max_epochs=self.conf.epochs)

    def _add_custom_events(self):
        raise NotImplementedError("Please implement abstract function _add_custom_events.")

    @staticmethod
    def _train_function(engine, trainer, batch):
        raise NotImplementedError("Please implement abstract function _train_function.")

    @staticmethod
    def _valid_function(engine, trainer, batch):
        raise NotImplementedError("Please implement abstract function _valid_function.")

    @staticmethod
    def _test_function(engine, trainer, batch):
        raise NotImplementedError("Please implement abstract function _test_function.")


class CapsuleTrainer(Trainer):
    """ Trainer of a capsule network.


    """

    @staticmethod
    @flex_profile
    def _train_function(engine, trainer, batch):
        trainer.model.train()
        trainer.optimizer.zero_grad()

        data = batch[0].to(trainer.device)
        labels = batch[1].to(trainer.device)

        logits, recon, _, entropy, _ = trainer.model(data, labels)

        total_loss, margin_loss, recon_loss, entropy_loss = trainer.loss(data, labels, logits, recon, entropy)

        acc = trainer.model.compute_acc(logits, labels)

        total_loss.backward()
        trainer.optimizer.step()

        return {"loss": total_loss.item(), "time": (time.time(), data.shape[0]), "acc": acc.item()}

    @staticmethod
    def _valid_function(engine, trainer, batch):

        model = trainer.model
        sparse = trainer.conf.sparse
        model.eval()

        with torch.no_grad():
            data = batch[0].to(trainer.device)
            labels = batch[1].to(trainer.device)

            sparse.set_off()

            logits, recon, _, entropy, _ = model(data)
            loss, _, _, _ = trainer.loss(data, labels, logits, recon, entropy)

            acc = trainer.model.compute_acc(logits, labels).item()

            sparse.set_on()

            # compute acc on validation set with no sparsity on inference
            if trainer.conf.sparse.is_sparse:

                logits_sparse, _, _, _, _ = model(data)
                acc_sparse = model.compute_acc(logits_sparse, labels).item()

            else:
                acc_sparse = None

        return {"loss": loss.item(), "acc": acc, "epoch": trainer.model.epoch, "acc_sparse": acc_sparse}

    @staticmethod
    def _test_function(engine, trainer, batch):

        model = trainer.model
        sparse = trainer.conf.sparse
        model.eval()

        with torch.no_grad():
            data = batch[0].to(trainer.device)
            labels = batch[1].to(trainer.device)

            sparse.set_off()

            logits, _, _, entropy, activations = model(data)

            # none sparse metrics
            acc = model.compute_acc(logits, labels).item()
            entropy = entropy.data
            prob_h = calc_entropy(model.compute_probs(logits), dim=1).mean().item()

            sparse.set_on()

            if trainer.conf.sparse.is_sparse:

                logits_sparse, _, _, entropy_sparse, activations_sparse = model(data)

                # sparse metrics
                acc_sparse = model.compute_acc(logits_sparse, labels).item()
                entropy_sparse = entropy_sparse.data
                prob_h_sparse = calc_entropy(model.compute_probs(logits_sparse), dim=1).mean().item()

            else:
                acc_sparse = entropy_sparse = prob_h_sparse = activations_sparse = None

        # return test_dict
        return {"acc": acc, "acc_sparse": acc_sparse, "entropy": entropy, "entropy_sparse": entropy_sparse,
                "prob_h": prob_h, "prob_h_sparse": prob_h_sparse, "activations": activations,
                "activations_sparse": activations_sparse}

    def _add_custom_events(self):

        # bool indicating whether method is sparse
        is_sparse = self.conf.sparse.is_sparse

        # name of sparse method
        sparse_name = self.conf.sparse.sparse_str

        # legend of the visdom plot, order should be identical to metric names
        legend = [f"{sparse_name}_no", sparse_name] if is_sparse else None

        # Add metric and to track the acc and the entropy of the logits
        ValueEpochMetric(lambda x: x["acc"]).attach(self.test_engine, "acc")
        ValueEpochMetric(lambda x: x["prob_h"]).attach(self.test_engine, "prob_h")

        if self.conf.compute_activation:
            # Add activiation metric, number of capsules in second layer is first of all_but_prim
            num_caps_layer2 = self.conf.architecture.all_but_prim[0].caps
            activation_metric = ActivationEpochMetric(lambda x: x["activations"], num_caps_layer2)
            activation_metric.attach(self.test_engine, "activations")

        # Add metric to track entropy
        caps_sizes = self.model.caps_sizes
        entropy_metric = EntropyEpochMetric(lambda x: x["entropy"], caps_sizes, self.conf.routing_iters)
        entropy_metric.attach(self.test_engine, "entropy")

        # if sparsity method is used, plot both with and without using this method on inference
        if is_sparse:

            # Add acc sparse metric to the validation engine
            ValueEpochMetric(lambda x: x["acc_sparse"]).attach(self.valid_engine, "acc_sparse")

            # Add acc and logit entropy metric to test engine
            ValueEpochMetric(lambda x: x["acc_sparse"]).attach(self.test_engine, "acc_sparse")
            ValueEpochMetric(lambda x: x["prob_h_sparse"]).attach(self.test_engine, "prob_h_sparse")

            # Add metric to track entropy
            entropy_metric_sparse = EntropyEpochMetric(lambda x: x["entropy_sparse"], caps_sizes,
                                                self.conf.routing_iters)
            entropy_metric_sparse.attach(self.test_engine, "entropy_sparse")

        # plot entropy of the softmax of logits
        prob_h = VisEpochPlotter(vis=self.vis,
                                 metric_names=["prob_h", "prob_h_sparse"] if is_sparse else "prob_h" ,
                                 ylabel="H",
                                 title="Entropy of the softmaxed logits",
                                 env_name=self.conf.model_name,
                                 legend=legend)
        self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, prob_h)

        # Add metric and plots to track the mean entropy of the weights after routing
        # if sparsity method is used, plot both with and without using this method on inference
        # plot the entropy at the last routing iteration, the last index is routing_iters-1
        entropy_plot = VisEpochPlotter(vis=self.vis,
                                 metric_names=["entropy", "entropy_sparse"] if is_sparse else "entropy",
                                 ylabel="H",
                                 title="Average Entropy (after routing)",
                                 env_name=self.conf.model_name,
                                 legend=legend,
                                 transform=lambda h: h["avg"][-1]) # transform selects the entropy at the last rout iter
        self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, entropy_plot)

        if self.conf.compute_activation:
            activations_plot = VisEpochPlotter(vis=self.vis,
                                     metric_names=["activations"],
                                     ylabel="||v||",
                                     title="Activations in 2th layer",
                                     env_name=self.conf.model_name,
                                     legend=list(range(num_caps_layer2)),
                                     use_metric_list=True)
            self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, activations_plot)

        # test acc plot
        acc_test_plot = VisEpochPlotter(vis=self.vis,
                                          metric_names=["acc", "acc_sparse"] if is_sparse else "acc",
                                          ylabel="acc",
                                          title="Test Accuracy",
                                          env_name=self.conf.model_name,
                                          legend=legend)
        self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, acc_test_plot)

        # valid acc plot
        acc_valid_plot = VisEpochPlotter(vis=self.vis,
                                        metric_names=["acc", "acc_sparse"] if is_sparse else "acc",
                                        ylabel="acc",
                                        title="Valid Accuracy",
                                        env_name=self.conf.model_name,
                                        legend=legend)
        self.valid_engine.add_event_handler(Events.EPOCH_COMPLETED, acc_valid_plot)

        # print ms per training example
        if self.conf.print_time:
            TimeMetric(lambda x: x["time"]).attach(self.train_engine, "time")
            self.train_engine.add_event_handler(Events.EPOCH_COMPLETED, LogEpochMetricHandler(
                'Time per example: {:.6f} ms', "time"))

        # save test acc of the best validition epoch to file
        if self.conf.save_best:

            # add _no to models where sparsify is turned off
            no_sparse_name = self.conf.model_name + "_no" if is_sparse else self.conf.model_name

            # Add score handler for the default inference: on valid and test the same sparsity as during training
            best_score_handler = SaveBestScore(score_valid_func=lambda engine: engine.state.metrics["acc"],
                                            score_test_func=lambda engine: engine.state.metrics["acc"],
                                            max_train_epochs=self.conf.epochs,
                                            model_name= no_sparse_name,
                                            sparse=sparse_name,
                                            score_file_name=self.conf.score_file_name,
                                            root_path=self.conf.exp_path)
            self.valid_engine.add_event_handler(Events.EPOCH_COMPLETED, best_score_handler.update_valid)
            self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, best_score_handler.update_test)

            # Add score handler for no sparsity during inference (if applied on training)
            if is_sparse:
                ValueEpochMetric(lambda x: x["acc_sparse"]).attach(self.valid_engine, "acc_sparse")
                best_score_handler = SaveBestScore(score_valid_func=lambda engine: engine.state.metrics["acc_sparse"],
                                            score_test_func=lambda engine: engine.state.metrics["acc_sparse"],
                                            max_train_epochs=self.conf.epochs,
                                            model_name=self.conf.model_name,
                                            sparse=sparse_name,
                                            score_file_name=self.conf.score_file_name,
                                            root_path=self.conf.exp_path)
                self.valid_engine.add_event_handler(Events.EPOCH_COMPLETED, best_score_handler.update_valid)
                self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, best_score_handler.update_test)

        # saves models best sparse model as well (sparse on
        if self.conf.save_trained and is_sparse:
            save_path = f"{self.conf.exp_path}/{self.conf.trained_model_path}"
            save_handler = ModelCheckpoint(save_path, f"{self.conf.model_name}_best_sparse",
                                           score_function=lambda engine: engine.state.metrics["acc_sparse"],
                                           n_saved=self.conf.n_saved,
                                           require_empty=False)
            self.valid_engine.add_event_handler(Events.EPOCH_COMPLETED, save_handler, {'': self.model})


#TODO remove?
class CNNTrainer(Trainer):

    @staticmethod
    def _train_function(engine, trainer, batch):
        trainer.model.train()
        trainer.optimizer.zero_grad()

        data = batch[0].to(trainer.device)
        labels = batch[1].to(trainer.device)

        logits = trainer.model(data)
        acc = trainer.model.compute_acc(logits, labels)

        loss = trainer.loss(logits, labels)

        loss.backward()
        trainer.optimizer.step()
        return {"loss": loss.item(), "time": (time.time(), data.shape[0]), "acc": acc}

    @staticmethod
    def _valid_function(engine, trainer, batch):
        trainer.model.eval()

        with torch.no_grad():
            data = batch[0].to(trainer.device)
            labels = batch[1].to(trainer.device)

            class_probs = trainer.model(data)

            loss = trainer.loss(class_probs, labels)
            acc = trainer.model.compute_acc(class_probs, labels)

        return {"loss": loss.item(), "acc": acc, "epoch": trainer.model.epoch}

    def _add_custom_events(self):

        if self.conf.print_time:
            TimeMetric(lambda x: x["time"]).attach(self.train_engine, "time")
            self.train_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                      VisEpochPlotter(self.vis, "time", "Time in s", "Time per sample", self.conf.model_name))
            self.train_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                      LogEpochMetricHandler('Time per example: {:.2f} sec', "time"))
