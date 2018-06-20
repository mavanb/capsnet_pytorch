from __future__ import print_function
from ignite_features.excessive_testing import excessive_testing_handler

import os
import numpy as np
import torch
import visdom
from ignite.handlers import ModelCheckpoint, EarlyStopping
from data.data_loader import get_train_valid_data
from ignite.engines.engine import Events, Engine
from ignite_features.metric import ValueEpochMetric, ValueIterMetric, TimeMetric, EntropyEpochMetric
from ignite_features.plot_handlers import VisEpochPlotter, VisIterPlotter
from ignite_features.log_handlers import LogTrainProgressHandler, LogEpochMetricHandler
from utils import get_device, flex_profile, get_logger, calc_entropy
from ignite_features.handlers import SaveBestScore
import time


class Trainer:
    """ Helper class to support a ignite training process. Call run to start training.

    Main tasks:
        - init visdom
        - set seed
        - log model architecture and parametres
        - limit train / valid samples in debug mode
        - split train data into train and validation
        - load model if required
        - init train and validate ignite engines
        - sets main metrics (both iter and epoch): loss and acc
        - add default events: model saving (each epoch), early stopping, log training progress
        - calls the validate engine after each training epoch, which runs one epoch.

    Function to be implemented:
    - _add_custom_events: function adds events specific to a child class to the engines.
    - _train_function: executes a training step. It takes the ignite engine, this class and the current batch as
        arguments. Should return a dict with keys:
            - 'loss': metric of this class
            - 'acc': metric of this class
            - any key that is expected by the custom events of the child class
    - _validate_function: same as _train_function, but for validate

    Arguments:
    :param model: model to be trained
    :param loss: loss of the model
    :param optimizer: optimizer used in gradient update
    :param dataset: dataset of torch.Dataset class
    :param conf: configuration obtained using config_utils.get_conf_logger
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

            # if no connection and should start
            if (not self.vis.check_connection()) and conf.start_visdom:
                self._log.info("No visdom connection found. Starting visdom.")
                import subprocess
                import sys

                # create visdom enviroment path if not exists
                if not os.path.exists(conf.visdom_path):
                    os.makedirs(conf.visdom_path)

                subprocess.Popen([f"{sys.executable}", "-m", "visdom.server", "-logging_level", "50", "-env_path",
                                  conf.visdom_path])

                retries = 0
                while (not self.vis.check_connection()) and retries < 10:
                    retries += 1
                    time.sleep(1)

                if self.vis.check_connection():
                    self._log.info("Succesfully started Visdom.")
                else:
                    raise RuntimeError("Could not start Visdom")

            # if no connection and shouldn't start
            elif not self.vis.check_connection():
                raise RuntimeError("Start visdom manually or set start_visdom to True")

            else:
                self._log.info("Use existing Visdom connection")
        else:
            # initializing visdom plots is slow.To avoid if statements for every added plot init vis as None
            # and handle this within the VisPlotter class.
            self.vis = None

        if conf.seed:
            torch.manual_seed(conf.seed)
            np.random.seed(conf.seed)

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
        self.test_loader = torch.utils.data.DataLoader(data_test, batch_size=conf.batch_size, drop_last=conf.drop_last,
                                                       **cuda_kwargs)

        model.to(self.device)
        torch.backends.cudnn.benchmark = conf.cudnn_benchmark

        if conf.load_model:
            if os.path.isfile(conf.model_load_path):
                if torch.cuda.is_available():
                    model = torch.load(conf.model_load_path)
                else:
                    model = torch.load(conf.model_load_path, map_location=lambda storage, loc: storage)
                self._log.info("Loaded model.")
            else:
                self._log.info("No model to load found. Start training new model.")

        self.train_engine = Engine(self._train_function, self)
        self.valid_engine = Engine(self._valid_function, self)
        self.test_engine = Engine(self._test_function, self)

        # add train metrics
        ValueIterMetric(lambda x: x["loss"]).attach(self.train_engine, "batch_loss")  # for plot and progress log
        ValueIterMetric(lambda x: x["acc"]).attach(self.train_engine, "batch_acc")  # for plot and progress log

        self.train_engine.add_event_handler(Events.ITERATION_COMPLETED,
                                            VisIterPlotter(self.vis, "batch_loss", "Loss", "Training Batch Loss", self.conf.model_name))
        self.train_engine.add_event_handler(Events.ITERATION_COMPLETED,
                                            VisIterPlotter(self.vis, "batch_acc", "Acc", "Training Batch Acc", self.conf.model_name))

        # add logs handlers, requires batch_loss and batch_acc metrics
        self.train_engine.add_event_handler(Events.ITERATION_COMPLETED, LogTrainProgressHandler())

        # add valid metrics
        ValueEpochMetric(lambda x: x["acc"]).attach(self.valid_engine, "acc")  # for plot and logging
        ValueEpochMetric(lambda x: x["loss"]).attach(self.valid_engine, "loss")  # for plot, logging and early stopping

        # add valid plots and logger
        # self.valid_engine.add_event_handler(Events.EPOCH_COMPLETED,
        #                                     VisEpochPlotter(self.vis, "acc", "Acc", "Validation Acc"))
        self.valid_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                            LogEpochMetricHandler('Validation set: {:.4f}', "acc"))

        # print start end testing
        self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, lambda _: self._log.info("Done testing"))

        # add events custom the events
        self._add_custom_events()

        # add early stopping, use total loss over epoch, stop if no improvement: higher score = better
        if conf.early_stop:
            early_stop_handler = EarlyStopping(patience=1,
                                               score_function=lambda engine: -engine.state.metrics["loss"],
                                               trainer=self.train_engine)
            self.valid_engine.add_event_handler(Events.COMPLETED, early_stop_handler)

        # saves models
        if conf.save_trained:
            save_handler = ModelCheckpoint(conf.trained_model_path, conf.model_name, save_interval=1,
                                           n_saved=conf.n_saved,
                                           create_dir=True, require_empty=False)
            self.train_engine.add_event_handler(Events.EPOCH_COMPLETED, save_handler, {'': model})

        # set epoch in state of train_engine to model epoch at start to resume training for loaded model.
        # Note: new models have epoch = 0.
        @self.train_engine.on(Events.STARTED)
        def update_epoch(engine):
            engine.state.epoch = model.epoch

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

    @staticmethod
    @flex_profile
    def _train_function(engine, trainer, batch):
        trainer.model.train()
        trainer.optimizer.zero_grad()

        data = batch[0].to(trainer.device)
        labels = batch[1].to(trainer.device)

        class_probs, reconstruction, _, _ = trainer.model(data, labels)

        total_loss, margin_loss, recon_loss = trainer.loss(data, labels, class_probs, reconstruction)

        acc = trainer.model.compute_acc(class_probs, labels)

        total_loss.backward()
        trainer.optimizer.step()

        return {"loss": total_loss.item(), "time": (time.time(), data.shape[0]), "acc": acc.item()}

    @staticmethod
    def _valid_function(engine, trainer, batch):
        trainer.model.eval()

        with torch.no_grad():
            data = batch[0].to(trainer.device)
            labels = batch[1].to(trainer.device)

            class_probs, reconstruction, _, entropy = trainer.model(data)
            total_loss, _, _ = trainer.loss(data, labels, class_probs, reconstruction)

            class_probs, reconstruction, _, entropy = trainer.model(data)
            acc = trainer.model.compute_acc(class_probs, labels)

            # compute acc on validation set with no sparsity on inference
            acc_sparse = None
            if trainer.conf.sparsify != "None":
                trainer.model.set_sparsify("None")
                class_probs, _, _, _ = trainer.model(data)
                acc_sparse = trainer.model.compute_acc(class_probs, labels)
                acc_sparse = acc_sparse.item()
                trainer.model.set_sparsify(trainer.conf.sparsify)

        return {"loss": total_loss.item(), "acc": acc.item(), "epoch": trainer.model.epoch, "acc_sparse": acc_sparse}

    @staticmethod
    def _test_function(engine, trainer, batch):

        model = trainer.model
        model.eval()

        # sparsify used during training
        train_sparsify = trainer.conf.sparsify

        # return dict
        test_dict = {}

        with torch.no_grad():
            data = batch[0].to(trainer.device)
            labels = batch[1].to(trainer.device)

            model.set_sparsify("None")

            logits, recon, caps, entropy = model(data)
            acc = model.compute_acc(logits, labels)
            probs = model.compute_probs(logits)
            prob_entropy = calc_entropy(probs, dim=1).mean().item()

            test_dict["None"] = {}
            test_dict["None"]["acc"] = acc.item()
            test_dict["None"]["entropy"] = entropy
            test_dict["None"]["prob_entropy"] = prob_entropy

            if train_sparsify != "None":

                model.set_sparsify(train_sparsify)

                logits, recon, caps, entropy = model(data)
                acc = model.compute_acc(logits, labels)
                probs = model.compute_probs(logits)
                prob_entropy = calc_entropy(probs, dim=1).mean().item()

                test_dict[train_sparsify] = {}
                test_dict[train_sparsify]["acc"] = acc.item()
                test_dict[train_sparsify]["entropy"] = entropy
                test_dict[train_sparsify]["prob_entropy"] = prob_entropy

        # set sparsity back to original setting
        model.set_sparsify(train_sparsify)

        return test_dict

    def _add_custom_events(self):

        # Add metric and plots to track the entropy of the logits
        # if sparsity method is used, plot both with and without using this method on inference
        ValueEpochMetric(lambda x: x["None"]["prob_entropy"]).attach(self.test_engine, "prob_h")
        if self.conf.sparsify != "None":
            ValueEpochMetric(lambda x: x[self.conf.sparsify]["prob_entropy"]).attach(self.test_engine, "prob_h_sparse")
        prop_names = "prob_h" if self.conf.sparsify == "None" else ["prob_h", "prob_h_sparse"]
        prop_legend = None if self.conf.sparsify == "None" else [f"{self.conf.sparsify}_no", self.conf.sparsify]
        prop_entropy_plot = VisEpochPlotter(self.vis, prop_names, "H", "Entropy of the softmaxed logits", self.conf.model_name, prop_legend)
        self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, prop_entropy_plot)

        # Add metric and plots to track the mean entropy of the weights after routing
        # if sparsity method is used, plot both with and without using this method on inference
        # plot the entropy at the last routing iteration, the last index is routing_iters-1

        caps_sizes = [l.caps for l in self.model.arch.other_layers]
        EntropyEpochMetric(lambda x: x["None"]["entropy"], caps_sizes,
                           self.conf.routing_iters).attach(self.test_engine, "entropy")
        if self.conf.sparsify != "None":
            EntropyEpochMetric(lambda x: x[self.conf.sparsify]["entropy"], caps_sizes,
                               self.conf.routing_iters).attach(self.test_engine, "entropy_sparse")

        h_names = "entropy" if self.conf.sparsify == "None" else ["entropy", "entropy_sparse"]
        h_legend = None if self.conf.sparsify == "None" else [f"{self.conf.sparsify}_no", self.conf.sparsify]
        h_plot = VisEpochPlotter(self.vis, h_names, "H", "Average Entropy (after routing)", self.conf.model_name, h_legend,
                                 lambda h: h["avg"][-1])
        self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, h_plot)

        ValueEpochMetric(lambda x: x["None"]["acc"]).attach(self.test_engine, "acc")
        if self.conf.sparsify != "None":
            ValueEpochMetric(lambda x: x[self.conf.sparsify]["acc"]).attach(self.test_engine, "acc_sparse")
        acc_weight_names = "acc" if self.conf.sparsify == "None" else ["acc", "acc_sparse"]
        acc_weight_legend = None if self.conf.sparsify == "None" else [f"{self.conf.sparsify}_no", self.conf.sparsify]
        acc_weight_plot = VisEpochPlotter(self.vis, acc_weight_names, "acc", "Test Accuracy", self.conf.model_name, acc_weight_legend)
        self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, acc_weight_plot)

        if self.conf.print_time:
            TimeMetric(lambda x: x["time"] * 1000).attach(self.train_engine, "time")
            self.train_engine.add_event_handler(Events.EPOCH_COMPLETED, LogEpochMetricHandler(
                'Time per example: {:.6f} ms', "time"))

        if self.conf.save_best:
            # Add score handler for the default inference: on valid and test the same sparsity as during training
            best_score_handler = SaveBestScore(score_valid_func=lambda engine: engine.state.metrics["acc"],
                                               score_test_func=lambda engine: engine.state.metrics["acc"],
                                               max_train_epochs=self.conf.epochs,
                                               model_name=self.conf.model_name,
                                               score_file_name=self.conf.score_file_name)
            self.valid_engine.add_event_handler(Events.EPOCH_COMPLETED, best_score_handler.update_valid)
            self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, best_score_handler.update_test)

            # Add score handler for no sparsity during inference (if applied on training)
            if self.conf.sparsify != "None":
                ValueEpochMetric(lambda x: x["acc_sparse"]).attach(self.valid_engine, "acc_sparse")
                best_score_handler = SaveBestScore(score_valid_func=lambda engine: engine.state.metrics["acc_sparse"],
                                                   score_test_func=lambda engine:
                                                   engine.state.metrics["acc_sparse"],
                                                   max_train_epochs=self.conf.epochs,
                                                   model_name=f"{self.conf.model_name}_no",
                                                   score_file_name=self.conf.score_file_name)
                self.valid_engine.add_event_handler(Events.EPOCH_COMPLETED, best_score_handler.update_valid)
                self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, best_score_handler.update_test)


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
