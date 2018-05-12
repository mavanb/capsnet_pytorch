from __future__ import print_function
from ignite_features.excessive_testing import excessive_testing_handler

import os
import numpy as np
import torch
import visdom
from ignite.handlers import ModelCheckpoint, EarlyStopping
from data.data_loader import get_train_valid_data
from ignite.engines.engine import Events, Engine
from ignite_features.metric import ValueMetric, ValueIterMetric, TimeMetric
from ignite_features.plot_handlers import VisEpochPlotter, VisIterPlotter
from ignite_features.log_handlers import LogTrainProgressHandler, LogEpochMetricHandler
from utils import get_device, flex_profile
import logging
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

    def __init__(self, model, loss, optimizer, dataset, conf):

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.conf = conf
        self.device = get_device()

        self._log = logging.getLogger(__name__ + "." + self.__class__.__name__)
        
        # init visdom
        self.vis = visdom.Visdom()
        if not self.vis.check_connection():
            raise RuntimeError("Visdom server not running.")

        if conf.seed:
            torch.manual_seed(conf.seed)
            np.random.seed(conf.seed)

        # print number of parameters in model
        num_parameters = np.sum([np.prod(list(p.shape)) for p in model.parameters()])
        self._log.info("Number of parameters model: {}".format(num_parameters))
        self._log.info("Model architecture: \n" + str(model))

        # init data sets
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        kwargs = {**kwargs, 'train_max': 4, 'valid_max': 2} if conf.debug else kwargs
        kwargs = {**kwargs, "seed": conf.seed} if conf.seed else kwargs
        self.train_loader, self.val_loader = get_train_valid_data(dataset, batch_size=conf.batch_size,
                                                                  drop_last=conf.drop_last,
                                                                  shuffle=conf.shuffle, **kwargs)

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
        self.eval_engine = Engine(self._validate_function, self)

        # add train metrics
        ValueIterMetric(lambda x: x["loss"]).attach(self.train_engine, "batch_loss")  # for plot and progress log
        ValueIterMetric(lambda x: x["acc"]).attach(self.train_engine, "batch_acc")  # for plot and progress log

        # add train plots
        if conf.plot_train_progress:
            self.train_engine.add_event_handler(Events.ITERATION_COMPLETED,
                                                VisIterPlotter(self.train_engine, self.vis, "batch_loss", "Loss",
                                                          "Training Batch Loss"))
            self.train_engine.add_event_handler(Events.ITERATION_COMPLETED,
                                                VisIterPlotter(self.train_engine, self.vis, "batch_acc", "Acc",
                                                          "Training Batch Acc"))

        # add logs handlers, requires batch_loss and batch_acc metrics
        self.train_engine.add_event_handler(Events.ITERATION_COMPLETED, LogTrainProgressHandler())

        # add eval metrics
        ValueMetric(lambda x: x["acc"]).attach(self.eval_engine, "acc")  # for plot and logging
        ValueMetric(lambda x: x["loss"]).attach(self.eval_engine, "loss")  # for plot, logging and early stopping

        # add eval plots
        if conf.plot_eval_acc:
            self.eval_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                            VisEpochPlotter(self.eval_engine, self.vis, "acc", "Acc", "Validation Acc"))
        self.eval_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                           LogEpochMetricHandler('Validation set: {:.2f}', "acc"))

        # add events custom the events
        self._add_custom_events()

        # add early stopping, use total loss over epoch.
        if conf.early_stop:
            early_stop_handler = EarlyStopping(patience=1,
                                               score_function=lambda engine: engine.state.metrics["loss"],
                                               trainer=self.train_engine)
            self.eval_engine.add_event_handler(Events.COMPLETED, early_stop_handler)

        # saves models
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
        def call_evaluator(_):
            self.eval_engine.run(self.val_loader)

        # make that epoch in eval_engine gives correct epoch (same train_engine was during run), but makes sure only
        # runs once
        @self.eval_engine.on(Events.STARTED)
        def set_train_epoch(engine):
            engine.state.epoch = self.train_engine.state.epoch - 1
            engine.state.max_epochs = self.train_engine.state.epoch

    def run(self):
        self.train_engine.run(self.train_loader, max_epochs=self.conf.epochs)

    def _add_custom_events(self):
        raise NotImplementedError("Please implement abstract function _add_custom_events.")

    @staticmethod
    def _train_function(engine, trainer, batch):
        raise NotImplementedError("Please implement abstract function _train_function.")

    @staticmethod
    def _validate_function(engine, trainer, batch):
        raise NotImplementedError("Please implement abstract function _validate_function.")


class CapsuleTrainer(Trainer):

    @flex_profile
    @staticmethod
    def _train_function(engine, trainer, batch):
        trainer.model.train()
        trainer.optimizer.zero_grad()

        data = batch[0].to(trainer.device)
        labels = batch[1].to(trainer.device)

        class_probs, reconstruction, _, rout_stats = trainer.model(data, labels)

        total_loss, margin_loss, recon_loss = trainer.loss(data, labels, class_probs, reconstruction)

        acc = trainer.model.compute_acc(class_probs, labels)

        total_loss.backward()
        trainer.optimizer.step()

        return {"loss": total_loss.item(), "time": (time.time(), data.shape[0]), "acc": acc, "rout_stats":
            rout_stats}

    @staticmethod
    def _validate_function(engine, trainer, batch):
        trainer.model.eval()

        with torch.no_grad():
            data = batch[0].to(trainer.device)
            labels = batch[1].to(trainer.device)

            class_probs, reconstruction, _, _ = trainer.model(data)
            total_loss, _, _ = trainer.capsule_loss(data, labels, class_probs, reconstruction)

            acc = trainer.model.compute_acc(class_probs, labels)

        return {"loss": total_loss.item(), "acc": acc, "epoch": trainer.model.epoch}

    def _add_custom_events(self):

        if self.conf.plot_mask_rato and self.conf.sparsify:
            # add metric tracking mask rato per epoch
            ValueMetric(lambda x: x["rout_stats"]["mask_rato"]).attach(self.train_engine, "mask_rato_epoch")

            # plot per epoch
            self.train_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                                VisEpochPlotter(self.train_engine, self.vis, "mask_rato_epoch", "Ratio",
                                                      "Mask Ratio per epoch"))

            # tracking mask per iter
            ValueIterMetric(lambda x: x["rout_stats"]["mask_rato"]).attach(self.train_engine, "mask_rato_iter")

            # plot per iter
            self.train_engine.add_event_handler(Events.ITERATION_COMPLETED,
                                                VisIterPlotter(self.train_engine, self.vis, "mask_rato_iter", "Ratio",
                                                     "Mask Ratio per iteration"))

        if self.conf.plot_deviations and self.conf.sparsify:
            # track maximum negative deviation per iter
            ValueIterMetric(lambda x: x["rout_stats"]["max_neg_devs"]).attach(self.train_engine, "max_neg_devs_iter")

            # plot metric per iter
            self.train_engine.add_event_handler(Events.ITERATION_COMPLETED,
                                                VisIterPlotter(self.train_engine, self.vis, "max_neg_devs_iter", "Ratio"
                                                               ,"Max neg devs per iteration"))

            # track average negative deviation per iter
            ValueIterMetric(lambda x: x["rout_stats"]["avg_neg_devs"]).attach(self.train_engine, "avg_neg_devs_iter")

            # plot metric per iter
            self.train_engine.add_event_handler(Events.ITERATION_COMPLETED,
                                                VisIterPlotter(self.train_engine, self.vis, "avg_neg_devs_iter", "Ratio"
                                                               , "Avg neg devs per iteration"))

        if self.conf.print_time:
            TimeMetric(lambda x: x["time"]).attach(self.train_engine, "time")
            self.train_engine.add_event_handler(Events.EPOCH_COMPLETED, VisEpochPlotter(self.train_engine, self.vis,
                                                                                "time", "Time in s", "Time per sample"))
            self.train_engine.add_event_handler(Events.EPOCH_COMPLETED, LogEpochMetricHandler(
                'Time per example: {:.4f} sec', "time"))

        if self.conf.excessive_testing:
            self.train_engine.add_event_handler(Events.EPOCH_COMPLETED, excessive_testing_handler(self.vis, self.conf, 3),
                                                self.model)


class CNNTrainer(Trainer):

    @flex_profile
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
    def _validate_function(engine, trainer, batch):
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
                                      VisEpochPlotter(self.train_engine, self.vis, "time", "Time in s", "Time per sample"))
            self.train_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                      LogEpochMetricHandler('Time per example: {:.2f} sec', "time"))