from __future__ import print_function

import os
import numpy as np
import torch
import visdom
from ignite.engines import Engine
from ignite.handlers import ModelCheckpoint, EarlyStopping
from utils import get_device
from data.data_loader import get_train_valid_data
from ignite.engines.engine import Events
from ignite_features.metric import ValueMetric, ValueIterMetric, TimeMetric
from ignite_features.plot_handlers import VisEpochPlotter, VisIterPlotter
from ignite_features.log_handlers import LogTrainProgressHandler, LogEpochMetricHandler


def default_run(logger, conf, dataset, model, train_function, validate_function, add_events):
    """ Function implements the procedures that are required for training most pytorch ignite DL models.
    Train function return dict must contain: loss and acc
    Validate function return dict must contain: loss, acc and epoch

    Main steps:
    - init visdom
    - set seed
    - log model architecture and parametres
    - limit train / valid samples in debug mode
    - split train data into train and validation
    - load model if required
    - init train and validate engines
    - add default events: model saving (each epoch), early stopping, log training progress,
    - add custom events

    :param logger: logger obtained using config_utils.get_conf_logger
    :param conf: configuration obtained using config_utils.get_conf_logger
    :param dataset: dataset of torch.Dataset class
    :param model: model to be trained
    :param train_function: train function, must return dict with at least loss and acc
    :param validate_function: validation function, must return dict with at least loss, acc and epoch
    :param add_events: function that adds the custom events to trainer and evaluator when called
    """

    # init visdom
    vis = visdom.Visdom()
    if not vis.check_connection():
        raise RuntimeError("Visdom server not running.")

    if conf.seed:
        torch.manual_seed(conf.seed)
        np.random.seed(conf.seed)

    # print number of parameters in model
    num_parameters = np.sum([np.prod(list(p.shape)) for p in model.parameters()])
    logger("Number of parameters model: {}".format(num_parameters))
    logger("Model architecture: \n" + str(model))

    # init data sets
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    kwargs = {**kwargs, 'train_max': 4, 'valid_max': 2} if conf.debug else kwargs
    kwargs = {**kwargs, "seed": conf.seed} if conf.seed else kwargs
    train_loader, val_loader = get_train_valid_data(dataset, batch_size=conf.batch_size, drop_last=conf.drop_last,
                                                    shuffle=conf.shuffle, **kwargs)

    model.to(get_device())

    if conf.load_model:
        if os.path.isfile(conf.model_load_path):
            if torch.cuda.is_available():
                model = torch.load(conf.model_load_path)
            else:
                model = torch.load(conf.model_load_path, map_location=lambda storage, loc: storage)
            logger("Loaded model.")
        else:
            logger("No model to load found. Start training new model.")

    trainer = Engine(train_function)
    evaluator = Engine(validate_function)

    # add train metrics
    ValueIterMetric(lambda x: x["loss"]).attach(trainer, "batch_loss")  # for plot and progress log
    ValueIterMetric(lambda x: x["acc"]).attach(trainer, "batch_acc")    # for plot and progress log

    # add train plots
    if conf.plot_train_progress:
        trainer.add_event_handler(Events.ITERATION_COMPLETED,
                                  VisIterPlotter(trainer, vis, "batch_loss", "Loss", "Training Batch Loss"))
        trainer.add_event_handler(Events.ITERATION_COMPLETED,
                                  VisIterPlotter(trainer, vis, "batch_acc", "Acc", "Training Batch Acc"))

    # add logs handlers, requires batch_loss and batch_acc metrics
    trainer.add_event_handler(Events.ITERATION_COMPLETED, LogTrainProgressHandler(logger))

    # add eval metrics
    ValueMetric(lambda x: x["acc"]).attach(evaluator, "acc")            # for plot and logging
    ValueMetric(lambda x: x["loss"]).attach(evaluator, "loss")          # for plot, logging and early stopping

    # add eval plots
    if conf.plot_eval_acc:
        evaluator.add_event_handler(Events.EPOCH_COMPLETED,
                                    VisEpochPlotter(evaluator, vis, "acc", "Acc", "Validation Acc"))
        evaluator.add_event_handler(Events.EPOCH_COMPLETED,
                                    LogEpochMetricHandler(logger, 'Validation set: {:.2f}', "acc"))

    # add events custom the events
    add_events(trainer, evaluator, train_loader, val_loader, vis)

    # add early stopping, use total loss over epoch.
    if conf.early_stop:
        early_stop_handler = EarlyStopping(patience=1, score_function=lambda engine: engine.state.metrics["loss"],
                                           trainer=trainer, logger=logger)
        evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

    # saves models
    save_handler = ModelCheckpoint(conf.trained_model_path, conf.model_name, save_interval=1, n_saved=conf.n_saved,
                                   create_dir=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, save_handler, {'': model})

    # set epoch in state of trainer to model epoch at start to resume training for loaded model. Note: new models have
    # epoch = 0.
    @trainer.on(Events.STARTED)
    def update_epoch(engine):
        engine.state.epoch = model.epoch

    # makes sure evaluator is started after train epoch, should be after all custom trainer epoch_completed events
    @trainer.on(Events.EPOCH_COMPLETED)
    def call_evaluator(_):
        evaluator.run(val_loader)

    # make that epoch in evaluator gives correct epoch (same trainer was during run), but makes sure only runs once
    @evaluator.on(Events.STARTED)
    def set_train_epoch(engine):
        engine.state.epoch = trainer.state.epoch - 1
        engine.state.max_epochs = trainer.state.epoch

    # kick everything off
    trainer.run(train_loader, max_epochs=conf.epochs)
