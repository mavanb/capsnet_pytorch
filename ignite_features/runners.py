from __future__ import print_function

import visdom
import torch
import numpy as np
from ignite.trainer import Trainer
from ignite.evaluator import Evaluator
from data_loader import get_train_valid_data
import os

def default_run(logger, conf, dataset, model, train_function, validate_function, add_events):

    # init ignite
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
    kwargs = {**kwargs, 'train_max': 4, 'valid_max': 2} if conf.debug == True else kwargs
    kwargs = {**kwargs, "seed": conf.seed} if conf.seed else kwargs
    train_loader, val_loader = get_train_valid_data(dataset, batch_size=conf.batch_size, drop_last=conf.drop_last,
                                                        shuffle=conf.shuffle, **kwargs)

    if torch.cuda.is_available():
        model.cuda()

    if conf.load:
        if os.path.isfile(conf.model_load_path):
            model = torch.load(conf.model_load_path)
            logger("Loaded model.")
        else:
            logger("No model to load found. Start training new model.")

    trainer = Trainer(train_function)
    trainer.current_epoch = model.epoch
    evaluator = Evaluator(validate_function)

    add_events(trainer, evaluator, train_loader, val_loader, vis)

    # kick everything off
    trainer.run(train_loader, max_epochs=conf.epochs)
