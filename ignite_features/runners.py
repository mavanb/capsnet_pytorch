from __future__ import print_function

import visdom
import torch
import numpy as np
from ignite.trainer import Trainer
from ignite.evaluator import Evaluator
from data_loader import get_train_valid_data


def default_run(conf, dataset, model, train_function, validate_function, add_events):

    # init ignite
    vis = visdom.Visdom()
    if not vis.check_connection():
        raise RuntimeError("Visdom server not running.")

    if conf.seed:
        torch.manual_seed(conf.seed)
        np.random.seed(conf.seed)

    # init data sets
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    kwargs = {**kwargs, 'train_max': 4, 'valid_max': 10} if conf.debug == True else kwargs
    kwargs = {**kwargs, "seed" : conf.seed} if conf.seed else kwargs
    train_loader, val_loader = get_train_valid_data(dataset, batch_size=conf.batch_size, **kwargs)

    if torch.cuda.is_available():
        model.cuda()

    if conf.load:
        model = torch.load(conf.model_checkpoint_path)

    trainer = Trainer(train_function)
    trainer.current_epoch = model.epoch
    evaluator = Evaluator(validate_function)

    add_events(trainer, evaluator, train_loader, val_loader, vis)

    # kick everything off
    trainer.run(train_loader, max_epochs=conf.epochs)
