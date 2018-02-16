from __future__ import print_function

# pytorch imports
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch
import visdom

# ignite import
from ignite.trainer import Trainer
from ignite.evaluator import Evaluator
from ignite.engine import Events
from ignite.handlers.evaluate import Evaluate
from ignite.handlers.logging import log_simple_moving_average

# custom ignite features
from ignite_features.handlers import *

# model imports
from nets import BasicCapsNet
from utils import variable
from loss import CapsuleLoss

from data_loader import get_train_valid_data


def run(conf, logger):

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
    train_loader, val_loader = get_train_valid_data(MNIST(download=False, root="./mnist", transform=ToTensor(),
                                                          train=True), batch_size=conf.batch_size, **kwargs)

    # init basic capsnet model
    model = BasicCapsNet(in_channels=1, digit_caps=10, vec_len_prim=8, vec_len_digit=16, routing_iters=3, prim_caps=32,
                         in_height=28, in_width=28)
    capsule_loss = CapsuleLoss(conf.m_plus, conf.m_min, conf.alpha, num_classes=10)

    optimizer = torch.optim.Adam(model.parameters())

    if torch.cuda.is_available():
        model.cuda()

    if conf.load:
        model = torch.load(conf.model_checkpoint_path)

    def training_update_function(batch):
        model.train()
        optimizer.zero_grad()

        data = variable(batch[0])
        labels = variable(batch[1])

        class_probs, reconstruction, _ = model(data, labels)

        loss, margin_loss, recon_loss = capsule_loss(data, labels, class_probs, reconstruction)

        loss.backward()
        optimizer.step()
        return loss.data[0], margin_loss.data[0], recon_loss.data[0]

    def validation_inference_function(batch):
        model.eval()

        data = variable(batch[0], volatile=True)
        labels = variable(batch[1])

        class_probs, reconstruction, _ = model(data)
        loss, _, _ = capsule_loss(data, labels, class_probs, reconstruction)

        acc = model.compute_acc(class_probs, labels)

        return loss.data[0], acc, model.epoch

    trainer = Trainer(training_update_function)
    trainer.current_epoch = model.epoch
    evaluator = Evaluator(validation_inference_function)

    # trainer event handlers
    trainer.add_event_handler(Events.ITERATION_COMPLETED,
                              log_simple_moving_average,
                              window_size=100,
                              metric_name="NLL",
                              history_transform=lambda x: x[0],
                              should_log=lambda trainer: trainer.current_iteration % conf.log_interval == 0,
                              logger=logger)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, get_plot_training_loss_handler(vis,
                                                                plot_every=conf.log_interval, transform=lambda x:x[0]))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, epoch_update, model)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, Evaluate(evaluator, val_loader, epoch_interval=1, clear_history=False))

    # evaluator event handlers
    evaluator.add_event_handler(Events.COMPLETED, get_log_validation_loss_and_accuracy_handler(logger), model)
    evaluator.add_event_handler(Events.COMPLETED, get_plot_validation_accuracy_handler(vis), trainer, model)
    evaluator.add_event_handler(Events.COMPLETED, early_stop_and_save_handler(conf), model)

    # kick everything off
    trainer.run(train_loader, max_epochs=conf.epochs)


if __name__ == "__main__":

    import configargparse
    import configurations.general_confs
    from configurations.config_utils import parse, get_logger

    # get general configs
    p = configargparse.get_argument_parser()

    # set module configs
    p.add('--basic_capsnet_config', is_config_file=True, default="configurations/basic_capsnet.conf",
          help='configurations file path')
    p.add_argument('--alpha', type=float, required=True, help="Alpha of CapsuleLoss")
    p.add_argument('--m_plus', type=float, required=True, help="m_plus of margin loss")
    p.add_argument('--m_min', type=float, required=True, help="m_min of margin loss")

    conf = parse(p)
    logger = get_logger(conf.log_file)

    # combined configs
    conf.model_checkpoint_path = "{}/{}".format(conf.trained_model_path, conf.model_name)

    # log configurations summary
    logger(p.format_values())

    run(conf, logger)
