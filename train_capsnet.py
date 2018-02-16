from __future__ import print_function

# pytorch imports
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# ignite import
from ignite.engine import Events
from ignite.handlers.evaluate import Evaluate
from ignite.handlers.logging import log_simple_moving_average

# custom ignite features
from ignite_features.handlers import *

# model imports
from nets import BasicCapsNet
from utils import variable
from loss import CapsuleLoss
from configurations.config_utils import get_conf_logger

from ignite_features.runners import default_run


def custom_args(parser):
    parser.add('--basic_capsnet_config', is_config_file=True, default="configurations/basic_capsnet.conf",
          help='configurations file path')
    parser.add_argument('--alpha', type=float, required=True, help="Alpha of CapsuleLoss")
    parser.add_argument('--m_plus', type=float, required=True, help="m_plus of margin loss")
    parser.add_argument('--m_min', type=float, required=True, help="m_min of margin loss")
    return parser


def main():
    conf, logger = get_conf_logger(custom_args=custom_args)

    dataset = MNIST(download=False, root="./mnist", transform=ToTensor(), train=True)

    model = BasicCapsNet(in_channels=1, digit_caps=10, vec_len_prim=8, vec_len_digit=16, routing_iters=3, prim_caps=32,
                         in_height=28, in_width=28)
    capsule_loss = CapsuleLoss(conf.m_plus, conf.m_min, conf.alpha, num_classes=10)

    optimizer = torch.optim.Adam(model.parameters())

    def train_function(batch):
        model.train()
        optimizer.zero_grad()

        data = variable(batch[0])
        labels = variable(batch[1])

        class_probs, reconstruction, _ = model(data, labels)

        loss, margin_loss, recon_loss = capsule_loss(data, labels, class_probs, reconstruction)

        loss.backward()
        optimizer.step()
        return loss.data[0], margin_loss.data[0], recon_loss.data[0]

    def validate_function(batch):
        model.eval()

        data = variable(batch[0], volatile=True)
        labels = variable(batch[1])

        class_probs, reconstruction, _ = model(data)
        loss, _, _ = capsule_loss(data, labels, class_probs, reconstruction)

        acc = model.compute_acc(class_probs, labels)

        return loss.data[0], acc, model.epoch

    def add_events(trainer, evaluator, train_loader, val_loader, vis):
        # trainer event handlers
        trainer.add_event_handler(Events.ITERATION_COMPLETED,
                                  log_simple_moving_average,
                                  window_size=100,
                                  metric_name="NLL",
                                  history_transform=lambda x: x[0],
                                  should_log=lambda trainer: trainer.current_iteration % conf.log_interval == 0,
                                  logger=logger)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, get_plot_training_loss_handler(vis,
                                                                                             plot_every=conf.log_interval,
                                                                                             transform=lambda x: x[0]))
        trainer.add_event_handler(Events.EPOCH_COMPLETED, epoch_update, model)
        trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                  Evaluate(evaluator, val_loader, epoch_interval=1, clear_history=False))

        # evaluator event handlers
        evaluator.add_event_handler(Events.COMPLETED, get_log_validation_loss_and_accuracy_handler(logger), model)
        evaluator.add_event_handler(Events.COMPLETED, get_plot_validation_accuracy_handler(vis), trainer, model)
        evaluator.add_event_handler(Events.COMPLETED, early_stop_and_save_handler(conf), model)

    default_run(conf, dataset, model, train_function, validate_function, add_events)


if __name__ == "__main__":
    main()
