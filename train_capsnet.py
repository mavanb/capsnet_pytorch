from __future__ import print_function

import time

# ignite import
from ignite.engine import Events
from ignite.handlers.evaluate import Evaluate
from ignite.handlers.logging import log_simple_moving_average
from torchvision import transforms

from configurations.config_utils import get_conf_logger
from data.data_loader import get_dataset
from ignite_features.excessive_testing import excessive_testing_handler
from ignite_features.handlers import *
from ignite_features.runners import default_run
from loss import CapsuleLoss
# own imports
from nets import BasicCapsNet
from utils import variable, flex_profile


def custom_args(parser):
    parser.add('--basic_capsnet_config', is_config_file=True, default="configurations/basic_capsnet.conf",
          help='configurations file path')
    parser.add_argument('--model_name', type=str, default="simple_caps_net", help='Name of the model.')
    parser.add_argument('--alpha', type=float, required=True, help="Alpha of CapsuleLoss")
    parser.add_argument('--m_plus', type=float, required=True, help="m_plus of margin loss")
    parser.add_argument('--m_min', type=float, required=True, help="m_min of margin loss")
    parser.add_argument('--prim_caps', type=int, required=True, help="Number of primary capsules")
    parser.add_argument('--routing_iters', type=int, required=True, help="Number of iterations in the routing algo.")
    parser.add_argument('--dataset', type=str, required=True, help="Either mnist or cifar10")
    parser.add_argument('--squash_dim', type=int, required=True, help="")
    parser.add_argument('--softmax_dim', type=int, required=True, help="")
    parser.add_argument('--stdev_W', type=float, required=True, help="stddev of W of capsule layer")
    parser.add_argument('--bias_routing', type=bool, default=False, help="whether to use bias in routing")
    parser.add_argument('--excessive_testing', type=bool, default=False, help="Do excessive tests on test set")
    return parser


def main():
    conf, logger = get_conf_logger(custom_args=custom_args)

    transform = transforms.ToTensor()
    dataset, data_shape = get_dataset(conf.dataset, transform=transform)

    model = BasicCapsNet(in_channels=data_shape[0], digit_caps=10, vec_len_prim=8, vec_len_digit=16,
                         routing_iters=conf.routing_iters, prim_caps=conf.prim_caps, in_height=data_shape[1],
                         in_width=data_shape[2], softmax_dim=conf.softmax_dim, squash_dim=conf.squash_dim,
                         stdev_W=conf.stdev_W, bias_routing=conf.bias_routing)

    capsule_loss = CapsuleLoss(conf.m_plus, conf.m_min, conf.alpha, num_classes=10)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    @flex_profile
    def train_function(batch):
        model.train()
        optimizer.zero_grad()

        data = variable(batch[0])
        labels = variable(batch[1])

        class_probs, reconstruction, _ = model(data, labels)

        loss, margin_loss, recon_loss = capsule_loss(data, labels, class_probs, reconstruction)

        acc = model.compute_acc(class_probs, labels)

        loss.backward()
        optimizer.step()

        return loss.data[0], margin_loss.data[0], recon_loss.data[0], (time.time(), data.shape[0]), acc

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
                                  window_size=5,
                                  metric_name="NLL",
                                  history_transform=lambda x: x[0],
                                  should_log=lambda trainer: trainer.current_iteration % conf.log_interval == 0,
                                  logger=logger)
        trainer.add_event_handler(Events.ITERATION_COMPLETED,
                                  log_simple_moving_average,
                                  window_size=5,
                                  metric_name="Accuracy",
                                  history_transform=lambda x: x[4],
                                  should_log=lambda trainer: trainer.current_iteration % conf.log_interval == 0,
                                  logger=logger)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, vis_training_loss_handler(vis,
                                                            plot_every=conf.log_interval, transform=lambda x: x[0]))

        if conf.print_time:
            trainer.add_event_handler(Events.EPOCH_COMPLETED, time_logger_handler(logger, transform=lambda x: x[3]))
        trainer.add_event_handler(Events.EPOCH_COMPLETED, epoch_update, model)
        if conf.excessive_testing:
            trainer.add_event_handler(Events.EPOCH_COMPLETED, excessive_testing_handler(vis, conf, 3), model)
        trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                  Evaluate(evaluator, val_loader, epoch_interval=1, clear_history=False))

        # evaluator event handlers
        evaluator.add_event_handler(Events.COMPLETED, get_log_validation_loss_and_accuracy_handler(logger), model)
        evaluator.add_event_handler(Events.COMPLETED, get_plot_validation_accuracy_handler(vis), trainer, model)
        evaluator.add_event_handler(Events.COMPLETED, early_stop_and_save_handler(conf, logger), model)


    default_run(logger, conf, dataset, model, train_function, validate_function, add_events)


if __name__ == "__main__":
    main()
