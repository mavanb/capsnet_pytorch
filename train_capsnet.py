from __future__ import print_function

import time

# ignite import
from ignite.engines.engine import Events
from ignite_features.log_handlers import LogEpochMetricHandler
from ignite_features.plot_handlers import VisEpochPlotter, VisIterPlotter
from ignite_features.metric import ValueMetric, TimeMetric, ValueIterMetric
from ignite_features.excessive_testing import excessive_testing_handler
from ignite_features.runners import default_run

# torch import
import torch
from torchvision import transforms

# model import
from nets import BasicCapsNet
from loss import CapsuleLoss

# utils
from utils import get_device, flex_profile
from configurations.config_utils import get_conf_logger, parse_bool
from data.data_loader import get_dataset


def custom_args(parser):
    parser.add('--basic_capsnet_config', is_config_file=True, default="configurations/basic_capsnet.conf",
          help='configurations file path')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model.')
    parser.add_argument('--alpha', type=float, required=True, help="Alpha of CapsuleLoss")
    parser.add_argument('--m_plus', type=float, required=True, help="m_plus of margin loss")
    parser.add_argument('--m_min', type=float, required=True, help="m_min of margin loss")
    parser.add_argument('--prim_caps', type=int, required=True, help="Number of primary capsules")
    parser.add_argument('--routing_iters', type=int, required=True, help="Number of iterations in the routing algo.")
    parser.add_argument('--dataset', type=str, required=True, help="Either mnist or cifar10")
    parser.add_argument('--squash_dim', type=int, required=True, help="")
    parser.add_argument('--softmax_dim', type=int, required=True, help="")
    parser.add_argument('--stdev_W', type=float, required=True, help="stddev of W of capsule layer")
    parser.add_argument('--bias_routing', type=parse_bool, required=True, help="whether to use bias in routing")
    parser.add_argument('--excessive_testing', type=parse_bool, required=True, help="Do excessive tests on test set")
    parser.add_argument('--track_mask_rato', type=parse_bool, required=True, help="Check mask rato")
    parser.add_argument('--sparse_threshold', type=float, required=True, help="Threshold of routing to sparsify.")
    parser.add_argument('--sparsify', type=parse_bool, required=True, help="Whether or not to sparsify parse tree.")
    return parser


def main():
    conf, logger = get_conf_logger(custom_args=custom_args)

    transform = transforms.ToTensor()
    dataset, data_shape, label_shape = get_dataset(conf.dataset, transform=transform)

    model = BasicCapsNet(in_channels=data_shape[0], digit_caps=label_shape, vec_len_prim=8, vec_len_digit=16,
                         routing_iters=conf.routing_iters, prim_caps=conf.prim_caps, in_height=data_shape[1],
                         in_width=data_shape[2], softmax_dim=conf.softmax_dim, squash_dim=conf.squash_dim,
                         stdev_W=conf.stdev_W, bias_routing=conf.bias_routing, sparse_threshold=conf.sparse_threshold,
                         sparsify=conf.sparsify)

    capsule_loss = CapsuleLoss(conf.m_plus, conf.m_min, conf.alpha, num_classes=label_shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    device = get_device()

    @flex_profile
    def train_function(engine, batch):
        model.train()
        optimizer.zero_grad()

        data = batch[0].to(device)
        labels = batch[1].to(device)

        class_probs, reconstruction, _, rout_stats = model(data, labels)

        total_loss, margin_loss, recon_loss = capsule_loss(data, labels, class_probs, reconstruction)

        acc = model.compute_acc(class_probs, labels)

        total_loss.backward()
        optimizer.step()

        return {"loss": total_loss.item(), "time": (time.time(), data.shape[0]), "acc": acc, "rout_stats":
            rout_stats}

    def validate_function(engine, batch):
        model.eval()

        with torch.no_grad():
            data = batch[0].to(device)
            labels = batch[1].to(device)

            class_probs, reconstruction, _, _ = model(data)
            total_loss, _, _ = capsule_loss(data, labels, class_probs, reconstruction)

            acc = model.compute_acc(class_probs, labels)

        return {"loss": total_loss.item(), "acc": acc, "epoch": model.epoch}

    def add_events(trainer, evaluator, train_loader, val_loader, vis):

        if conf.track_mask_rato:
            # add metric tracking mask rato per epoch
            ValueMetric(lambda x: x["rout_stats"]["mask_rato"]).attach(trainer, "mask_rato_epoch")

            # plot per epoch
            trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                VisEpochPlotter(trainer, vis, "mask_rato_epoch", "Ratio", "Mask Ratio per epoch"))

            # tracking mask per iter
            ValueIterMetric(lambda x: x["rout_stats"]["mask_rato"]).attach(trainer, "mask_rato_iter")

            # plot per iter
            trainer.add_event_handler(Events.ITERATION_COMPLETED,
                                VisIterPlotter(trainer, vis, "mask_rato_iter", "Ratio", "Mask Ratio per iteration"))

        # track maximum negative deviation per iter
        ValueIterMetric(lambda x: x["rout_stats"]["max_neg_devs"]).attach(trainer, "max_neg_devs_iter")

        # plot metric per iter
        trainer.add_event_handler(Events.ITERATION_COMPLETED,
                                  VisIterPlotter(trainer, vis, "max_neg_devs_iter", "Ratio", "Max neg devs per iteration"))

        # track average negative deviation per iter
        ValueIterMetric(lambda x: x["rout_stats"]["avg_neg_devs"]).attach(trainer, "avg_neg_devs_iter")

        # plot metric per iter
        trainer.add_event_handler(Events.ITERATION_COMPLETED,
                                  VisIterPlotter(trainer, vis, "avg_neg_devs_iter", "Ratio", "Avg neg devs per iteration"))

        if conf.print_time:
            TimeMetric(lambda x: x["time"]).attach(trainer, "time")
            trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                      VisEpochPlotter(trainer, vis, "time", "Time in s", "Time per sample"))
            trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                      LogEpochMetricHandler(logger, 'Time per example: {:.2f} sec', "time"))

        if conf.excessive_testing:
            trainer.add_event_handler(Events.EPOCH_COMPLETED, excessive_testing_handler(vis, conf, 3), model)

    default_run(logger, conf, dataset, model, train_function, validate_function, add_events)


if __name__ == "__main__":
    main()
