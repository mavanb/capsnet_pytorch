from __future__ import print_function

import time


# ignite import
from ignite.engines.engine import Events
from ignite_features.log_handlers import LogEpochMetricHandler
from ignite_features.plot_handlers import VisEpochPlotter
from ignite_features.metric import TimeMetric
from ignite_features.runners import default_run

# torch import
from torchvision import transforms
import torch

# model import
from nets import BaselineCNN
from torch.nn.modules.loss import NLLLoss

# utils
from configurations.config_utils import get_conf_logger
from data.data_loader import get_dataset
from utils import get_device


def custom_args(parser):
    parser.add('--baseline_cnn_config', is_config_file=True, default="configurations/baseline_cnn.conf",
          help='configurations file path')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model.')
    parser.add_argument('--dataset', type=str, required=True, help="Either mnist or cifar10")
    parser.add_argument('--stdev_W', type=float, required=True, help="stddev of W of capsule layer")
    return parser


def main():
    conf, logger = get_conf_logger(custom_args)

    transform = transforms.ToTensor()
    dataset, data_shape, label_shape = get_dataset(conf.dataset, transform=transform)

    model = BaselineCNN(classes=10, in_channels=data_shape[0], in_height=data_shape[1], in_width=data_shape[2])

    nlll = NLLLoss()

    optimizer = torch.optim.Adam(model.parameters())

    device = get_device()

    def train_function(engine, batch):
        model.train()
        optimizer.zero_grad()

        data = batch[0].to(device)
        labels = batch[1].to(device)

        logits = model(data)
        acc = model.compute_acc(logits, labels)

        loss = nlll(logits, labels)

        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "time": (time.time(), data.shape[0]), "acc": acc}

    def validate_function(engine, batch):
        model.eval()

        with torch.no_grad():
            data = batch[0].to(device)
            labels = batch[1].to(device)

            class_probs = model(data)

            loss = nlll(class_probs, labels)
            acc = model.compute_acc(class_probs, labels)

        return {"loss": loss.item(), "acc": acc, "epoch": model.epoch}

    def add_events(trainer, evaluator, train_loader, val_loader, vis):

        if conf.print_time:
            TimeMetric(lambda x: x["time"]).attach(trainer, "time")
            trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                      VisEpochPlotter(trainer, vis, "time", "Time in s", "Time per example"))
            trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                      LogEpochMetricHandler(logger, '\nTime per example: {:.2f} sec', "time"))

    default_run(logger, conf, dataset, model, train_function, validate_function, add_events)


if __name__ == "__main__":
    main()
