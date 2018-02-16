from __future__ import print_function

# pytorch imports
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.nn.modules.loss import NLLLoss

# ignite import
from ignite.engine import Events
from ignite.handlers.evaluate import Evaluate
from ignite.handlers.logging import log_simple_moving_average

# custom ignite features
from ignite_features.handlers import *

# model imports
from nets import BaselineCNN
from utils import variable
from configurations.config_utils import get_conf_logger

from ignite_features.runners import default_run


def custom_args(parser):
    parser.add_argument('--model_name', type=str, default="baseline_cnn", help='Name of the model.')
    return parser


def main():
    conf, logger = get_conf_logger(custom_args)

    dataset = MNIST(download=False, root="./mnist", transform=ToTensor(), train=True)

    model = BaselineCNN(classes=10, in_channels=1, in_height=28, in_width=28)

    nlll = NLLLoss()

    optimizer = torch.optim.Adam(model.parameters())

    def train_function(batch):
        model.train()
        optimizer.zero_grad()

        data = variable(batch[0])
        labels = variable(batch[1])

        logits = model(data)

        loss = nlll(logits, labels)

        loss.backward()
        optimizer.step()
        return loss.data[0], None, None

    def validate_function(batch):
        model.eval()

        data = variable(batch[0], volatile=True)
        labels = variable(batch[1])

        class_probs = model(data)

        loss = nlll(class_probs, labels)
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
