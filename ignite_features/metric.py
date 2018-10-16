""" Metrics

This module contains all implementations of the ignite Metric class.
"""

from __future__ import division

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
from ignite.engines import Events
import numpy as np


class IterMetric(Metric):
    """ Abstract class of iteration Metric

    Metric that is computed and reset at the end of every iteration instead after each epoch.
    """

    def attach(self, engine, name):
        engine.add_event_handler(Events.ITERATION_STARTED, self.started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)


class ValueEpochMetric(Metric):
    """ Metric that computes single value every epoch.

    Calculates the average of some value that must be average of the number of batches per epoch.
    """

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        self._sum += output
        self._num_examples += 1.0

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'ValueMetric must have received at least one batch before it can be computed')
        return self._sum / self._num_examples


class ValueIterMetric(ValueEpochMetric, IterMetric):
    """ ValueMetric with is computed and reset at every iteration instead of epoch."""
    pass


class EntropyEpochMetric(Metric):
    """ Entropy metric per epoch

    Entropy per layer per routing iter and entropy average correct for capsule size per routing iter.
    """
    def __init__(self, output_transform, sizes, iters):

        self.sizes = sizes.cpu().numpy()
        self.iters = iters
        self.num_layers = len(self.sizes)

        self._layers = np.zeros((self.num_layers, self.iters))
        self._num_examples = 0

        super().__init__(output_transform)

    def reset(self):
        self._layers.fill(0)
        self._num_examples = 0

    def update(self, entropy):

        # take mean over the batch index convert to numpy
        entropy = entropy.mean(dim=1).cpu().numpy()

        self._layers += entropy
        self._num_examples += 1.0

    def compute(self):

        layers = self._layers / self._num_examples

        weights = self.sizes / sum(self.sizes)
        average = (layers * weights.reshape(-1, 1)).sum(axis=0)

        return {"layers": layers, "avg": average}


class ActivationEpochMetric(Metric):
    """Average activations of the second layer """

    def __init__(self, output_transform, num_capsules):

        self._batch_avg = np.zeros(num_capsules)
        self._num_examples = 0

        super().__init__(output_transform)

    def reset(self):
        self._batch_avg.fill(0)
        self._num_examples = 0

    def update(self, activation):

        # take mean over the batch index convert to numpy
        activation = activation.mean(dim=0).cpu().numpy()

        self._batch_avg += activation
        self._num_examples += 1.0

    def compute(self):
        avg = self._batch_avg / self._num_examples
        return avg


class EntropyIterMetric(EntropyEpochMetric, IterMetric):
    pass


class TimeMetric(Metric):
    """ Metric that calculated the average time computation per sample over an epoch."""

    def reset(self):
        self._avg_diff = 0.0
        self._prev_time = 0.0
        self._num_examples = 0

    def update(self, output):
        # convert time to ms
        new_time = output[0] * 1000

        if self._prev_time:
            batch_size = output[1]
            new_diff = (new_time - self._prev_time)
            total = self._num_examples + batch_size
            # _avg_diff gives time per sample. Thus, to update we compute the weighted average:
            # avg_diff * (num_examples / total) + avg_new_diff * (batch_size / total)
            # avg_new_diff * batch_size = (new_diff / batch_size) * batch_size = new_diff
            self._avg_diff = (self._avg_diff * self._num_examples + new_diff) / total
            self._num_examples += batch_size
        self._prev_time = new_time

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'TimeMetric must have received at least one batch before it can be computed')
        return self._avg_diff


