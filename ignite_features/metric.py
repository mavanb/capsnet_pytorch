from __future__ import division

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
from ignite.engines import Events
import numpy as np

class IterMetric(Metric):
    """ Abstract class of Metric that is computed and reset at the end of every iteration. """

    def attach(self, engine, name):
        engine.add_event_handler(Events.ITERATION_STARTED, self.started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)


class ValueEpochMetric(Metric):
    """
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

    # entropy per layer per routing iter
    # entropy average correct for size per routing iter

    def __init__(self, output_transform, sizes, iters):

        self.sizes = sizes
        self.iters = iters
        self.num_layers = len(sizes)

        self._layers = np.zeros((self.num_layers, self.iters))
        self._num_examples = 0

        super().__init__(output_transform)

    def reset(self):
        self._layers.fill(0)
        self._num_examples = 0

    def update(self, entropy_values):

        assert type(entropy_values) == list and type(entropy_values[0]) == list, "Entropy metrics expects list of list"

        assert len(entropy_values) == self.num_layers, "The entropy values a have different size than the layers."

        assert set([len(l) for l in entropy_values]) == {self.iters}

        self._layers += np.asarray(entropy_values, dtype=float)
        self._num_examples += 1.0

    def compute(self):

        layers = self._layers / self._num_examples

        weights = np.asarray(self.sizes) / sum(self.sizes)
        average = (layers * weights.reshape(-1, 1)).sum(axis=0)

        return {"layers": layers, "avg": average}


class EntropyIterMetric(EntropyEpochMetric, IterMetric):
    pass


class TimeMetric(Metric):
    """ Metric that calculated the average time computation per sample over an epoch."""

    def reset(self):
        self._avg_diff = 0.0
        self._prev_time = 0.0
        self._num_examples = 0

    def update(self, output):
        new_time = output[0]
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


