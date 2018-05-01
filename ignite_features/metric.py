from __future__ import division

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
from ignite.engines import Events


class IterMetric(Metric):
    """ Abstract class of Metric that is computed and reset at the end of every iteration. """

    def attach(self, engine, name):
        engine.add_event_handler(Events.ITERATION_STARTED, self.started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)


class ValueMetric(Metric):
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


class ValueIterMetric(ValueMetric, IterMetric):
    """ ValueMetric with is computed and reset at every iteration instead of epoch."""
    pass


class TimeMetric(Metric):
    """ Metric that calculated the average time computation per sample over an epoch."""

    def reset(self):
        self._diff = 0.0
        self._prev_time = 0.0
        self._num_examples = 0

    def update(self, output):
        new_time = output[0]
        if self._prev_time:
            batch_size = output[1]
            new_diff = (new_time - self._prev_time)
            self._diff = (self._diff * self._num_examples + new_diff * batch_size) /\
                         (self._num_examples + batch_size)
            self._num_examples += batch_size
        self._prev_time = new_time

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'TimeMetric must have received at least one batch before it can be computed')
        return self._diff


