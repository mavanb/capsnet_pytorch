import numpy as np
import logging
import visdom
from abc import abstractmethod


class VisPlotter:

    def __init__(self, engine, vis, metric_name, xlabel, ylabel, title, plot_every):
        assert isinstance(vis, visdom.Visdom), "Argument engine should be an instance of Engine"
        self._logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._logger.addHandler(logging.NullHandler())
        self.vis = vis
        self.metric_name = metric_name
        self.plot_every = plot_every

        self.win = self.vis.line(X=np.array([1]), Y=np.array([np.nan]),
                                 opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))

    def __call__(self, engine):
        assert self.metric_name in engine.state.metrics.keys(), \
            f"Engine state does not contain the metric {self.metric_name}"
        X, Y, make_plot_bool = self.get_x_y(engine)

        if make_plot_bool:
            self.vis.line(X=np.array([X]), Y=np.array([Y]), win=self.win, update='append')

    @abstractmethod
    def get_x_y(self, engine):
        pass


class VisIterPlotter(VisPlotter):

    def __init__(self, engine, vis, metric_name, ylabel, title, plot_every=1, xlabel="# Iterations"):
        super().__init__(engine, vis, metric_name, xlabel, ylabel, title, plot_every)

    def get_x_y(self, engine):
        X = engine.state.iteration
        Y = engine.state.metrics[self.metric_name]
        make_plot_bool = X % self.plot_every == 0
        return X, Y, make_plot_bool


class VisEpochPlotter(VisPlotter):
    def __init__(self, engine, vis, metric_name, ylabel, title, plot_every=1, xlabel="# Epochs"):
        super().__init__(engine, vis, metric_name, xlabel, ylabel, title, plot_every)

    def get_x_y(self, engine):
        X = engine.state.epoch
        Y = engine.state.metrics[self.metric_name]
        make_plot_bool = X % self.plot_every == 0
        return X, Y, make_plot_bool