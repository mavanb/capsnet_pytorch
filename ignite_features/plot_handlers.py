import numpy as np
import visdom
from abc import abstractmethod
from utils import flex_profile

class VisPlotter:
    """ Class to easily make visdom plots. To avoid heaps of if statements in the code the use of visdom is handelled
    in the class itself. If vis is None, nothing is done. Flex profile showed that the first vis.line call takes a
    lot of time.
    """

    @flex_profile
    def __init__(self, engine, vis, metric_name, xlabel, ylabel, title, plot_every):
        if vis:
            assert isinstance(vis, visdom.Visdom), "If vis is not None, a Visdom istance should be given as argument"
            self.use_visdom = True
            self.vis = vis
            self.metric_name = metric_name
            self.plot_every = plot_every

            self.win = self.vis.line(X=np.array([1]), Y=np.array([np.nan]),
                                     opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))
        else:
            self.use_visdom = False

    def __call__(self, engine):
        if self.use_visdom:
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