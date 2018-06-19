import numpy as np
import visdom
from abc import abstractmethod
from utils import flex_profile


class VisPlotter:
    """ Class to easily make visdom plots. To avoid heaps of if statements in the code the use of visdom is handelled
    in the class itself. If vis is None, nothing is done. Flex profile showed that the first vis.line call takes a
    lot of time.
    """

    def __init__(self, vis, metric_names, ylabel, title, env_name, legend=None, transform=lambda x:x):
        if vis:
            assert isinstance(vis, visdom.Visdom), "If vis is not None, a Visdom instance should be given as argument"
            self.use_visdom = True
            self.vis = vis
            self.metric_names = metric_names if isinstance(metric_names, list) else [metric_names]
            self.num_lines = len(self.metric_names)
            self.legend = {"legend": legend} if legend else {}
            self.env = env_name
            self.transform = transform

            self.win = self.vis.line(env = env_name, X=np.ones(self.num_lines).reshape(1, -1), Y=np.zeros(self.num_lines).reshape(1, -1)
                            * np.nan, opts=dict(xlabel=self.get_x_label(), ylabel=ylabel, title=title, **self.legend))
        else:
            self.use_visdom = False

    def __call__(self, engine):
        if self.use_visdom:

            # check if all metric names are in the state
            for metric_name in self.metric_names:
                assert metric_name in engine.state.metrics.keys(), \
                    f"Engine state does not contain the metric {metric_name}"

            # repeat the X for every line/metric
            X = np.column_stack([self.get_x(engine.state) for _ in range(self.num_lines)])

            y_list = []
            for metric_name in self.metric_names:
                y_raw = engine.state.metrics[metric_name]
                y = self.transform(y_raw)
                assert isinstance(y, np.float64) or isinstance(y, float), "Y value should after transform be a float"
                y_list.append(y)

            # get Y
            Y = np.column_stack(y_list)

            # plot
            self.vis.line(env=self.env, X=X, Y=Y, win=self.win, update='append', opts=self.legend)

    @abstractmethod
    def get_x_label(self):
        pass

    @staticmethod
    @abstractmethod
    def get_x(state):
        pass


class VisIterPlotter(VisPlotter):

    def get_x_label(self):
        return "# Itertations"

    @staticmethod
    def get_x(state):
        return state.iteration


class VisEpochPlotter(VisPlotter):

    def get_x_label(self):
        return "# Epochs"

    @staticmethod
    def get_x(state):
        return state.epoch
