""" Plot handlers

This module all classes used to generate visdom plots. The abstract VisPlotter is implemented by two classes to plot
values every epoch or every iteration.

"""

import numpy as np
import visdom
from abc import abstractmethod


class VisPlotter:
    """Abstract visdom plot classes.

    Class to easily make visdom plots. To avoid heaps of if statements in the other code the use of visdom is
    checked in this class. If vis is None, nothing is done. Flex profile showed that the first vis.line call takes a
    lot of time, so while testing ise recommended to set use_visdom to False in the config.

    Args:
        vis (Visdom): Connected visdom instance.
        metric_names (str or list of str): String of the metric name or a list of strings of multiple metric names.
        ylabel (str): Label of the y-axis.
        title (str): Title of the plot.
        env_name (str): Visdom enviroment name.
        legend (list of str, optional): Names in the legend. Defaults to None.
        transform: (callable, optional): Transformation applied to the metric. Defaults to a unit transformation.
        use_metric_list (bool, optional): If True, VisPlotter expects the metric to point to a numpy.ndarray. If False, it expect
            point to a float. Default to Fals.
    """

    def __init__(self, vis, metric_names, ylabel, title, env_name, legend=None, transform=lambda x:x,
                 use_metric_list=False):

        # initializing visdom plots is slow. To avoid if statements for every added plot vis  can be None
        if vis:
            assert isinstance(vis, visdom.Visdom), "If vis is not None, a Visdom instance should be given as argument"
            assert not use_metric_list or not len(metric_names) > 1, "If use_metric_list, only one metric must be given."
            self.use_visdom = True
            self.vis = vis
            self.metric_names = metric_names if isinstance(metric_names, list) else [metric_names]
            self.legend = {"legend": legend} if legend else {}
            self.env = env_name
            self.transform = transform

            if use_metric_list:
                # the metric plots multiple lines using 1 metric, retrieve the number of lines
                # form the size of the legend
                assert legend is not None, "If use_metric_list, a legend must be given."
                self.num_lines = len(legend)
            else:
                # if metric represents a float, get the number of lines from the number of metrics
                self.num_lines = len(self.metric_names)

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

                # get the metric from the engine
                y_raw = engine.state.metrics[metric_name]

                # apply the given transformation to the metric
                y = self.transform(y_raw)

                #
                if type(y) == np.ndarray:
                    y_list = y

                    assert y.shape == (self.num_lines,), "The length of the input list should equal the number of lines."

                    if len(self.metric_names) > 1:
                        raise ValueError("If plot input is a list, only one metric can be given.")

                else:
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
    """Plot every iteration. """

    def get_x_label(self):
        return "# Itertations"

    @staticmethod
    def get_x(state):
        return state.iteration


class VisEpochPlotter(VisPlotter):
    """Plot every epoch. """

    def get_x_label(self):
        return "# Epochs"

    @staticmethod
    def get_x(state):
        return state.epoch
