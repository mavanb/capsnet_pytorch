from abc import abstractmethod


class LogMetricHandler:
    def __init__(self, message, metric_name, log_every=1):
        self.log_every = log_every
        self.message = message
        self.metric_name = metric_name

    def __call__(self, engine):
        assert self.metric_name in engine.state.metrics.keys(), \
            f"Engine state does not contain the metric {self.metric_name}"
        should_log, output = self.get_output(engine)
        if should_log:
            print(self.message.format(output))

    @abstractmethod
    def get_output(self, engine):
        raise NotImplementedError()


class LogIterMetricHandler(LogMetricHandler):

    def get_output(self, engine):
        return engine.state.iteration % self.log_every == 0, engine.state.metrics[self.metric_name]


class LogEpochMetricHandler(LogMetricHandler):
    def get_output(self, engine):
        return engine.state.iteration % self.log_every == 0, engine.state.metrics[self.metric_name]


class LogTrainProgressHandler:
    def __call__(self, engine):
        # -1 and +1 because counting from 1, to make sure it prints 1/X and X/X
        iteration_in_epoch = ((engine.state.iteration - 1) % len(engine.state.dataloader)) + 1

        # new line if last iteration
        end = "\n" if len(engine.state.dataloader) == iteration_in_epoch else ""

        # new line before new iteration
        start = "\n" if iteration_in_epoch == 1 else ""

        print(f"\r{start}Epoch[{engine.state.epoch}/{engine.state.max_epochs}] "
                    f"Iteration[{iteration_in_epoch}/{len(engine.state.dataloader)}] "
                    f"Acc: {engine.state.metrics['batch_acc']:.3f} "
                    f"Loss: {engine.state.metrics['batch_loss']:.3f}", end=end)


