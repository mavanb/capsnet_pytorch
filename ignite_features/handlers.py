import logging
import os
from ignite.engines import Engine


class SaveBestScore:
    """SaveBestScore handler can be used to save the best score to file (high score is considered better).
    """

    def __init__(self, score_function, max_train_epochs, model_name, score_file_name, root_path="."):
        assert callable(score_function), "Argument score_function should be a function"
        self.score_function = score_function
        self. max_train_epochs = max_train_epochs
        self.model_name = model_name
        self.best_score = None
        self._logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._logger.addHandler(logging.NullHandler())

        self.file_path = f"{root_path}/{score_file_name}.csv"

        if not os.path.isfile(self.file_path):
            with open(self.file_path, 'w') as outf:
                outf.write(f"ModelName;EpochBest;EpochsTotal;Score\n")

    def __call__(self, engine):
        score = self.score_function(engine)

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = engine.state.epoch
        elif score > self.best_score:
            self.best_score = score
            self.best_epoch = engine.state.epoch

        # if final epoch, save best score to file
        if engine.state.epoch == self.max_train_epochs:
            self._logger.info("Save best score to file")

            with open(self.file_path, 'a') as outf:
                outf.write(f"{self.model_name};{self.best_epoch};{self.max_train_epochs};{self.best_score}\n")

