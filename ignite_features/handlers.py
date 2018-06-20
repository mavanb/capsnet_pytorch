import logging
import os
from ignite.engines import Engine


class SaveBestScore:
    """SaveBestScore handler can be used to save the best score to file (high score is considered better).

    Args:
        score_valid_func (callable): function to retrieve the valid score from the valid engine
        score_test_func (callable): function to retrieve the test score from the test engine
        max_train_epochs (int): number of training epochs, write best test score at max, is written to file
        model_name (str): name of the model, is written to file
            score_file_name (str): name of the file
        root_path (str, optional): path of the folder to save file
    """

    def __init__(self, score_valid_func, score_test_func, max_train_epochs, model_name, score_file_name,
                 root_path="./best_acc"):

        assert callable(score_valid_func), "Argument score_function should be a function"
        assert callable(score_test_func), "Argument score_function should be a function"

        self.score_valid_func = score_valid_func
        self.score_test_func = score_test_func

        self.max_train_epochs = max_train_epochs
        self.model_name = model_name

        # keep track of best valid score to determine best valid epoch
        self.best_valid_score = None

        # best valid epoch, used to retrieve best test epoch
        self.best_valid_epoch = None

        # array of each test score per epoch
        self.test_scores = []

        # to check whether both test engine and valid engine are called an equal amount of time and in right order
        self.valid_updates = 0
        self.test_updates = 0

        self._logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._logger.addHandler(logging.NullHandler())

        self.file_path = f"{root_path}/{score_file_name}.csv"

        if not os.path.exists(root_path):
            os.makedirs(root_path)

        if not os.path.isfile(self.file_path):
            with open(self.file_path, 'w') as outf:
                outf.write(f"ModelName;EpochBest;EpochsTotal;Score\n")

    def update_valid(self, valid_engine):
        """ Should be called by the validation engine. Determines which epoch has the best score on the
        validation set."""

        score = self.score_valid_func(valid_engine)

        if self.best_valid_score is None:
            self.best_valid_score = score
            self.best_valid_epoch = valid_engine.state.epoch
        elif score > self.best_valid_score:
            self.best_valid_score = score
            self.best_valid_epoch = valid_engine.state.epoch

        self.valid_updates += 1

    def update_test(self, test_engine):

        self.test_updates += 1
        assert self.test_updates == self.valid_updates, "Test and valid engine are not called equally or in order."

        # retrieve score from engine
        score = self.score_test_func(test_engine)

        # append score to list
        self.test_scores.append(score)

        # if final epoch, save best score to file
        if test_engine.state.epoch == self.max_train_epochs:
            self._logger.info("Save best score to file")

            # get the test score of the best validation epoch, epoch count form 1, thus
            best_valid_score = self.test_scores[self.best_valid_epoch - 1]
            with open(self.file_path, 'a') as outf:
                outf.write(
                    f"{self.model_name};{self.best_valid_epoch};{self.max_train_epochs};{best_valid_score:0.6}\n")

    def __call__(self, engine):
        score = self.score_function(engine)

        if self.best_valid_score is None:
            self.best_valid_score = score
            self.best_valid_epoch = engine.state.epoch
        elif score > self.best_valid_score:
            self.best_valid_score = score
            self.best_valid_epoch = engine.state.epoch



