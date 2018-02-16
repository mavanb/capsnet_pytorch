import numpy as np
import torch
import os


def get_plot_training_loss_handler(vis, plot_every, transform=lambda x: x, window_size=100):
    train_loss_plot_window = vis.line(X=np.array([1]), Y=np.array([np.nan]),
                                      opts=dict(
                                          xlabel='# Iterations',
                                          ylabel='Loss',
                                          title='Training Loss')
                                      )

    def plot_training_loss_to_visdom(trainer):
        if trainer.current_iteration % plot_every == 0:
            vis.line(X=np.array([trainer.current_iteration]),
                     Y=np.array([trainer.history.simple_moving_average(window_size=window_size, transform=transform)]),
                     win=train_loss_plot_window,
                     update='append')

    return plot_training_loss_to_visdom


def get_plot_validation_accuracy_handler(vis):
    val_accuracy_plot_window = vis.line(X=np.array([1]), Y=np.array([np.nan]),
                                        opts=dict(
                                            xlabel='# Epochs',
                                            ylabel='Accuracy',
                                            title='Validation Accuracy')
                                        )

    def plot_val_accuracy_to_visdom(evaluator, trainer, model):
        accuracy = sum([accuracy for (loss, accuracy, epoch) in evaluator.history if epoch is model.epoch])
        accuracy = (accuracy * 100.) / len(evaluator.dataloader.dataset)
        vis.line(X=np.array([trainer.current_epoch]),
                 Y=np.array([accuracy]),
                 win=val_accuracy_plot_window,
                 update='append')

    return plot_val_accuracy_to_visdom


def get_log_validation_loss_and_accuracy_handler(logger):
    def log_validation_loss_and_accuracy(evaluator, model):
        avg_loss = np.mean([loss for (loss, accuracy, epoch) in evaluator.history if epoch is model.epoch])
        accuracy = np.mean([accuracy for (loss, accuracy, epoch) in evaluator.history if epoch is model.epoch])
        logger('\nValidation set: Average loss: {:.4f}, {:.0f}%\n'.format(avg_loss, accuracy * 100.))

    return log_validation_loss_and_accuracy


def epoch_update(engine, model):
    model.epoch = engine.current_epoch


def early_stop_and_save_handler(conf):
    def save(model):
        if conf.save_trained:
            if not os.path.exists(conf.trained_model_path):
                os.makedirs(conf.trained_model_path)
            torch.save(model, conf.model_checkpoint_path)

    def early_stop_and_save(engine, model):
        prev_total_loss = [loss for (loss, accuracy, epoch) in engine.history if epoch is model.epoch-1]
        prev_loss = np.mean(prev_total_loss) if prev_total_loss else np.infty
        cur_loss = np.mean([loss for (loss, accuracy, epoch) in engine.history if epoch is model.epoch])
        if cur_loss < prev_loss:
            save(model)
        save(model)

    return early_stop_and_save

