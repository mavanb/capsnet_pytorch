import torch
import time
from ignite_features.trainer import Trainer
from utils import calc_entropy
from ignite_features.metrics import ValueEpochMetric, TimeMetric, EntropyEpochMetric, \
    ActivationEpochMetric
from ignite_features.plot_handlers import VisEpochPlotter
from ignite_features.log_handlers import  SaveBestScore, LogEpochMetricHandler
from ignite.engine.engine import Events
from ignite.handlers import ModelCheckpoint


class CapsuleTrainer(Trainer):
    """ Trainer of a capsule network.
    """

    def _train_function(self, engine, batch):
        self.model.train()
        self.optimizer.zero_grad()

        data = batch[0].to(self.device)
        labels = batch[1].to(self.device)

        logits, recon, _, entropy, _ = self.model(data, labels)

        total_loss, margin_loss, recon_loss, entropy_loss = self.loss(data, labels, logits, recon, entropy)

        acc = self.model.compute_acc(logits, labels)

        total_loss.backward()
        self.optimizer.step()

        return {"loss": total_loss.item(), "time": (time.time(), data.shape[0]), "acc": acc.item()}

    def _valid_function(self, engine, batch):

        model = self.model
        sparse = self.conf.sparse
        model.eval()

        with torch.no_grad():
            data = batch[0].to(self.device)
            labels = batch[1].to(self.device)

            sparse.set_off()

            logits, recon, _, entropy, _ = model(data)
            loss, _, _, _ = self.loss(data, labels, logits, recon, entropy)

            acc = self.model.compute_acc(logits, labels).item()

            sparse.set_on()

            # compute acc on validation set with no sparsity on inference
            if self.conf.sparse.is_sparse:

                logits_sparse, _, _, _, _ = model(data)
                acc_sparse = model.compute_acc(logits_sparse, labels).item()

            else:
                acc_sparse = None

        return {"loss": loss.item(), "acc": acc, "epoch": self.model.epoch, "acc_sparse": acc_sparse}

    def _test_function(self, engine, batch):

        model = self.model
        sparse = self.conf.sparse
        model.eval()

        with torch.no_grad():
            data = batch[0].to(self.device)
            labels = batch[1].to(self.device)

            sparse.set_off()

            logits, _, _, entropy, activations = model(data)

            # none sparse metrics
            acc = model.compute_acc(logits, labels).item()
            entropy = entropy.data
            prob_h = calc_entropy(model.compute_probs(logits), dim=1).mean().item()

            sparse.set_on()

            if self.conf.sparse.is_sparse:

                logits_sparse, _, _, entropy_sparse, activations_sparse = model(data)

                # sparse metrics
                acc_sparse = model.compute_acc(logits_sparse, labels).item()
                entropy_sparse = entropy_sparse.data
                prob_h_sparse = calc_entropy(model.compute_probs(logits_sparse), dim=1).mean().item()

            else:
                acc_sparse = entropy_sparse = prob_h_sparse = activations_sparse = None

        # return test_dict
        return {"acc": acc, "acc_sparse": acc_sparse, "entropy": entropy, "entropy_sparse": entropy_sparse,
                "prob_h": prob_h, "prob_h_sparse": prob_h_sparse, "activations": activations,
                "activations_sparse": activations_sparse}

    def _add_custom_events(self):

        # bool indicating whether method is sparse
        is_sparse = self.conf.sparse.is_sparse

        # name of sparse method
        sparse_name = self.conf.sparse.sparse_str

        # legend of the visdom plot, order should be identical to metric names
        legend = [f"{sparse_name}_no", sparse_name] if is_sparse else None

        # Add metric and to track the acc and the entropy of the logits
        ValueEpochMetric(lambda x: x["acc"]).attach(self.test_engine, "acc")
        ValueEpochMetric(lambda x: x["prob_h"]).attach(self.test_engine, "prob_h")

        if self.conf.compute_activation:
            # Add activiation metric, number of capsules in second layer is first of all_but_prim
            num_caps_layer2 = self.conf.architecture.all_but_prim[0].caps
            activation_metric = ActivationEpochMetric(lambda x: x["activations"], num_caps_layer2)
            activation_metric.attach(self.test_engine, "activations")

        # Add metric to track entropy
        caps_sizes = self.model.caps_sizes
        entropy_metric = EntropyEpochMetric(lambda x: x["entropy"], caps_sizes, self.conf.routing_iters)
        entropy_metric.attach(self.test_engine, "entropy")

        # if sparsity method is used, plot both with and without using this method on inference
        if is_sparse:

            # Add acc sparse metric to the validation engine
            ValueEpochMetric(lambda x: x["acc_sparse"]).attach(self.valid_engine, "acc_sparse")

            # Add acc and logit entropy metric to test engine
            ValueEpochMetric(lambda x: x["acc_sparse"]).attach(self.test_engine, "acc_sparse")
            ValueEpochMetric(lambda x: x["prob_h_sparse"]).attach(self.test_engine, "prob_h_sparse")

            # Add metric to track entropy
            entropy_metric_sparse = EntropyEpochMetric(lambda x: x["entropy_sparse"], caps_sizes,
                                                self.conf.routing_iters)
            entropy_metric_sparse.attach(self.test_engine, "entropy_sparse")

        # plot entropy of the softmax of logits
        prob_h = VisEpochPlotter(vis=self.vis,
                                 metric=["prob_h", "prob_h_sparse"] if is_sparse else "prob_h" ,
                                 y_label="H",
                                 title="Entropy of the softmaxed logits",
                                 env_name=self.conf.model_name,
                                 legend=legend,
                                 input_type="multiple" if is_sparse else "single")
        self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, prob_h)

        # Add metric and plots to track the mean entropy of the weights after routing
        # if sparsity method is used, plot both with and without using this method on inference
        # plot the entropy at the last routing iteration, the last index is routing_iters-1
        entropy_plot = VisEpochPlotter(vis=self.vis,
                                 metric=["entropy", "entropy_sparse"] if is_sparse else "entropy",
                                 y_label="H",
                                 title="Average Entropy (after routing)",
                                 env_name=self.conf.model_name,
                                 legend=legend,
                                 transform=lambda h: h["avg"][-1], # transform selects the entropy at the last rout iter
                                 input_type="multiple" if is_sparse else "single")
        self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, entropy_plot)

        if self.conf.compute_activation:
            activations_plot = VisEpochPlotter(vis=self.vis,
                                     metric="activations",
                                     y_label="||v||",
                                     title="Activations in 2th layer",
                                     env_name=self.conf.model_name,
                                     legend=list(range(num_caps_layer2)),
                                     input_type="array")
            self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, activations_plot)

        # test acc plot
        acc_test_plot = VisEpochPlotter(vis=self.vis,
                                          metric=["acc", "acc_sparse"] if is_sparse else "acc",
                                          y_label="acc",
                                          title="Test Accuracy",
                                          env_name=self.conf.model_name,
                                          legend=legend,
                                          input_type="multiple" if is_sparse else "single")
        self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, acc_test_plot)

        # valid acc plot
        acc_valid_plot = VisEpochPlotter(vis=self.vis,
                                        metric=["acc", "acc_sparse"] if is_sparse else "acc",
                                        y_label="acc",
                                        title="Valid Accuracy",
                                        env_name=self.conf.model_name,
                                        legend=legend,
                                        input_type="multiple" if is_sparse else "single")
        self.valid_engine.add_event_handler(Events.EPOCH_COMPLETED, acc_valid_plot)

        # save test acc of the best validition epoch to file
        if self.conf.save_best:

            # add _no to models where sparsify is turned off
            no_sparse_name = self.conf.model_name + "_no" if is_sparse else self.conf.model_name

            # Add score handler for the default inference: on valid and test the same sparsity as during training
            best_score_handler = SaveBestScore(score_valid_func=lambda engine: engine.state.metrics["acc"],
                                            score_test_func=lambda engine: engine.state.metrics["acc"],
                                            start_epoch=self.model.epoch,
                                            max_train_epochs=self.conf.epochs,
                                            model_name= no_sparse_name,
                                            score_file_name=self.conf.score_file_name,
                                            root_path=self.conf.exp_path)
            self.valid_engine.add_event_handler(Events.EPOCH_COMPLETED, best_score_handler.update_valid)
            self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, best_score_handler.update_test)

            # Add score handler for no sparsity during inference (if applied on training)
            if is_sparse:
                ValueEpochMetric(lambda x: x["acc_sparse"]).attach(self.valid_engine, "acc_sparse")
                best_score_handler = SaveBestScore(score_valid_func=lambda engine: engine.state.metrics["acc_sparse"],
                                            score_test_func=lambda engine: engine.state.metrics["acc_sparse"],
                                            start_epoch=self.model.epoch,
                                            max_train_epochs=self.conf.epochs,
                                            model_name=self.conf.model_name,
                                            score_file_name=self.conf.score_file_name,
                                            root_path=self.conf.exp_path)
                self.valid_engine.add_event_handler(Events.EPOCH_COMPLETED, best_score_handler.update_valid)
                self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, best_score_handler.update_test)

        # saves models best sparse model as well (sparse on
        if self.conf.save_trained and is_sparse:
            save_path = f"{self.conf.exp_path}/{self.conf.trained_model_path}"
            save_handler = ModelCheckpoint(save_path, f"{self.conf.model_name}_best_sparse",
                                           score_function=lambda engine: engine.state.metrics["acc_sparse"],
                                           n_saved=self.conf.n_saved,
                                           require_empty=False)
            self.valid_engine.add_event_handler(Events.EPOCH_COMPLETED, save_handler, {'': self.model})
