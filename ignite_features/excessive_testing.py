"""
Excessive testing is used for tests on the test set that can only be performed on during traing. This avoids
have to save all models. This module is not meant for normal testing.
"""

from __future__ import print_function
from torchvision import transforms
from data.data_loader import get_dataset
from utils import get_device
import torch
import numpy as np


def test_routing_iters_handler(routing_test_iters, vis, test_loader, conf):
    """ Test the effect of routing by looking at differences in logits for different number of routing iters."""

    value_names = [str(i + 1) for i in range(routing_test_iters)]
    value_legend = ["{} iter".format(i + 1) for i in range(routing_test_iters)]
    value_X_init = np.column_stack([0 for _ in value_names])
    value_Y_init = np.column_stack([np.nan for _ in value_names])

    win_acc = vis.line(X=value_X_init, Y=value_Y_init, name=value_names,
                               opts=dict(xlabel='Epoch', ylabel='acc', title='Accuracy on test set',
                                         legend=value_legend))

    # diff_names = [str(i + 1) for i in range(routing_test_iters -1)]
    # diff_legend = ["{}-{} diff".format(i+1, i+2) for i in range(routing_test_iters -1)]
    # diff_X_init = np.column_stack([0 for _ in diff_names])
    # diff_Y_init = np.column_stack([np.nan for _ in diff_names])

    # win_logits_max = vis.line(X=diff_X_init, Y=diff_Y_init, name=diff_names,
    #                           opts=dict(xlabel='Epoch', ylabel='max diff', title='Relative maximum logit difference.',
    #                                     legend=diff_legend))
    # win_logits_mean = vis.line(X=diff_X_init, Y=diff_Y_init, name=diff_names,
    #                            opts=dict(xlabel='Epoch', ylabel='mean diff', title='Relative mean logit difference.',
    #                                      legend=diff_legend))

    device = get_device()

    def test_routing_iters(engine, model):

        # set sparsify in model to false, else max diff is compared to zerod entries
        model.set_sparsify("None")

        with torch.no_grad():

            num_batches = len(test_loader)
            acc_values = np.zeros(shape=(num_batches, routing_test_iters))
            logits_mean_rel_diff = np.zeros(shape=(num_batches, routing_test_iters -1))
            logits_max_rel_diff = np.zeros(shape=(num_batches, routing_test_iters -1))

            for batch_idx, batch in enumerate(test_loader):

                model.eval()

                data = batch[0].to(device)
                labels = batch[1].to(device)

                for routing_iter in range(routing_test_iters):
                    model.routing_iters = routing_iter + 1
                    class_logits, recon, final_caps, stats = model(data)
                    acc = model.compute_acc(class_logits, labels)

                    acc_values[batch_idx, routing_iter] = acc

                    if routing_iter > 0:
                        diff = torch.abs((prev_logits / class_logits) - 1).data
                        logits_max_rel_diff[batch_idx, routing_iter-1] = diff.max(dim=1)[0].mean()
                        logits_mean_rel_diff[batch_idx, routing_iter-1] = diff.mean()

                    prev_logits = class_logits
                if conf.debug and batch_idx > 3:
                    break

            mean_diff = logits_mean_rel_diff.mean(axis=0)
            max_diff = logits_max_rel_diff.max(axis=0)
            acc_mean = acc_values.mean(axis=0)
            epoch = engine.state.epoch
            # vis.line(X=np.column_stack((epoch, epoch)), Y=np.column_stack(max_diff), update="append",
            #          win=win_logits_max, opts={"legend": diff_legend})
            # vis.line(X=np.column_stack((epoch, epoch)), Y=np.column_stack(mean_diff), update="append",
            #          win=win_logits_mean, opts={"legend": diff_legend})
            vis.line(X=np.column_stack((epoch, epoch, epoch)), Y=np.column_stack(acc_mean), update="append",
                     win=win_acc, opts={"legend": value_legend})

        # set sparsify back to configuration
        model.set_sparsify(conf.sparsify)
    return test_routing_iters


def excessive_testing_handler(vis, conf, routing_test_iters):
    """ Excessive testing handler. Add all tests that have to be performed to the test_func_list. """

    transform = transforms.ToTensor()
    dataset, _, data_shape, label_shape = get_dataset(conf.dataset, transform=transform)
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, drop_last=True, **kwargs)

    test_func_list = []

    # replace if multipe tests are implemented
    test_func_list.append(test_routing_iters_handler(routing_test_iters, vis, test_loader, conf))

    def excessive_testing(engine, model):
        for f in test_func_list:
            f(engine, model)
    return excessive_testing








