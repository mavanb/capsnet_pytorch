import os
import argparse
import torch
import torch.utils.data as Data
import torchvision
from nets import BasicCapsNet
from utils import variable
from loss import CapsuleLoss
import numpy as np
import time


def main(conf):



    train_data = torchvision.datasets.MNIST(root='./mnist', train=True, transform=torchvision.transforms.ToTensor(),
                                            download=True)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=conf.batch_size, shuffle=True, **kwargs)

    test_data = torchvision.datasets.MNIST(root='./mnist', train=False, transform=torchvision.transforms.ToTensor(),
                                           download=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=conf.batch_size, shuffle=False, **kwargs)

    # init basic capsnet model
    model = BasicCapsNet(in_channels=1, digit_caps=10, vec_len_prim=8, vec_len_digit=16, routing_iters=3, prim_caps=32,
                         in_height=28, in_width=28)

    if torch.cuda.is_available():
        model.cuda()

    if conf.load:
        model = torch.load(conf.model_checkpoint_path)

    optimizer = torch.optim.Adam(model.parameters())
    capsule_loss = CapsuleLoss(conf.m_plus, conf.m_min, conf.alpha, num_classes=10)

    iters = len(train_loader)
    iters_valid = len(test_loader)
    best_loss = np.infty

    # loop over epochs
    for epoch in range(model.epoch, conf.epochs):

        # loop over train batches
        start = time.time()
        for iter, (data_batch, label_batch) in enumerate(train_loader):

            if conf.debug and iter > 2:
                break

            data = variable(data_batch)
            labels = variable(label_batch)

            class_probs, reconstruction, _ = model(data, labels)

            loss, margin_loss, recon_loss = capsule_loss(data, labels, class_probs, reconstruction)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            if conf.print_time and iter % 50 == 1:
                speed = (time.time() - start) / (conf.batch_size * iter)
                print("Speed: {}".format(speed))

            print("\rEpoch: {}  Iteration: {}/{} ({:.1f}%)  Total loss: {:.5f}    Margin: {:.5f}   Recon: {:.5f} ".
                  format(epoch, iter + 1, iters, iter * 100 / iters, loss.data[0], margin_loss.data[0], recon_loss.data[0]), end="")

        # loop over valid batches, compute average loss and accuracy todo: use valid data instead of test here
        acc_values = []
        loss_values = []
        for iter, (data_batch, label_batch) in enumerate(test_loader):

            if conf.debug and iter > 2:
                break

            data = variable(data_batch)
            labels = variable(label_batch)

            class_probs, reconstruction, _ = model(data)

            acc = model.compute_acc(class_probs, labels)
            acc_values.append(acc)

            loss, _, _ = capsule_loss(data, labels, class_probs, reconstruction)
            loss_values.append(loss.data[0])
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(iter + 1, iters_valid, iter * 100 / iters_valid),
                  end=" " * 10)

        acc_mean = np.mean(acc_values)
        loss_mean = np.mean(loss_values)

        print("\rEpoch: {}  Validation accuracy: {:.4f}% {}".format(epoch, acc_mean * 100,
                                                                    " (loss improved)" if loss_mean < best_loss else ""))

        # store epoch
        model.epoch += 1

        # early stop after one non-improved epoch
        if loss_mean < best_loss:
            best_loss = loss_mean

            if conf.save_trained:
                if not os.path.exists(conf.trained_model_path):
                    os.makedirs(conf.trained_model_path)
                torch.save(model, conf.model_checkpoint_path)
        else:
            break


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=None, help="Torch and numpy random seed. To ensure repeatability.")
    parser.add_argument('--trained_model_path', type=str, default="./trained_models", help='Path of checkpoints.')
    parser.add_argument('--save_trained', type=bool, default=True, help='Save fully trained model for inference.')
    parser.add_argument('--model_name', type=str, default="simple", help='Name of the model.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size.')
    parser.add_argument('--model_type', type=str, default="simple_caps_net", help='simple_caps_net or toy_caps_net')
    parser.add_argument('--debug', type=bool, default=False, help="debug mode: break early")
    parser.add_argument('--print_time', type=bool, default=False, help="print train time per sample")
    parser.add_argument('--load', type=bool, default=False, help="load previously trained model")

    # loss params
    parser.add_argument('--alpha', type=float, default=0.025, help="Alpha of CapsuleLoss")
    parser.add_argument('--m_plus', type=float, default=0.9, help="m_plus of margin loss")
    parser.add_argument('--m_min', type=float, default=0.1, help="m_min of margin loss")

    config = parser.parse_args()

    # combined configs
    config.model_checkpoint_path = "{}/{}".format(config.trained_model_path, config.model_name)

    # temp configs
    # configurations.epochs = 3

    main(config)

