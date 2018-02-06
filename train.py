import torch
import torch.utils.data as Data
import torchvision
import config as conf
from nets import BasicCapsNet
from utils import variable
from loss import CapsuleLoss
import numpy as np

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

train_data = torchvision.datasets.MNIST(root='./mnist', train=True, transform=torchvision.transforms.ToTensor(),
                                        download=True)
train_loader = Data.DataLoader(dataset=train_data, batch_size=conf.batch_size, shuffle=True)


test_data = torchvision.datasets.MNIST(root='./mnist', train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=conf.batch_size, shuffle=True)


model = BasicCapsNet(in_channels=1, digit_caps=10, vec_len_prim=8, vec_len_digit=16, routing_iters=3, prim_caps=32,
                     in_height=28, in_width=28)
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)
capsule_loss = CapsuleLoss(conf.m_plus, conf.m_min, conf.alpha, num_classes=10)

iters = len(train_loader)
iters_valid = len(test_loader)
best_loss = np.infty

for epoch in range(conf.epochs):

    for iter, (data_batch, label_batch) in enumerate(train_loader):

        data = variable(data_batch)
        labels = variable(label_batch)

        class_probs, reconstruction = model(data, labels, train=True)

        loss = capsule_loss(data, labels, class_probs, reconstruction)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        print("\rEpoch: {}  Iteration: {}/{} ({:.1f}%)  Total loss: {:.5f}".
              format(epoch, iter + 1, iters, iter * 100 / iters, loss.data[0]), end="")

    acc_values = []
    loss_values = []
    for iter, (data_batch, label_batch) in enumerate(test_loader):
        data = variable(data_batch)
        labels = variable(label_batch)

        class_probs, reconstruction = model(data, labels, train=False)

        acc = model.compute_acc(class_probs, labels)
        acc_values.append(acc)

        loss = capsule_loss(data, labels, class_probs, reconstruction)
        loss_values.append(loss.data[0])
        print("\rEvaluating the model: {}/{} ({:.1f}%)".format(iter + 1, iters_valid, iter * 100 / iters_valid),
              end=" " * 10)

    acc_mean = np.mean(acc_values)
    loss_mean = np.mean(loss_values)

    print("\rEpoch: {}  Validation accuracy: {:.4f}% {}".format(epoch, acc_mean * 100,
                                                                " (loss improved)" if loss_mean < best_loss else ""))

    if loss_mean < best_loss:
        best_loss = loss_mean


