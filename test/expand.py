"""
This module checks whether expanding the batch dimension of a weight parameter has influence on:
- training time
- performance (acc)

We apply a very simple linear model to MNIST: one linear operation with a very large number of parameters.
After calculating the backward pass the algo requires about 8gb.

We compare one model in which we expand the W to the batch size, and one model in which we don't. We train each m
model multiple times to limit stochasticity.

Conclusion: expanding has no influence on both training time and acc.
"""

import torch
from torch import nn
import time
import numpy as np
from utils import get_device
from torchvision import transforms
from data.data_loader import get_dataset
from nets import _Net

device = get_device()

class LinearOp(_Net):

    def __init__(self, expand):
        super().__init__()
        self.expand = expand

        self.W1 = nn.Parameter(torch.randn(1, 10000, 784))
        self.W2 = nn.Parameter(torch.randn(1, 10, 10000))

    def forward(self, input):
        input = input.view(-1, 784, 1)
        b = input.shape[0]
        if self.expand:
            W1 = self.W1.expand(b, 10000, 784)
        else:
            W1 = self.W1
        input = torch.matmul(W1, input)
        input = torch.matmul(self.W2, input)
        return input.view(-1, 10)

runs = 4
for expand in [False, True]:

    print(f"\n --- When Expand = {expand} ---- \n")

    time_list = []
    acc_list = []

    for i in range(runs):

        model = LinearOp(expand)
        model.to(get_device())

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        c_ent_loss = nn.modules.loss.CrossEntropyLoss()

        transform = transforms.ToTensor()
        traindata, testdata, data_shape, label_shape = get_dataset("mnist", transform=transform)
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        train_loader = torch.utils.data.DataLoader(traindata, batch_size=128, drop_last=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(testdata, batch_size=128, drop_last=True, **kwargs)

        for batch_idx, batch in enumerate(train_loader):
            model.train()
            start = time.time()

            data = batch[0].to(device)
            labels = batch[1].to(device)

            optimizer.zero_grad()

            predict = model(data)
            loss = c_ent_loss(predict, labels)

            loss.backward()
            optimizer.step()

            diff_time = (time.time() - start)

            print(f"\r Run: {i}/{runs} Batch: {batch_idx}/{len(train_loader)} Loss: {loss.item():0.4f}", end="")

            time_list.append(diff_time)

        for batch_idx, batch in enumerate(test_loader):
            model.eval()

            with torch.no_grad():
                data = batch[0].to(device)
                labels = batch[1].to(device)

                logits = model(data)

                loss = c_ent_loss(logits, labels)
                acc = model.compute_acc(logits, labels)
                acc_list.append(acc.item())

    print(f"\nTrain time: {np.asarray(time_list).sum()/runs:0.4f}")
    print(f"Acc test set: {np.asarray(acc_list).mean():0.4f}")

