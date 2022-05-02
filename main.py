from net import NN
from utils import *
import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt


def data_generation():
    '''
    This function is used to generate the data.
    The input data is a 1 by 10 vector.
    One of the value in the vector will be changed to 1, others remain 0.
    If the index of value 1 is even, the label will be [1, 0].
    If the index of value 0 is even, the label will be [0, 1].
    :return: the one-hot data vector, the label
    '''
    data = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    index = np.random.randint(0, 10)
    data[index] = 1
    if index % 2 == 0:
        label = torch.tensor([1, 0], dtype=torch.float32)
    else:
        label = torch.tensor([0, 1], dtype=torch.float32)

    return torch.as_tensor(data, dtype=torch.float32), label


def train(net):
    loss_rec = []
    step = []
    for i in range(5000):
        train, label = data_generation()
        optimizer.zero_grad()
        output = net(train)
        mse = torch.nn.MSELoss()
        loss = mse(output, label)
        loss_rec.append(loss.item())
        step.append(i)
        loss.backward()
        optimizer.step()

    return step, loss_rec


def test(net):
    correct = 0
    for i in range(1000):
        train, label = data_generation()
        output = net(train)
        actual = tensor_to_item(torch.argmax(label, dim=0))
        predict = tensor_to_item(torch.argmax(output, dim=0))
        if predict == actual:
            correct += 1
    accuracy = correct / 1000
    print(accuracy * 100, '%')


if __name__ == '__main__':
    net = NN(10, 2)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

    step, loss_rec = train(net)
    test(net)

    plt.figure()
    plt.plot(step, loss_rec, color='blue', label='Loss')
    plt.show()

