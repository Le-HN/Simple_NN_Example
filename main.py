from net import NN
from utils import *
from parameters import *
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
    train, label = data_generation()
    optimizer.zero_grad()
    output = net(train)
    mse = torch.nn.MSELoss()
    loss = mse(output, label)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(net):
    correct = 0
    for i in range(TEST_PARAMS.TEST_STEPS):
        train, label = data_generation()
        output = net(train)
        actual = tensor_to_item(torch.argmax(label, dim=0))
        predict = tensor_to_item(torch.argmax(output, dim=0))
        if predict == actual:
            correct += 1
    accuracy = correct / TEST_PARAMS.TEST_STEPS
    print(accuracy * 100, '%')
    return accuracy


if __name__ == '__main__':
    net = NN(10, 2)
    optimizer = optim.SGD(net.parameters(), lr=TRAIN_PARAMS.L_R, momentum=TRAIN_PARAMS.MOMENTUM)

    loss_recs = []
    acc_recs = []
    training_steps = []
    testing_steps = []
    training_step = 0
    for episode in range(TRAIN_PARAMS.EPISODE):
        for i in range(TRAIN_PARAMS.TRAINING_STEPS):
            loss_rec = train(net)
            loss_recs.append(loss_rec)
            training_steps.append(training_step)
            training_step += 1
        accuracy = test(net)
        acc_recs.append(accuracy)
        testing_steps.append(episode)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_steps, loss_recs, color='blue')
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)

    plt.subplot(1, 2, 2)
    plt.plot(testing_steps, acc_recs, color='red')
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)

    plt.tight_layout()
    plt.show()

