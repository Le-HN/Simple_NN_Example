import torch


def tensor_to_item(tensor):
    return tensor.cpu().detach().numpy().item()


def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()