import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = output
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def sens_accuracy(output, target):
    return accuracy(output, target)