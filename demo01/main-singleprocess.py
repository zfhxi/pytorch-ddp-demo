import os, sys
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

workspace_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(workspace_path)
from models import ConvNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nodes", default=1, type=int, metavar="N")
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    parser.add_argument("-nr", "--nr", default=0, type=int, help="ranking within the nodes")
    parser.add_argument("--epochs", default=2, type=int, metavar="N", help="number of total epochs to run")
    args = parser.parse_args()
    train(0, args)


def train(gpu, args):
    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Data loading code
    # train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)

    train_dataset = torchvision.datasets.MNIST(root="/data/czm", train=True, transform=transforms.ToTensor(), download=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch + 1, args.epochs, i + 1, total_step, loss.item()))
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))


if __name__ == "__main__":
    main()
