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
    parser.add_argument("--nodes", default=1, type=int, help="number of machines(or nodes) you used")
    parser.add_argument("--node-rank", default=0, type=int, help="global ranking within the nodes, 0 for master, 1,2,... for other nodes")
    # parser.add_argument("--gpus", default=1, type=int, help="number of gpus per node")
    parser.add_argument("--epochs", default=2, type=int, help="number of total epochs to run")

    #########################################################
    # parser.add_argument("--local_rank", type=int)
    #########################################################

    args = parser.parse_args()
    args.gpu_nums = torch.cuda.device_count()

    #########################################################
    args.world_size = args.gpu_nums * args.nodes  #
    os.environ["MASTER_ADDR"] = "127.0.0.1"  #
    os.environ["MASTER_PORT"] = "9999"  #
    mp.spawn(train, nprocs=args.gpu_nums, args=(args,))  #
    # train(0, args)
    #########################################################


def train(gpu_id, args):
    ############################################################
    rank = args.node_rank * args.gpu_nums + gpu_id
    dist.init_process_group(backend="nccl", init_method="env://", world_size=args.world_size, rank=rank)
    ############################################################

    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu_id)
    model.cuda(gpu_id)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu_id)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    ###############################################################
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id])
    ###############################################################

    # Data loading code
    # train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)

    train_dataset = torchvision.datasets.MNIST(root="/data/czm", train=True, transform=transforms.ToTensor(), download=False)

    ################################################################
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, sampler=train_sampler)
    ################################################################

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
            if (i + 1) % 100 == 0 and gpu_id == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch + 1, args.epochs, i + 1, total_step, loss.item()))

    if gpu_id == 0:
        print("Training complete in: " + str(datetime.now() - start))


if __name__ == "__main__":
    main()
