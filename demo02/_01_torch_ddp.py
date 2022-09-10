import os
import argparse
from datetime import datetime
import torch
from torch import nn
from torch import distributed as dist
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import numpy as np

from models import ConvNet

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=20, type=int, help="number of total epochs to run")

#########################################################
# parser.add_argument("--local_rank", type=int)
#########################################################

args = parser.parse_args()
args.nprocs = torch.cuda.device_count()


def reduce_tensor(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


# step 2: 与peer process建立连接
def init_process(rank, nprocs, backend="nccl"):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    # 若当前进程为master process，init_process_group会在MASTER_ADDR:MASTER_PORT上创建socket listener来处理来自其他进程的连接
    # 当所有进程完成连接后，该方法将在进程间建立peer connections以便实现进程通信
    dist.init_process_group(backend, rank=rank, world_size=nprocs)
    print(f"Rank {rank + 1}/{nprocs} process initialized.\n")


# step 4: 使用DistributedSampler来指明如何切分数据
def get_dataloader(rank, nprocs):
    dataloader_kwargs = {"pin_memory": True, "batch_size": 100}
    sampler_kwargs = {"num_replicas": nprocs, "rank": rank}
    train_dataset = torchvision.datasets.MNIST(root="/data/czm", train=True, transform=transforms.ToTensor(), download=False)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, **sampler_kwargs)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, **dataloader_kwargs)
    test_dataset = torchvision.datasets.MNIST(root="/data/czm", train=False, transform=transforms.ToTensor(), download=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, **sampler_kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, **dataloader_kwargs)
    return train_dataloader, test_dataloader, train_sampler, test_sampler


def main(rank, nprocs):
    start = datetime.now()
    # step 1: 设定随机种子，保证模型的Reproducibility
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    # step 2:
    init_process(rank, nprocs)
    # step 3: 隔离“数据写入操作”到master process
    # if rank == 0:
    # downloading_dataset()
    # downloading_model_weights()
    # print(f"Only rank {rank+1} executed!")
    # dist.barrier()  # 堵塞直到上面rank==0下的语句执行完

    # step 5: 将模型与输入数据都加载到对应的gpu上，并且模型必须通过DistributedDataParallel包装
    model = ConvNet()
    model.cuda(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    # step 4:
    train_dataloader, test_dataloader, train_sampler, test_sampler = get_dataloader(rank, nprocs)
    criterion = nn.CrossEntropyLoss().cuda(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3 * nprocs)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
        train(epoch, train_dataloader, model, optimizer, criterion, rank)

        if epoch % 5 == 0:
            test(test_dataloader, model, criterion, rank, nprocs)
            pass
            # torch.save(model.state_dict(), f"/spell/checkpoints/model_{epoch}.pth")
        # torch.save(model.state_dict(), f'/spell/checkpoints/model_final.pth')
    if rank == 0:
        print("Training complete in: " + str(datetime.now() - start))


def train(epoch, train_dataloader, model, optimizer, criterion, rank):
    model.train()
    total_step = len(train_dataloader)
    for i, (images, labels) in enumerate(train_dataloader):
        images = images.cuda(rank, non_blocking=True)
        labels = labels.cuda(rank, non_blocking=True)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)[0]

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0 and rank == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}".format(epoch + 1, args.epochs, i + 1, total_step, loss.item(), acc.item()))


def test(test_dataloader, model, criterion, rank, world_size):
    losses = AverageMeter("Loss", ":.4e")
    top1acc = AverageMeter("Acc@1", ":6.2f")
    top5acc = AverageMeter("Acc@1", ":6.2f")
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_dataloader):
            images = images.cuda(rank, non_blocking=True)
            labels = labels.cuda(rank, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            top1, top5 = accuracy(outputs, labels, topk=(1, 5))
            dist.barrier()

            reduced_loss = reduce_tensor(loss, world_size)
            reduced_top1 = reduce_tensor(top1, world_size)
            reduced_top5 = reduce_tensor(top5, world_size)

            losses.update(reduced_loss.item(), images.size(0))
            top1acc.update(reduced_top1.item(), images.size(0))
            top5acc.update(reduced_top5.item(), images.size(0))
        if rank == 0:
            print(" * Loss {avg_loss.avg:.4f} Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(avg_loss=losses, top1=top1acc, top5=top5acc))


if __name__ == "__main__":

    # step 0: 入口点
    mp.spawn(main, args=(args.nprocs,), nprocs=args.nprocs, join=True)


# 步骤：
# step 0: 当前进程的rank（秩，进程被分配的序号）会传给spawn的入口点（本例中的train函数），作为其第一个参数。
# step 1: 模型中的随机初始化应该被禁止，这对模型在整个训练过程中保持同步很重要。否者可能导致错误的梯度，从而模型不收敛。
# step 2: train函数在真正工作之前，需要和它对应的进程配置连接，这通过init_process_group来完成。
# step 3: 将任何文件I/O操作都隔离到master process中进行。如下载数据的操作，这样可以防止所有进程下载数据，若同时写入同一文件，可能会造成数据损坏。
# step 4: 通过传入参数rank和world_size来引导DistributedSampler完成数据切分。每个进程在每个step都会接收到来自dataset本地拷贝的batch_size大小的数据。本例中，每个step实际batch size为8x3=24
# step 5: 通过cuda(rank)来将模型和tensor加载到正确的gpu设备上，并且模型必须通过DistributedDataParallel包装
