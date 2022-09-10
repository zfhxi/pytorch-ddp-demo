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
import horovod.torch as hvd

from models import ConvNet

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=20, type=int, help="number of total epochs to run")

#########################################################
# parser.add_argument("--local_rank", type=int)
#########################################################

args = parser.parse_args()
args.nprocs = torch.cuda.device_count()


def reduce_tensor(tensor):
    rt = tensor.clone()
    hvd.allreduce(rt, name="barrier")
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


# step 4: 使用DistributedSampler来指明如何切分数据
def get_dataloader():
    dataloader_kwargs = {"pin_memory": True, "batch_size": 100}
    sampler_kwargs = {"num_replicas": hvd.size(), "rank": hvd.rank()}
    train_dataset = torchvision.datasets.MNIST(root="/data/czm", train=True, transform=transforms.ToTensor(), download=False)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, **sampler_kwargs)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, **dataloader_kwargs)
    test_dataset = torchvision.datasets.MNIST(root="/data/czm", train=False, transform=transforms.ToTensor(), download=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, **sampler_kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, **dataloader_kwargs)
    return train_dataloader, test_dataloader, train_sampler, test_sampler


def main():
    start = datetime.now()
    # step 1: 设定随机种子，保证模型的Reproducibility
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    # step 2: 初始化horovod，需要设定每个worker的device、以及线程数
    hvd.init()
    torch.cuda.set_device(hvd.rank())
    torch.set_num_threads(1)  # limit # of CPU threads to be used per worker.

    # step 3: 隔离“数据写入操作”到master process
    # if hvd.rank() == 0:
    # downloading_dataset()
    # downloading_model_weights()

    # step 4: 制定切分数据策略
    train_dataloader, test_dataloader, train_sampler, test_sampler = get_dataloader()
    # step 5: 封装模型、优化器
    model = ConvNet()
    model.cuda()
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3 * hvd.size())
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), compression=hvd.Compression.fp16, op=hvd.Average)
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
        train(epoch, train_dataloader, model, optimizer, criterion)

        if (epoch+1) % 5 == 0:
            test(test_dataloader, model, criterion)
            pass
            # torch.save(model.state_dict(), f"/spell/checkpoints/model_{epoch}.pth")
        # torch.save(model.state_dict(), f'/spell/checkpoints/model_final.pth')
    if hvd.rank() == 0:
        print("Training complete in: " + str(datetime.now() - start))


def train(epoch, train_dataloader, model, optimizer, criterion):
    model.train()
    total_step = len(train_dataloader)
    for i, (images, labels) in enumerate(train_dataloader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)[0]

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:  # and hvd.rank() == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}".format(epoch + 1, args.epochs, i + 1, total_step, loss.item(), acc.item()))


def test(test_dataloader, model, criterion):
    losses = AverageMeter("Loss", ":.4e")
    top1acc = AverageMeter("Acc@1", ":6.2f")
    top5acc = AverageMeter("Acc@1", ":6.2f")
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_dataloader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            top1, top5 = accuracy(outputs, labels, topk=(1, 5))

            reduced_loss = reduce_tensor(loss)
            reduced_top1 = reduce_tensor(top1)
            reduced_top5 = reduce_tensor(top5)

            losses.update(reduced_loss.item(), images.size(0))
            top1acc.update(reduced_top1.item(), images.size(0))
            top5acc.update(reduced_top5.item(), images.size(0))
        if hvd.rank() == 0:
            print(" * Loss {avg_loss.avg:.4f} Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(avg_loss=losses, top1=top1acc, top5=top5acc))


if __name__ == "__main__":
    main()
