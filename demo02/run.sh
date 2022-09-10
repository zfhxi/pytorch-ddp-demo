#python _01_torch_ddp.py
#NCCL_DEBUG=INFO horovodrun -np 3 -H localhost:4 python 02_torch_hvd.py