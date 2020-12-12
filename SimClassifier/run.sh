CUDA_VISIBLE_DEVICES='0,1,2,3' python3 main.py --pretrained -a resnet50 --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ./data
