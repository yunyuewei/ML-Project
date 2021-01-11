CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' python -W ignore  main.py  --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --world-size 1 --rank 0 --pretrained  -a resnet152  ./data/

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' python -W ignore  main.py  --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --world-size 1 --rank 0 --pretrained  -a resnet101  ./data/

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' python -W ignore  main.py  --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --world-size 1 --rank 0 --pretrained  -a resnet50  ./data/

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' python -W ignore  main.py  --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --world-size 1 --rank 0 --pretrained  -a resnet18  ./data/

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' python -W ignore  main.py  --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --world-size 1 --rank 0 --pretrained  -a densenet201  ./data/

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' python -W ignore  main.py  --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --world-size 1 --rank 0 --pretrained  -a densenet169  ./data/

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' python -W ignore  main.py  --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --world-size 1 --rank 0 --pretrained  -a densenet161  ./data/

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' python -W ignore  main.py  --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --world-size 1 --rank 0 --pretrained  -a densenet121  ./data/

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' python -W ignore  main.py  --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --world-size 1 --rank 0 --pretrained  -a vgg11_bn  ./data/

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' python -W ignore  main.py  --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --world-size 1 --rank 0 --pretrained  -a vgg13_bn  ./data/

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' python -W ignore  main.py  --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --world-size 1 --rank 0 --pretrained  -a vgg16_bn  ./data/

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' python -W ignore  main.py  --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --world-size 1 --rank 0 --pretrained  -a vgg19_bn  ./data/


#unpretrained

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' python -W ignore  main.py  --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --world-size 1 --rank 0   -a resnet152  ./data/

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' python -W ignore  main.py  --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --world-size 1 --rank 0   -a resnet101  ./data/

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' python -W ignore  main.py  --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --world-size 1 --rank 0   -a resnet50  ./data/

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' python -W ignore  main.py  --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --world-size 1 --rank 0   -a resnet18  ./data/

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' python -W ignore  main.py  --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --world-size 1 --rank 0   -a densenet201  ./data/

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' python -W ignore  main.py  --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --world-size 1 --rank 0   -a densenet169  ./data/

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' python -W ignore  main.py  --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --world-size 1 --rank 0   -a densenet161  ./data/

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' python -W ignore  main.py  --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --world-size 1 --rank 0   -a densenet121  ./data/

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' python -W ignore  main.py  --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --world-size 1 --rank 0   -a vgg11_bn  ./data/

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' python -W ignore  main.py  --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --world-size 1 --rank 0   -a vgg13_bn  ./data/

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' python -W ignore  main.py  --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --world-size 1 --rank 0   -a vgg16_bn  ./data/

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' python -W ignore  main.py  --dist-url 'tcp://127.0.0.1:23143' --dist-backend 'nccl' --world-size 1 --rank 0   -a vgg19_bn  ./data/


