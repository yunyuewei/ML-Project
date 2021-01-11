
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from visualize import visual_importance
from collections import OrderedDict

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))



def main():
    imgs = ['data/train/15/img_7224.jpg', 'data/train/26/img_12875.jpg']

    # create model
    path = 'resnet152_model_best.pth.tar'
    ckpt = torch.load(path, map_location=torch.device('cpu'))
    arch = ckpt['arch']
    print(arch)
    model = models.__dict__[arch]()
    model.fc = torch.nn.Linear(model.fc.in_features, 42)

    model_ckpt = OrderedDict()
    for key, value in ckpt['state_dict'].items():
        model_ckpt[key.replace('module.', '')] = value
    
    model.load_state_dict(model_ckpt)
    print("> Model loaded: {}".format(path))

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    infer_trans =  transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    visual_importance(model, infer_trans, imgs)

    


if __name__ == '__main__':
    main()
