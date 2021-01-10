

import torchvision.transforms as transforms
import torchvision.models as models
from visualize import visual_importance

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))



def main():
    arch = 'resnet18'
    imgs = ['data/train/21/img_9906.jpg','data/train/39/img_19381.jpg']

    # create model
    model = models.__dict__[arch]()

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
