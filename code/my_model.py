import segmentation_models_pytorch as smp
import torch.nn as nn
import torchvision.models as models


# torchvision models
class FCN_RESNET50(nn.Module):
    def __init__(self):
        super(FCN_RESNET50, self).__init__()
        self.model = models.segmentation.fcn_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(512, 29, kernel_size=1)

    def forward(self, x):
        return self.model(x)


def FCN_RESNET101():
    model = models.segmentation.fcn_resnet101(pretrained=True)
    model.classifier[4] = nn.Conv2d(512, 29, kernel_size=1)
    return model


def DEEPLABV3_RESNET50():
    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[-1] = nn.Conv2d(256, 29, kernel_size=1)
    return model


def DEEPLABV3_RESNET101():
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.classifier[-1] = nn.Conv2d(256, 29, kernel_size=1)
    return model


def EFFICIENT_UNET():
    model = smp.Unet(
        encoder_name="efficientnet-b0",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=29,  # model output channels (number of classes in your dataset)
    )
    return model
