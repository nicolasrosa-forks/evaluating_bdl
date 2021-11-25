# ======= #
#  Model  #
# ======= #

# Torch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0, std=1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(mean=0, std=1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def conv_bn_elu(in_channels, out_channels, kernel_size, stride=1, padding=0, bn=True, elu=True):
    bias = not bn

    # Conv-BN-ELU Block
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))

    if bn:
        layers.append(nn.BatchNorm2d(out_channels))

    if elu:
        layers.append(nn.ELU(alpha=0.2, inplace=True))  # inplace=True means that it will modify the input directly, without allocating any additional output.

    layers = nn.Sequential(*layers)

    # Initialize the Weights
    for m in layers.modules():
        init_weights(m)

    return layers

def conv_bn_sigm(in_channels, out_channels, kernel_size, stride=1, padding=0, bn=True, sigm=False):
    bias = not bn

    # Conv-BN-Sigmoid Block
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))

    if bn:
        layers.append(nn.BatchNorm2d(out_channels))

    if sigm:
        layers.append(nn.Sigmoid())
    
    layers = nn.Sequential(*layers)

    # Initialize the Weights
    for m in layers.modules():
        init_weights(m)

    return layers

def convt_bn_elu(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bn=True, elu=True):
    bias = not bn
    
    # ConvT-BN-ELU Block
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    
    if elu:
        layers.append(nn.ELU(alpha=0.2, inplace=True))
    
    layers = nn.Sequential(*layers)

    # Initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers

class DepthEstimationNet(nn.Module):
    def __init__(self, pretrained=False):
        # super(DepthEstimationNet, self).__init__()  # Python2
        super().__init__()                            # Python3

        self.total_num_params = None

        self.layers = 34
        self.pretrained = pretrained

        # Network Layers Declaration, Resnet-34
        # https://arxiv.org/pdf/1512.03385.pdf
        self.conv1_d = conv_bn_elu(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_img = conv_bn_elu(3, 32, kernel_size=3, stride=1, padding=1)
        
        self.conv1_1 = conv_bn_elu(32, 64, kernel_size=3, stride=1, padding=1)
        
        pretrained_model = resnet.__dict__['resnet{}'.format(self.layers)](pretrained=self.pretrained)
        if not self.pretrained:
            pretrained_model.apply(init_weights)

        self.conv2 = pretrained_model._modules['layer1']  # Purple Block?,  64-Channels Layers
        self.conv3 = pretrained_model._modules['layer2']  # Green  Block?, 128-Channels Layers
        self.conv4 = pretrained_model._modules['layer3']  # Salmon Block?, 256-Channels Layers
        self.conv5 = pretrained_model._modules['layer4']  # Blue   Block?, 512-Channels Layers
        del pretrained_model # (free memory)

        # print(self.conv2)
        # print(self.conv3)
        # print(self.conv4)
        # print(self.conv5)
        
        if self.layers <=34:
            num_channels = 512
        elif self.layers >= 50:
            num_channels = 2048

        self.conv6 = conv_bn_elu(num_channels, 512, kernel_size=3, stride=2, padding=1)

        self.convt5 = convt_bn_elu(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convt4 = convt_bn_elu(in_channels=(512+256), out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convt3 = convt_bn_elu(in_channels=(256+128), out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convt2 = convt_bn_elu(in_channels=(128+64), out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convt1 = convt_bn_elu(in_channels=(64+64), out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Variance values cannot be negative. But log(vars) have negative values for numbers < 0.0.
        # self.convf_mean = conv_bn_selu(in_channels=128, out_channels=1, kernel_size=1, stride=1, bn=False, selu=False)
        # self.convf_var = conv_bn_selu(in_channels=128, out_channels=1, kernel_size=1, stride=1, bn=False, selu=False)
        self.convf_mean = conv_bn_sigm(in_channels=128, out_channels=1, kernel_size=1, stride=1, bn=False, sigm=True)
        self.convf_var = conv_bn_elu(in_channels=128, out_channels=1, kernel_size=1, stride=1, bn=False, elu=False)

        self.total_num_params = sum(param.numel() for param in self.parameters())

        print("'DepthEstimationNet' object created!")

    def forward(self, img):
        # (img has shape: (batch_size, h, w, 3)) (bgr)
        # (sparse has shape: (batch_size, h, w))

        # Add/Permute dimension to/of input tensors
        img = img.permute(0, 3, 1, 2)                   # (shape: (batch_size, 3, h, w))
        # sparse = torch.unsqueeze(sparse, dim=1)         # (shape: (batch_size, 1, h, w))

        # Network Architecture
        # (Encoder)
        # conv1_d = self.conv1_d(sparse)                  # (shape: (batch_size, 32, h, w))
        conv1_img = self.conv1_img(img)                 # (shape: (batch_size, 32, h, w))

        # conv1 = torch.cat((conv1_d, conv1_img), dim=1)  # (shape: (batch_size, 64, h, w))
        conv1 = conv1_img                               # (shape: (batch_size, 32, h, w))
        conv1_1 = self.conv1_1(conv1)
        
        conv2 = self.conv2(conv1_1)                       # (shape: (batch_size, 64, h, w)), Purple Block?
        conv3 = self.conv3(conv2)                       # (shape: (batch_size, 128, h/2, w/2)), Green Block?
        conv4 = self.conv4(conv3)                       # (shape: (batch_size, 256, h/4, w/4)), Salmon Block?
        conv5 = self.conv5(conv4)                       # (shape: (batch_size, 512, h/8, w/8)), Blue Block?
        conv6 = self.conv6(conv5)                       # (shape: (batch_size, 512, h/16, w/16)), Extra Conv Block?
        
        # (Decoder)
        convt5 = self.convt5(conv6)                     # (shape: (batch_size, 256, h/8, w/8)))
        y = torch.cat((convt5, conv5), dim=1)           # (shape: (batch_size, 256+512, h/8, w/8))

        convt4 = self.convt4(y)                         # (shape: (batch_size, 128, h/4, w/4))
        y = torch.cat((convt4, conv4), dim=1)           # (shape: (batch_size, 128+256, h/4, w/4))

        convt3 = self.convt3(y)                         # (shape: (batch_size, 64, h/2, w/2))
        y = torch.cat((convt3, conv3), dim=1)           # (shape: (batch_size, 64+128, h/2, w/2))

        convt2 = self.convt2(y)                         # (shape: (batch_size, 64, h, w))
        y = torch.cat((convt2, conv2), dim=1)           # (shape: (batch_size, 64+64, h, w))

        convt1 = self.convt1(y)                         # (shape: (batch_size, 64, h, w))
        y = torch.cat((convt1, conv1_1), dim=1)           # (shape: (batch_size, 64+64, h, w))

        mean = self.convf_mean(y)                      # (shape: (batch_size, 1, h, w))
        log_var = self.convf_var(y)                    # (shape: (batch_size, 1, h, w))

        # Since mean and log_var share the same layers, it's good to indicate the magnitude diffence between the 
        # predicted variables.
        # mean = 100*mean  # Here, we are suggesting that the mean values should be hundred times bigger than variance values
        max_depth = 80  # KITTI Dataset
        mean = max_depth*mean  # Here, we are suggesting that the mean values should be hundred times bigger than variance values

        return (mean, log_var)

    def print_total_num_params(self):
        print("Total number of params: {}\n".format(self.total_num_params))

# Debug
model = DepthEstimationNet()
