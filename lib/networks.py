import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import time
from transformer import Transformer, PositionEmbeddingSine, MLP 

cudnn.deterministic = True
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

class TF_resnet(nn.Module):
    def __init__(self, resnet, cfg):
        super(TF_resnet, self).__init__()
        self.tf_dim = cfg.tf_dim
        self.tf_en_num = cfg.tf_en_num 
        self.tf_de_num = cfg.tf_de_num 
        self.num_lms = cfg.num_lms
        self.input_size = cfg.input_size
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.feat_dim = resnet.fc.weight.size()[1]

        self.deconv1 = nn.ConvTranspose2d(self.feat_dim, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_deconv1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_deconv2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_deconv3 = nn.BatchNorm2d(512)

        nn.init.normal_(self.deconv1.weight, std=0.001)
        if self.deconv1.bias is not None:
            nn.init.constant_(self.deconv1.bias, 0)
        nn.init.constant_(self.bn_deconv1.weight, 1)
        nn.init.constant_(self.bn_deconv1.bias, 0)

        nn.init.normal_(self.deconv2.weight, std=0.001)
        if self.deconv2.bias is not None:
            nn.init.constant_(self.deconv2.bias, 0)
        nn.init.constant_(self.bn_deconv2.weight, 1)
        nn.init.constant_(self.bn_deconv2.bias, 0)

        nn.init.normal_(self.deconv3.weight, std=0.001)
        if self.deconv3.bias is not None:
            nn.init.constant_(self.deconv3.bias, 0)
        nn.init.constant_(self.bn_deconv3.weight, 1)
        nn.init.constant_(self.bn_deconv3.bias, 0)

        self.cls_layer = nn.Conv2d(512, self.num_lms, kernel_size=1, stride=1, padding=0)
        self.x_layer = nn.Conv2d(512, self.num_lms, kernel_size=1, stride=1, padding=0)
        self.y_layer = nn.Conv2d(512, self.num_lms, kernel_size=1, stride=1, padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc0 = nn.Linear(self.feat_dim, 512)
        self.domain_cls = nn.Linear(512, 1)

        nn.init.normal_(self.cls_layer.weight, std=0.001)
        if self.cls_layer.bias is not None:
            nn.init.constant_(self.cls_layer.bias, 0)

        nn.init.normal_(self.x_layer.weight, std=0.001)
        if self.x_layer.bias is not None:
            nn.init.constant_(self.x_layer.bias, 0)

        nn.init.normal_(self.y_layer.weight, std=0.001)
        if self.y_layer.bias is not None:
            nn.init.constant_(self.y_layer.bias, 0)

        nn.init.normal_(self.fc0.weight, std=0.001)
        if self.fc0.bias is not None:
            nn.init.constant_(self.fc0.bias, 0)

        nn.init.normal_(self.domain_cls.weight, std=0.001)
        if self.domain_cls.bias is not None:
            nn.init.constant_(self.domain_cls.bias, 0)

        self.fmap_size = (int(self.input_size[0]/4), int(self.input_size[1]/4))
        self.fc1 = nn.Linear(self.tf_dim, self.num_lms*self.tf_dim)

        nn.init.normal_(self.fc1.weight, std=0.001)
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)

        self.pos_layer = PositionEmbeddingSine(self.tf_dim//2)
        self.transformer = Transformer(d_model=self.tf_dim,
                                       num_encoder_layers=self.tf_en_num,
                                       num_decoder_layers=self.tf_de_num,
                                       num_queries=self.num_lms)
        self.query_embed = nn.Embedding(self.num_lms, self.tf_dim)
        self.bbox_embed = MLP(self.tf_dim, 512, 2, 3)
        self.input_proj = nn.Conv2d(512, self.tf_dim, kernel_size=1)

    def forward(self, x, coef_grl):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        ##############
        x_domain = GradReverse.grad_reverse(x, coef_grl)
        x_domain = self.avgpool(x_domain)
        x_domain = x_domain.view(x_domain.size(0), -1)
        x_domain = self.fc0(x_domain)
        x_domain = F.relu(x_domain)
        x_domain = self.domain_cls(x_domain)
        x = self.deconv1(x)
        x = self.bn_deconv1(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = self.bn_deconv2(x)
        x = F.relu(x)
        x = self.deconv3(x)
        x = self.bn_deconv3(x)
        x = F.relu(x)
        cls = self.cls_layer(x)
        offset_x = self.x_layer(x)
        offset_y = self.y_layer(x)
        x = self.input_proj(x)
        pos_embed = self.pos_layer(x)
        x_pool = F.avg_pool2d(x+pos_embed, self.fmap_size).squeeze(2).squeeze(2)
        dq = self.fc1(x_pool)
        dq = dq.view(-1, self.num_lms, self.tf_dim).permute(1,0,2)
        hs, _, atten_weights_list, self_atten_weights_list = self.transformer(x, None, self.query_embed.weight, pos_embed, dq)
        # bs x num_lms x 2
        outputs_coord = self.bbox_embed(hs.squeeze(0))
        return cls, offset_x, offset_y, outputs_coord, x_domain

