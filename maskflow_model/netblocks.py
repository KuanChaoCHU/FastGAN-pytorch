# type: ignore
import os
from util import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
from torch.optim import lr_scheduler
import torchvision.models as models
import functools
import inspect


#__all__ = []


###############################################################################
# essential functions
###############################################################################
class Identity(nn.Module):
    def forward(self, x):
        return x
    

def get_norm_layer(norm_type='batch'):
    """ Return a normalization layer
    BatchNorm: learnable affine parameters and tracking the running mean/variance
    InstanceNorm: no    
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()  # make it equal position to nn.Module
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler  (copied)
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='kaiming', init_gain=0.02):
    """ Initialize network weights
    
    init_type (str) -- normal | xavier | kaiming | orthogonal 
    
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.(copied)
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    
    if isinstance(net, torch.nn.DataParallel):
        name = net.module.__class__.__name__
    else:
        name = net.__class__.__name__    
        
    print('initialize network {} with {}'.format(name, init_type))  
    net.apply(init_func)


def init_net(net, parallisim='basic', init_type='kaiming', init_gain=0.02, 
             gpu_ids=[], neglect_init_weight=False):
    """ Initialize a network: 1. register CPU/GPU device; 2. initialize the network weights
    
    init_type (str) -- normal | xavier | kaiming | orthogonal 
    init_gain (float) -- scaling factor for normal, xavier and orthogonal
    gpu_ids (int list) -- e.g., 0,1,2     
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        
        if parallisim == 'basic':       
            net.to(gpu_ids[0])  # net to GPU
            net = torch.nn.DataParallel(net, gpu_ids)  # entry-level parallelism
        elif parallisim == 'advanced':
            pass           
        
    if not neglect_init_weight:
        init_weights(net, init_type, init_gain)
    return net


def distributed_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # TCP/UDP
    
    # initialize the process group
    torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)


def distributed_cleanup():
    torch.distributed.destroy_process_group()


###############################################################################
# ResNet
###############################################################################
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 global_pool=True, include_last_fc=False, fpn_level=1):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.is_global_pool = global_pool
        self.is_include_last = include_last_fc
        self.fpn_level = fpn_level

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
                
        x = self.layer1(x)                        # input shape 224
        x_fpn_layer2 = self.layer2(x)             # out: (512, 28, 28)
        x_fpn_layer1 = self.layer3(x_fpn_layer2)  # out: (1024, 14, 14)
        x_fpn_layer0 = self.layer4(x_fpn_layer1)  # out: (2048, 7, 7)
               
        if not self.is_global_pool:
            if self.fpn_level == 1:
                return [x_fpn_layer0]    
            elif self.fpn_level == 2:
                return [x_fpn_layer0, x_fpn_layer1]
            elif self.fpn_level == 3:
                return [x_fpn_layer0, x_fpn_layer1, x_fpn_layer2]

        x = self.avgpool(x_fpn_layer0)
        x = x.view(x.size(0), -1)
        if not self.is_include_last:
            return x
        
        x = self.fc(x)
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


###############################################################################
# common nets
###############################################################################
def define_G(noise_d, semantic_d, agg_mode):
    pass


class Generator(nn.Module):
    """ opt: noise_d semantic_d agg_mode feature_d G_hidden_size
    """
    
    def __init__(self, opt, layer):
        super().__init__()
        self.semantic_d = opt.sem_d
        self.noise_d = opt.sem_d if opt.noise_d is None else opt.noise_d
        self.agg_mode = opt.g_agg_mode
        self.feature_d = utils.get_layer_feature_d(opt, layer)
        
        if self.agg_mode == 'multiply':
            assert self.noise_d == self.semantic_d, "Can't perform multiply for noise and sem with different size."
            self.input_length = self.semantic_d
        elif self.agg_mode == 'concatenate':
            self.input_length = self.noise_d + self.semantic_d
                
        self.linear1 = nn.Linear(self.input_length, opt.g_hidden_size)
        self.linear2 = nn.Linear(opt.g_hidden_size, self.feature_d)
    
    def forward(self, noise, sem):
        if self.agg_mode == 'multiply':
            x = torch.mul(noise, sem)
        elif self.agg_mode == 'concatenate':
            x = torch.cat([noise, sem], dim=-1)
        x = self.linear1(x)
        x = nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.linear2(x)
        x = nn.ReLU()(x)
        return x

    def freeze(self):
        # add this function to G in netblock
        for param in self.parameters():
            param.requires_grad = False


def define_D():
    pass


class Reconstructor(nn.Module):
    """
    """
    
    def __init__(self, opt, layer):
        super().__init__()
        feature_d = utils.get_layer_feature_d(opt, layer)
        self.linear1 = nn.Linear(feature_d, opt.g_hidden_size)  
        self.linear2 = nn.Linear(opt.g_hidden_size, opt.sem_d)
        
    def forward(self, feature):
        x =self.linear1(feature)
        x = nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.linear2(x)
        x = nn.ReLU()(x)  # dangerous! sem can have negtive value in some dataset
        return x


class Discriminator(nn.Module):
    ''' Returns the discriminator net  
        NOTE: BatchNormalization results in huge degradation of the accuracy.
              (original paper) 
        ***** sem dim is much smaller than feature dim, concat->embedding?
        abandon
        sem_embedding = KL.Flatten()(KL.Embedding(self.semantic_dim, self.feature_dim)(sem))
    '''
    
    def __init__(self, opt, layer):
        super().__init__()
        self.agg_mode = opt.d_agg_mode
        self.feature_d = utils.get_layer_feature_d(opt, layer)
        self.sem_d = opt.sem_d
    
        if self.agg_mode == 'dense_multiply':  # 0707 fixed: typo as add_module(a func name)
            self.align = nn.Linear(self.sem_d, self.feature_d)
            self.input_length = self.feature_d
        elif self.agg_mode == 'concatenate':
            self.input_length = self.feature_d + self.sem_d 
        self.linear1 = nn.Linear(self.input_length, opt.d_hidden_size)  # vanilla 1024; others 4096
        self.linear2 = nn.Linear(opt.d_hidden_size, 1)    
    
    def forward(self, feature, sem):
        if self.agg_mode == 'dense_multiply':
            sem_aligned = self.align(sem)
            x = torch.mul(feature, sem_aligned)
        elif self.agg_mode == 'concatenate':
            x = torch.cat([feature, sem], dim=-1)
        x = self.linear1(x)
        x = nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.linear2(x)              
        return x


class SoftmaxClassifier(nn.Module):
    """
    """
    
    def __init__(self, opt, layer, seen_only=False, nb_unseen=0):
        super().__init__()
        feature_d = utils.get_layer_feature_d(opt, layer)
        if not seen_only:
            self.classifier = nn.Linear(feature_d, opt.nb_classes)
        else:
            self.classifier = nn.Linear(feature_d, opt.nb_classes-nb_unseen)
        self.logsoftmax = nn.LogSoftmax(dim=1)       
        
    def forward(self,x):
        x = self.classifier(x)
        x = self.logsoftmax(x)
        return x
    
    
class YoloClassifier3(nn.Module):
    """ BatchNorm 1d
    """
    
    def __init__(self, opt, nb_cbr_blocks=2, bn_layer=True):
        super().__init__()
        self.nb_cbr_blocks = nb_cbr_blocks
    
        self.conv1 = nn.Conv2d(2048, 1024, 1, bias=not(bn_layer)) 
        self.bn1 = nn.BatchNorm1d(1024)
        
        self.conv2 = nn.Conv2d(1024, 512, 1)
        self.bn2 = nn.BatchNorm1d(512)
        self.conv3 = nn.Conv2d(512, 1024, 3, padding=(1,1))
        
        self.conv4 = nn.Conv2d(1024, 512, 1)
        self.bn3 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv2d(512, 1024, 3, padding=(1,1))
        
        self.conv6 = nn.Conv2d(1024, 216, 1)  
    
    def forward(self, x):
        """ Input x shape=(batch, 7, 7, 2048) note: need to swap dimension order
        """
        x = self.conv1(x)
        #x = torch.flatten(x).unsqueeze(dim=0)
        x = x.view([x.size()[0], x.size()[1], -1])  # merge the H W dimension to pass through BN1d
        x = self.bn1(x)
        x = x.view([x.size()[0], x.size()[1], 7, 7])  # 7 or 13? hard-coded, very bad...
        x = nn.ReLU()(x)
        #for i in range(len(self.nb_cbr_blocks)):
        x = self.conv2(x)
        x = x.view([x.size()[0], x.size()[1], -1])
        x = self.bn2(x)
        x = x.view([x.size()[0], x.size()[1], 7, 7])  # 7 or 13? 
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        
        x = self.conv4(x)
        x = x.view([x.size()[0], x.size()[1], -1])
        x = self.bn3(x)
        x = x.view([x.size()[0], x.size()[1], 7, 7])  # 7 or 13? 
        x = nn.ReLU()(x)
        x = self.conv5(x)
        x = nn.ReLU()(x)
        
        x = self.conv6(x)
        return x       


class MaskEffConv2d(nn.Module):
    # padding: fixed to 'zeros'
    # kernel_size: fixed to 3
    def __init__(self, in_channels, out_channels, alpha):
        super(MaskEffConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 3 
        self.padding = (1,1)
        self.alpha = alpha

        self.register_buffer('kernel_mask', torch.FloatTensor([[0.0, 0.0, 0.0],
                                                               [0.0, 1-self.alpha, 0.0],
                                                               [0.0, 0.0, 0.0]]))

        self.weight = nn.Parameter(torch.FloatTensor(
            out_channels,in_channels,self.kernel_size,self.kernel_size))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels,))
               
    def forward(self, x, locs, is_inference=False):
        if is_inference:
            return self._route1(x)
        # locs should have the same shape as x (e.g., [B, 1, 19, 19]) and is binary: 1 for syn grids 
        raw_mask = locs.repeat((1, self.in_channels, 1, 1)) 
                
        # go route 1
        route1_mask = raw_mask.detach().clone()
        route1_mask[route1_mask == 1] = self.alpha
        route1_mask[route1_mask == 0] = 1
        res1 = x * route1_mask
        res1 = self._route1(res1)
                
        # go route 2
        route2_mask = raw_mask
        res2 = x * route2_mask
        res2 = self._route2(res2)
                 
        return res1 + res2

    def _route1(self, x):
        route1_weight = self.weight
        return F.conv2d(x, route1_weight, self.bias, padding=self.padding)

    def _route2(self, x):
        route2_weight = self.weight * self.kernel_mask
        route2_bias = self.bias * 0.0 
        return F.conv2d(x, route2_weight, route2_bias, padding=self.padding)


class YoloClassifier(nn.Module):
    """ BatchNorm 2d has 3*3 and bn
    # stable: before applying maskeffconv2d 
    """    
    def __init__(self, opt, nb_cbr_blocks=2, bn_layer=True):
        super().__init__()
        self.nb_cbr_blocks = nb_cbr_blocks
        
        self.conv1 = nn.Conv2d(2048, 1024, 1, bias=not(bn_layer)) 
        self.bn1 = nn.BatchNorm2d(1024)
        
        self.conv2 = nn.Conv2d(1024, 512, 1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = MaskEffConv2d(512, 1024, opt.maskeffconv2d_alpha)
        
        self.conv4 = nn.Conv2d(1024, 512, 1)
        self.bn3 = nn.BatchNorm2d(512)
        #self.conv5 = MaskEffConv2d(512, 1024, opt.maskeffconv2d_alpha)
        self.conv5 = nn.Conv2d(512, 1024, 3, padding=(1,1))
        
        self.conv6 = nn.Conv2d(1024, 216, 1)
        
    def forward(self, x, batch_syn_locs, is_inference=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x, batch_syn_locs, is_inference)
        x = nn.ReLU()(x) 
        
        x = self.conv4(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)
        #x = self.conv5(x, batch_syn_locs, is_inference)
        x = self.conv5(x)
        x = nn.ReLU()(x) 
                       
        x = self.conv6(x)
        return x       
    

class YoloClassifier_normal(nn.Module):
    """ BatchNorm 2d has 3*3 and bn
    # stable: before applying maskeffconv2d 
    """    
    def __init__(self, opt, nb_cbr_blocks=2, bn_layer=True):
        super().__init__()
        self.nb_cbr_blocks = nb_cbr_blocks
        
        self.conv1 = nn.Conv2d(2048, 1024, 1, bias=not(bn_layer)) 
        self.bn1 = nn.BatchNorm2d(1024)
        
        self.conv2 = nn.Conv2d(1024, 512, 1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 1024, 3, padding=(1,1))
        
        self.conv4 = nn.Conv2d(1024, 512, 1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1024, 3, padding=(1,1))
        
        self.conv6 = nn.Conv2d(1024, 216, 1)
        
    def forward(self, x, batch_syn_locs=None, is_inference=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)  # was first missed
        
        x = self.conv4(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)
        x = self.conv5(x)
        x = nn.ReLU()(x)  # was first missed
                       
        x = self.conv6(x)
        return x


class YoloClassifierRefHead(nn.Module):
    """ BatchNorm 2d has 3*3 and bn    
    """    
    def __init__(self, opt, nb_cbr_blocks=2, bn_layer=True):
        super().__init__()
        self.nb_cbr_blocks = nb_cbr_blocks
        
        self.conv1 = nn.Conv2d(2048, 1024, 1, bias=not(bn_layer)) 
        self.bn1 = nn.BatchNorm2d(1024)
        
        self.conv2 = nn.Conv2d(1024, 512, 1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 1024, 3, padding=(1,1))
        
        self.conv4 = nn.Conv2d(1024, 512, 1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1024, 3, padding=(1,1))
     
        self.conv6 = nn.Conv2d(1024, 216, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)  
        
        x = self.conv4(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)
        x = self.conv5(x)
        x = nn.ReLU()(x)  
        
        x = self.conv6(x)
        return x


class YoloClassifierRefHead_v2(nn.Module):
    """ Change: 3*3 conv all to 1*1
    """
    
    def __init__(self, opt, nb_cbr_blocks=2, bn_layer=True):
        super().__init__()
        self.nb_cbr_blocks = nb_cbr_blocks
        
        self.conv1 = nn.Conv2d(2048, 1024, 1, bias=not(bn_layer)) 
        self.bn1 = nn.BatchNorm2d(1024)
        
        self.conv2 = nn.Conv2d(1024, 512, 1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 1024, 1)
        
        self.conv4 = nn.Conv2d(1024, 512, 1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1024, 1)
        
        self.conv6 = nn.Conv2d(1024, 216, 1)
        
    def forward(self, x, batch_syn_locs):        
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)  
        
        x = self.conv4(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)
        x = self.conv5(x)
        x = nn.ReLU()(x)  
                       
        x = self.conv6(x)
        return x             
        
        
class YoloClassifier_L1(nn.Module):
    """ For layer1 prediction 
    """    

    def __init__(self, opt, bn_layer=True):
        super().__init__()
        self.conv1 = nn.Conv2d(2048, 512, 1)
        self.bn1 = nn.BatchNorm2d(512)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(1536, 512, 1)
        
        self.conv3 = nn.Conv2d(512, 1024, 3, padding=(1,1))
        self.conv4 = nn.Conv2d(1024, 512, 1)
        self.bn2 = nn.BatchNorm2d(512)
        
        self.conv5 = nn.Conv2d(512, 1024, 3, padding=(1,1))
        self.conv6 = nn.Conv2d(1024, 512, 1)
        self.bn3 = nn.BatchNorm2d(512)
        
        self.conv7 = nn.Conv2d(512, 216, 1)        
    
    def forward(self, feat0, feat1, is_get_mixfeat=-1):
        x = self.conv1(feat0)  # out: (7, 7, 512)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.upsample1(x)  # out: (14, 14, 512)
        x = torch.cat((x, feat1), dim=1)  # out: (14, 14, 1536)
        if is_get_mixfeat == 0:
            return x
        out1 = self.conv2(x)  # out: (14, 14, 512)
        if is_get_mixfeat == 1:
            return out1        

        x = self.conv3(out1)
        x = nn.ReLU()(x)
        x = self.conv4(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        
        x = self.conv5(x)
        x = nn.ReLU()(x)
        x = self.conv6(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)
        
        x = self.conv7(x)
        return [out1, x]        
    
    
class YoloClassifier_L2(nn.Module):
    """ For layer2 prediction   
    """
    
    def __init__(self, opt, bn_layer=True):
        super().__init__()
        self.conv1 = nn.Conv2d(512, 256, 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(768, 512, 1)
        
        self.conv3 = nn.Conv2d(512, 1024, 3, padding=(1,1))
        self.conv4 = nn.Conv2d(1024, 512, 1)
        self.bn2 = nn.BatchNorm2d(512)
        
        self.conv5 = nn.Conv2d(512, 1024, 3, padding=(1,1))
        self.conv6 = nn.Conv2d(1024, 512, 1)
        self.bn3 = nn.BatchNorm2d(512)
        
        self.conv7 = nn.Conv2d(512, 216, 1)          
    
    def forward(self, feat0_1, feat2, is_get_mixfeat=-1):
        x = self.conv1(feat0_1)  # out: (14, 14, 256)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.upsample1(x)  # out: (28, 28, 256)
        x = torch.cat((x, feat2), dim=1)  # out: (28, 28, 768)
        if is_get_mixfeat == 0:
            return x 
        x = self.conv2(x)  # out: (28, 28, 512)
        if is_get_mixfeat == 1:
            return x

        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.conv4(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        
        x = self.conv5(x)
        x = nn.ReLU()(x)
        x = self.conv6(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)
        
        x = self.conv7(x)
        return x
    

class FeatureExtractor:
    """ Supports ResNet50 and ResNet101
    """

    def __init__(self, opt, pre_trained=True, include_global_pool=True, fpn_l=1):
        self.opt = opt
        
        # Models
        if opt.feature_extractor == 'ResNet101':
            self.feature_extractor = resnet101(pretrained=pre_trained, global_pool=include_global_pool,
                                               include_last_fc=False, fpn_level=fpn_l)
        elif opt.feature_extractor == 'ResNet50':
            self.feature_extractor = resnet50(pretrained=pre_trained, global_pool=include_global_pool,
                                              include_last_fc=False, fpn_level=fpn_l)
        
        #self.feature_extractor = init_net(self.feature_extractor, gpu_ids=opt.gpu_ids,
        #                                  neglect_init_weight=pre_trained)
    
    def __call__(self, input_data):
        return self.feature_extractor(input_data)
    

###############################################################################
# other layers
###############################################################################
def depthwise_separable_conv2d(*args, **kwargs):
    """ Theoritically faster than a normal Conv2d
        
    """
    #arg_names = inspect.getargspec(nn.Conv2d)[0]  # deprecated? ('self' is included)
    arg_names = list(inspect.signature(nn.Conv2d).parameters.keys())
    
    for idx, arg in enumerate(args):
        kwargs[arg_names[idx]] = arg  
    
    kwargs_1 = kwargs.copy()  # shallow copy! 
    kwargs_1['out_channels'] = kwargs_1['in_channels']
    kwargs_1['groups'] = kwargs_1['in_channels']

    kwargs_2 = kwargs.copy()  # shallow copy!
    kwargs_2['padding'] = 0
    kwargs_2['kernel_size'] = 1
    
    model = nn.Sequential(
              nn.Conv2d(**kwargs_1),  # depthwise
              nn.Conv2d(**kwargs_2)   # pointwise
            )    
    return model


###############################################################################
# Loss functions
###############################################################################
def cal_gradient_penalty(netD, real_data, fake_data, semantics, device, typ='mixed', constant=1.0, lambda_gp=10.0):
    """ Calculate the gradient penalty used in WGAN-GP 
    
    Parameters:
        netD (nn.Module)
        real_data (tensor)
        fake_data (tensor)
        device (str)
        type (str)
        constant (float)
        lambda_gp (float)
    """
    if lambda_gp > 0.0:
        if typ == 'real':
            interpolatesv = real_data
        elif typ == 'fake':
            interpolatesv = fake_data
        elif typ == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv, semantics)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=-1) - constant) ** 2).mean() * lambda_gp
        return gradient_penalty, gradients        
    else:
        return 0.0, None
   
   
class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """
    def __init__(self, gan_mode='wgangp', target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def smooth_l1_loss(input, target, reduction='mean', beta=1.0):
    # since we don't have the beta option yet in v1.1.0,
    # Note: 0705 found a mistake (x -> n).
    # Results using this function before could be checked again 
    if beta < 1e-8:
        loss = torch.abs(input - target)
    else:
        x = torch.abs(input - target)
        cond = x < beta
        loss = torch.where(cond, 0.5 * x ** 2 / beta, x - 0.5 * beta)
    
    if reduction == 'mean':
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss

    
def classification_loss(*args, **kwargs):
    return nn.NLLLoss(*args, **kwargs)


def euclidean_dist_loss(pred, target):
    diff = pred - target
    l2norm = torch.norm(diff, 2, dim=-1)
    return torch.mean(l2norm)
    


    
"""    
criticD_real = netD(input_resv, input_attv)
criticD_real = criticD_real.mean()
criticD_real.backward(mone)
    
criticD_fake = netD(fake.detach(), input_attv)
criticD_fake = criticD_fake.mean()
criticD_fake.backward(one)    
    
gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
gradient_penalty.backward()    
"""    

"""
train D:
    loss_wasserstein = 
    loss_gp

train G:
    loss_wasserstein


"""