# Rewrite inv_util/inverter.py
#

import argparse
import random
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from tqdm import tqdm

from models import Generator, Discriminator, my_weights_init
from operation import load_params, get_dir, save_to_pickle
from operation import ImageFolder, InfiniteSamplerWrapper
from inv_model.stylegan_encoder_network import StyleGANEncoderNet
from diffaug import DiffAugment
import lpips
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)


POLICY = 'color,translation'


def convert_DP_state_dict(dp_state_dict):
    # convert the state_dict saved from DataParallel model 
    new_state_dict = OrderedDict()
    for k, v in dp_state_dict.items():
        name = k[7:]  # remove the 'module.' prefix
        new_state_dict[name] = v   
    return new_state_dict


def calc_loss_dist(feats_t):
    return (torch.mean(feats_t) - 0) ** 2.0 + (torch.std(feats_t) - 1) ** 2.0


def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]


def train_d(opt, net, data, label="real"):
    """Train function of discriminator"""
    if label=="real":
        part = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part] = net(data, label, part=part)
        err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() * opt.coef_D_adv 
        if opt.coef_D_self > 0.0:
            err_ss = percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
                     percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
                     percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
            err_ss = err_ss * opt.coef_D_self
            err = err + err_ss
        err.backward()
        return pred.mean().item(), rec_all, rec_small, rec_part
    else:
        pred = net(data, label)
        err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean() * opt.coef_D_adv
        err.backward()
        return pred.mean().item()


def main(opt):
    data_root = opt.path
    total_iterations = opt.iter    
    batch_size = opt.batch_size
    im_size = opt.im_size
    checkpoint = opt.ckpt
    ndf = 64
    ngf = 64
    nz = opt.noise_d
    nlr = 5e-6 * opt.lr_scale  #0.0002 * 0.01 
    nbeta1 = 0.5
    dataloader_workers = 8
    current_iteration = 0
    save_interval = 100
    saved_model_folder, saved_image_folder = get_dir(opt)
    device = torch.device("cuda:0")
    
    transform_list = [
            transforms.Resize((int(im_size),int(im_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # this only works on tensor
    ]
    trans = transforms.Compose(transform_list)

    dataset = ImageFolder(root=data_root, transform=trans)
    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=True))
    
    # Models: G, pretrained and fixed
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)  # nc fixed to 3    
    ckpt = torch.load(checkpoint)
    netG.load_state_dict(convert_DP_state_dict(ckpt['g']))  # 0908 I forgot to do this...
    load_params(netG, ckpt['g_ema'])
    netG.to(device)
    del ckpt
    
    # Models: E
    netE = StyleGANEncoderNet(
        resolution=im_size, w_space_dim=nz,
        use_wscale=opt.E_ws, use_bn=opt.E_bn
    )
    netE.apply(my_weights_init)
    netE.to(device)

    # Models: D
    netD = Discriminator(ndf=ndf, im_size=im_size)
    netD.apply(my_weights_init)
    netD.to(device)

    optimE = optim.Adam(netE.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))

    # For visualize
    fixed_realImg = next(dataloader).to(device)
    logs = []

    for iteration in tqdm(range(current_iteration, total_iterations+1)):
        if iteration == opt.iter_p2:
            print("Phase 2, change coef")
            opt.coef_D_adv = 1.0
            opt.coef_D_self = 0.0
            opt.coef_E_adv = 0.02  # finetune this
        
        real_image = next(dataloader)
        real_image = real_image.to(device)

        inv_z = netE(real_image)
        recon_images = netG(inv_z)
    
        #if opt.use_diffaug:
            #real_image = DiffAugment(real_image, policy=POLICY)
            #recon_images = [DiffAugment(recon, policy=POLICY) for recon in recon_images]

        # Train D
        netD.zero_grad()
        train_d(opt, netD, real_image, label='real')
        train_d(opt, netD, [rc.detach() for rc in recon_images], label='fake')
        optimD.step()

        # Train E
        netE.zero_grad()
        loss_E = 0.0
        
        loss_pix = torch.mean((real_image - recon_images[0]) ** 2) * 1.0
        loss_E = loss_E + loss_pix
         
        loss_feat = percept(recon_images[0], real_image).sum() * opt.coef_E_feat
        loss_E = loss_E + loss_feat 
        
        loss_adv = -netD(recon_images, "fake").mean() * opt.coef_E_adv
        loss_E = loss_E + loss_adv   
        
        loss_dist = calc_loss_dist(inv_z) * opt.coef_E_dist
        loss_E = loss_E + loss_dist

        loss_E.backward()         
        optimE.step()
                
        # print, visualize, save
        if iteration % 100 == 0:
            message = f'lossE: pix={loss_pix.item():.4f}, feat={loss_feat.item():.4f}, adv={loss_adv.item():.4f}, dist={loss_dist.item():.4f}; inv_z: max={inv_z.max():.4f}'
            print(message)
            logs.append([iteration, message])
           
        if iteration % (save_interval*10) == 0:
            with torch.no_grad():
                vutils.save_image( torch.cat([
                    fixed_realImg,
                    netG(netE(fixed_realImg))[0]
                ]).add(1).mul(0.5), saved_image_folder+'/rec_%d.jpg'%iteration )

        #'''
        #if iteration % (save_interval*50) == 0 or iteration == total_iterations:
        if iteration in [15000, 25000, 35000]:
            torch.save({'e': netE.state_dict(),
                        'opt_e': optimE.state_dict()}, saved_model_folder+'/allEncoder_%d.pth'%iteration)                 
        #'''

    save_to_pickle(saved_model_folder+'/losslog.pkl', logs)    
    return 


if __name__ == '__main__':

    # check paper for iter, loss coef
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, default='dataset/river_k/img', help='')
    parser.add_argument('--noise_d', type=int, default=512, help='or 256')
    parser.add_argument('--ckpt', type=str, default='train_results/latent512_riverK/models/all_25000.pth', help='')
    parser.add_argument('--im_size', type=int, default=512, help='')
    parser.add_argument('--name', type=str, default='NewEncoder2_woDss', help='change the name when for fromal run')

    parser.add_argument('--batch_size', type=int, default=8, help='')
    parser.add_argument('--iter', type=int, default=40000, help='30k? check paper')
    parser.add_argument('--iter_p2', type=int, default=16000, help='40% when to change the coef')
    parser.add_argument('--lr_scale', type=float, default=1.0, help='1.0')
    
    parser.add_argument('--coef_D_adv', type=float, default=0.0, help='')
    parser.add_argument('--coef_D_self', type=float, default=0.0, help='use selfsupervise or not')
    
    parser.add_argument('--coef_E_feat', type=float, default=0.00005*100, help='')
    parser.add_argument('--coef_E_adv', type=float, default=0.0, help='0.1')
    parser.add_argument('--coef_E_dist', type=float, default=0.1, help='')    
      
    parser.add_argument('--E_bn', type=bool, default=False, help='')
    parser.add_argument('--E_ws', type=bool, default=False, help='')
    parser.add_argument('--use_diffaug', type=bool, default=False, help='')

    opt, _ = parser.parse_known_args()
    print(opt)

    #latent = torch.rand((2,opt.noise_d), device='cuda:0')
    main(opt)
    #G, E, D, dataloader = main(opt)



    #else:
    #    ckpt = torch.load(checkpoint, map_location=lambda a,b: a)
    #    netG = nn.DataParallel(netG.to(device))
    #    netG.load_state_dict(ckpt['g'])
    