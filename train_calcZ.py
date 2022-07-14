# reference: inverter.py
# calculate and optimize the latent vector for real image
# save them in .npy file without shuffle 
# batch_size = 1
import os
import argparse

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import lpips
from operation import ImageFolder
from operation import get_dir, load_params, convert_DP_state_dict
from models import Generator
from inv_model.stylegan_encoder_network import StyleGANEncoderNet
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)


def main(opt):
    data_root = opt.path
    dataset_count = opt.dataset_count  
    total_iterations = opt.iter
    checkpoint_e = opt.ckpt_e
    batch_size = 1
    im_size = opt.im_size
    ndf = 64
    ngf = 64
    nz = 512
    nlr = 0.01
    nbeta1 = 0.5
    dataloader_workers = 1
    current_iteration = 0
    save_interval = 100
    device = torch.device("cuda:0")
    saved_model_folder, saved_image_folder = get_dir(opt)

    transform_list = [
            transforms.Resize((int(im_size),int(im_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans = transforms.Compose(transform_list)

    dataset = ImageFolder(root=data_root, transform=trans)
    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=dataloader_workers, pin_memory=True))
    
    # Model G
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)
    ckpt_g = torch.load(opt.ckpt_g)
    netG.load_state_dict(convert_DP_state_dict(ckpt_g['g']))
    load_params(netG, ckpt_g['g_ema'])
    netG.to(device)
    del ckpt_g

    # Model E
    netE = StyleGANEncoderNet(resolution=im_size, w_space_dim=nz)
    ckpt_e = torch.load(opt.ckpt_e)
    netE.load_state_dict(ckpt_e['e'])
    netE.to(device)
    del ckpt_e 
    
    # Model F
    #netF = percept

    real_latent_vectors = np.zeros((dataset_count, nz), dtype='float32')
    real_latent_vectors_name = f'z_{opt.start_img_idx}_{opt.start_img_idx+dataset_count-1}.npy'
 
    m1_pix = 0.2
    m1_feat = 1.0

    m2_pix = 1.0
    m2_feat = 0.005
    m2_reg = 2.0
  
    m3_pix = 0
    m3_feat = 0
    m3_reg = 0

    for idx in range(dataset_count):
        if idx == 20:
            break
        img_idx = opt.start_img_idx + idx
        real_image = next(dataloader).to(device)

        if opt.mode == 2:
            init_z = torch.randn(1, nz, requires_grad=True, device=device)
        elif opt.mode == 1:
            with torch.no_grad():
                init_z = netE(real_image)
            init_z.requires_grad = True

        optimZ = torch.optim.Adam([init_z], lr=nlr, betas=(nbeta1, 0.999))

        for iteration in tqdm(range(current_iteration, total_iterations+1)):
            loss = 0.0
            optimZ.zero_grad()
            recon_images = netG(init_z)
            
            if opt.mode == 1:
                loss_m1 = percept(F.avg_pool2d(recon_images[0],2,2), F.avg_pool2d(real_image,2,2)).sum() * m1_feat + F.mse_loss(recon_images[0], real_image) * m1_pix
                loss = loss + loss_m1

            elif opt.mode == 2:
                # Reconstruction loss
                loss_pix = torch.mean((recon_images[0] - real_image) ** 2) * m2_pix   
                loss = loss + loss_pix

                # Perceptual loss
                loss_feat = percept(recon_images[0], real_image).sum() * m2_feat   
                loss = loss + loss_feat

                # Domain regularization loss
                z_hat = netE(recon_images[0])
                loss_reg = torch.mean((z_hat - init_z) ** 2) * m2_reg
                loss = loss + loss_reg

            elif opt.mode == 3:
                # Reconstruction loss
                loss_pix = torch.mean((recon_images[0] - real_image) ** 2) * m2_pix   
                loss = loss + loss_pix

                # Perceptual loss
                loss_feat = percept(recon_images[0], real_image).sum() * m2_feat   
                loss = loss + loss_feat

                # Domain regularization loss
                z_hat = netE(recon_images[0])
                loss_reg = torch.mean((z_hat - init_z) ** 2) * m2_reg
                loss = loss + loss_reg

            loss.backward()
            optimZ.step()
   
            # print, vis, save 
            if iteration % 20 == 0:
                if opt.mode == 1:
                    message = f'[Idx: {img_idx}] loss m1: {loss_m1.item():.3f}'
                elif opt.mode ==2:
                    message = f'[Idx: {img_idx}] loss: pix={loss_pix.item():.3f}, feat={loss_feat.item():.3f}, reg={loss_reg.item():.3f}'
                print(message)
            
            if iteration % 50 == 0 or iteration == total_iterations:
                with torch.no_grad():
                    vutils.save_image( torch.cat([
                        real_image,
                        netG(init_z)[0]
                    ]).add(1).mul(0.5), saved_image_folder+f'/{img_idx:0>3}_iter{iteration}.jpg' )

            if iteration == total_iterations:
                real_latent_vectors[idx, :] = init_z.detach().cpu().numpy()    

    # save latent vectors
    np.save(os.path.join(saved_model_folder, real_latent_vectors_name), real_latent_vectors)
    print('job complete')
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, default='dataset/river_k/img', help='')
    parser.add_argument('--dataset_count', type=int, default=307, help='rush, hand-crafted')
    parser.add_argument('--start_img_idx', type=int, default=1, help='should match the file name, e.g., 001.jpg')
    parser.add_argument('--iter', type=int, default=100, help='1k is enough?')
    
    parser.add_argument('--mode', type=int, default=2, help='1 or 2, different loss')
    parser.add_argument('--name', type=str, default='testCalcZm2test', help='experiment name')

    parser.add_argument('--im_size', type=int, default=512, help='')
    parser.add_argument('--ckpt_g', type=str, default='train_results/latent512_riverK/models/all_25000.pth', help='')
    parser.add_argument('--ckpt_e', type=str, default='train_results/NewEncoder/models/allEncoder_15000.pth', help='15k(no adv),25k,(35k)')
    
    opt, _ = parser.parse_known_args()
    print(opt)
    
    init_z = main(opt)