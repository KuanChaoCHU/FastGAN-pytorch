import os
import argparse
from PIL import Image

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

from models import Generator
from operation import get_dir, load_params, convert_DP_state_dict


def main(args):
    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    nz = 256
    nlr = 0.0002
    nbeta1 = 0.5
    use_cuda = True
    multi_gpu = False
    dataloader_workers = 8
    current_iteration = 0
    save_interval = 100
    saved_model_folder, saved_image_folder = get_dir(args)
    
    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:0")

    transform_list = [
            transforms.Resize((int(im_size),int(im_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans = transforms.Compose(transform_list)

    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)    
    ckpt = torch.load(checkpoint)
    netG.load_state_dict(convert_DP_state_dict(ckpt['g']))
    load_params(netG, ckpt['g_ema'])
    #netG.eval() unstable, harms
    netG.to(device)
    del ckpt

    sample_img_fpath = os.path.join(args.path, f'{args.image_name:0>3}.jpg')
    sample_img = Image.open(sample_img_fpath)
    sample_img = trans(sample_img).unsqueeze(dim=0).cuda()

    realZ = np.load(args.realZ_fpath)
    sample_z = realZ[int(args.image_name) -1].reshape(1,-1)

    normal = np.load(args.normal_fpath).reshape(1,-1)
    normal_t = torch.from_numpy(normal).float().cuda()

    fixed_noise_fp = 'train_results/realzG0/models/g0.npy'
    fixed_noise = np.load(fixed_noise_fp)
    fixed_noise_t = torch.from_numpy(fixed_noise).cuda()

    a3z = fixed_noise_t + args.alpha * 5000 * normal_t
    a2z = fixed_noise_t + args.alpha * 3000 * normal_t
    a1z = fixed_noise_t + args.alpha * 1000 * normal_t
    a0z = fixed_noise_t + args.alpha * 0 * normal_t
    b1z = fixed_noise_t + args.alpha * -1000 * normal_t
    b2z = fixed_noise_t + args.alpha * -3000 * normal_t
    b3z = fixed_noise_t + args.alpha * -5000 * normal_t

    def syn(z, G=netG, idx=7):
        with torch.no_grad():
            img_t = G(z)[0][idx].unsqueeze(dim=0)
        return img_t
 
    # 0915 I see the problem. Because BN is in training mode. should load the full batch


    with torch.no_grad():
        vutils.save_image( torch.cat([
            sample_img, syn(a3z), syn(a2z), syn(a1z), syn(a0z), syn(b1z), syn(b2z), syn(b3z)
            ]).add(1).mul(0.5), saved_image_folder + f'/Edited_{args.image_name:0>3}.jpg')
        
    return sample_z, normal   
 
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='image edit')
    parser.add_argument('--normal_fpath', type=str, default='normal.npy', help='')
    parser.add_argument('--realZ_fpath', type=str, default='realZ.npy', help='')
    parser.add_argument('--path', type=str, default='dataset/river_k_subset/full', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--image_name', type=int, default=164, help='')
    parser.add_argument('--alpha', type=float, default=0.05, help='')

    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='editsample', help='experiment name')
    parser.add_argument('--group', type=int, default=0, help='')
    parser.add_argument('--iter', type=int, default=50000, help='(50k) number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=1, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=512, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='train_results/testRiverK/models/all_30000.pth', help='(None is invaild in this script) checkpoint weight path')

    args = parser.parse_args()
    print(args)

    x = main(args)