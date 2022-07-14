# seems the first channel is x-axis...but the sign...is at 2nd qradrant!?
import os
import json
import argparse
from PIL import Image

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as vf
from torchvision import utils as vutils
from scipy.io import loadmat

from maskflow_model import netblocks
from operation import get_dir, save_to_pickle
from diffaug import DiffAugment
from models import UpBlockComp_mine, my_weights_init


DATA_ROOT = 'dataset/river_fukui'
IM_SIZE = 512
JSON_FNAME = 'instances_fukui.json'
BATCH_SIZE = 8
HFLIP_RATE = 0.5
POLICY = 'color'
DEVICE = torch.device("cuda:0")


transform_list1 = [
    transforms.Resize((int(IM_SIZE),int(IM_SIZE))),
    transforms.ToTensor(),  # from Image: range 0~1
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # from Image: range -1~1
]
img_trans = transforms.Compose(transform_list1)


def transform_image(fpath, trans=img_trans):
    raw_image = Image.open(fpath)
    return trans(raw_image)


def transform_mask(fpath, im_size=IM_SIZE):
    raw_image = Image.open(fpath)
    resized_image = vf.resize(raw_image, (int(im_size),int(im_size)))
    resized_image_t = torch.from_numpy(np.array(resized_image)).float().unsqueeze(dim=0)
    return resized_image_t
 

def transform_flow(fpath, im_size=IM_SIZE):
    raw_ndarray = loadmat(fpath)['flow']
    resized_ndarray = cv2.resize(raw_ndarray, dsize=(im_size, im_size), interpolation=cv2.INTER_CUBIC)
    resized_image_t = torch.from_numpy(resized_ndarray)
    return resized_image_t.permute(2,0,1)


class MaskFlowDataset(Dataset):
    """ The map-style dataset. implement __getitem()__ and __len()__
    """
    def __init__(self, json_fpath, 
                 trans_image_fn, trans_mask_fn, trans_flow_fn,
                 hfilp_rate=0.5):        
        with open(json_fpath) as f:
            self.instances_json = json.load(f)    
        self.trans_image_fn = trans_image_fn
        self.trans_mask_fn = trans_mask_fn
        self.trans_flow_fn = trans_flow_fn
        
        self.hfilp_rate = hfilp_rate
        self.T_hfilp = transforms.RandomHorizontalFlip(p=1.0)
        
    def __getitem__(self, index):
        sample = self.instances_json['instances'][index]
        image = self.trans_image_fn(sample['image'])
        mask = self.trans_mask_fn(sample['mask'])
        flow = self.trans_flow_fn(sample['flow'])
        next_mask = self.trans_mask_fn(sample['next_mask'])
        next_flow = self.trans_flow_fn(sample['next_flow'])
        
        if torch.rand(1) < self.hfilp_rate:
            image = self.T_hfilp(image)
            mask = self.T_hfilp(mask)
            next_mask = self.T_hfilp(next_mask)
            flow = self.T_hfilp(flow)
            flow[0,...] = -flow[0,...]
            next_flow = self.T_hfilp(next_flow)
            next_flow[0,...] = -next_flow[0,...]

        return image, mask, flow, next_mask, next_flow

    def __len__(self):
        return len(self.instances_json['instances'])

    def pre_shuffle(self):
        pass





def lossV(pred_V_next, real_V_next, M):
    # V shape: (B, 2, H, W), M shape: (B, 1, H, W)
    # l1-loss
    loss_x = (torch.abs(pred_V_next[:,0,...] - real_V_next[:,0,...]).unsqueeze(dim=1) * M).mean()
    loss_y = (torch.abs(pred_V_next[:,1,...] - real_V_next[:,1,...]).unsqueeze(dim=1) * M).mean()
    return loss_x + loss_y


def lossM(pred_M_diff, real_M_next, real_M):
    real_M_diff = real_M_next - real_M
    loss = torch.abs(pred_M_diff - real_M_diff).mean()
    return loss


#x = DiffAugment(x, policy=POLICY)

class ImageMaskFlowModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.netFE = netblocks.resnet50(pretrained=True, global_pool=False,
                                        include_last_fc=False, fpn_level=1)

        self.upblock1 = UpBlockComp_mine(2048, 512)
        self.upblock2 = UpBlockComp_mine(512, 128)
        self.upblock3 = UpBlockComp_mine(128, 32)
        self.upblock_a1 = UpBlockComp_mine(32, 8)
        self.upblock_a2 = UpBlockComp_mine(8, 4)
        self.upblock_b1 = UpBlockComp_mine(32, 8)
        self.upblock_b2 = UpBlockComp_mine(8, 2)
        self.lastblock_a = nn.Conv2d(4, 2, 1)
        self.lastblock_b = nn.Conv2d(2, 1, 1)
        
        self.init_names = [
            'upblock1', 'upblock2', 'upblock3', 'upblock_a1', 'upblock_a2',
            'upblock_b1', 'upblock_b2', 'lastblock_a', 'lastblock_b'
        ]
        for name in self.init_names:
            getattr(self, name).apply(my_weights_init)

    def forward(self, I, M, V):
        featmap = self.netFE(I)[0]
        out = self.upblock1(featmap)
        out = self.upblock2(out)
        out = self.upblock3(out)  # (B, 32, 128, 128)
        
        out_a = self.upblock_a1(out)
        out_a = self.upblock_a2(out_a)
        out_a = M * out_a
        out_a = self.lastblock_a(out_a)

        out_b = self.upblock_b1(out)
        out_b = self.upblock_b2(out_b)
        out_b = V * out_b
        out_b = self.lastblock_b(out_b)
        out_b = torch.tanh(out_b)        
        return out_a, out_b  # pred V_t+1(M_t), pred delta_M

 


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='IMV_test', help='change the name when for fromal run')
    parser.add_argument('--epochs', type=int, default=100, help='')
    parser.add_argument('--lr', type=float, default=1e-3, help='1e-3 1e-4 1e-5')
    parser.add_argument('--coef_lossM', type=float, default=10.0, help='10 or 100')
    opt, _ = parser.parse_known_args()

    saved_model_folder, saved_image_folder = get_dir(opt)

    dataset = MaskFlowDataset(os.path.join(DATA_ROOT, JSON_FNAME), 
                              transform_image, transform_mask, transform_flow)    
    
    model = ImageMaskFlowModel()
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    log = []

    for epoch in range(opt.epochs):
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=8, pin_memory=True)        
        loss_epoch_value = 0.0
        batch_count = 0

        for b_idx, data in enumerate(dataloader):
            loss_batch = 0.0

            data[0] = DiffAugment(data[0], policy=POLICY)
            data = [d.to(DEVICE) for d in data]

            optimizer.zero_grad()
            predV, predM = model(data[0], data[1], data[2])
            loss_V = lossV(predV, data[4], data[1])
            loss_M = lossM(predM, data[3], data[1])
            loss_batch = loss_batch + loss_V + loss_M
            loss_epoch_value = loss_epoch_value + loss_batch.item()
            batch_count += 1
            loss_batch.backward()
            optimizer.step()

            # print & log
            message = f'[Ep {epoch}/{opt.epochs-1}][Batch_id: {b_idx}] loss_V: {loss_V.item():.4f}, loss_M: {loss_M.item():.4f}, loss_ep: {loss_epoch_value/batch_count:.4f}'
            print(message)
            log.append(message)
            
    save_to_pickle(saved_model_folder+'/losslog.pkl', log)        
            




