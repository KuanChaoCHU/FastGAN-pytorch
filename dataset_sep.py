# dataset total = 312
# bs = 12
# groups = 26
import os
import shutil
import numpy as np


def mkdirs(paths):
    """Create empty directories if they don't exist
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)
    return None


def mkdir(path):
    """Create a single
    """
    if not os.path.exists(path):
        os.makedirs(path)


DATASET_COUNT = 312
BATCH_SIZE = 12
ROOT = 'dataset/river_k_subset'
SRC_FOLDER = os.path.join(ROOT, 'full')

groups = DATASET_COUNT // BATCH_SIZE
group_index = np.full((groups, BATCH_SIZE), fill_value=-1)
group0 = [2,5,10,14,37,107,138,164,189,204,301,305]
group_index[0,:] = group0

all_images = np.arange(DATASET_COUNT) + 1
pool_images = np.delete(all_images, [i-1 for i in group0])
np.random.shuffle(pool_images)
print('shuf1')
np.random.shuffle(pool_images)
print('shuf2')
np.random.shuffle(pool_images)
print('shuf3')  # just for fun  
pool_images = pool_images.reshape(-1,BATCH_SIZE)
for l in range(len(pool_images)):
    pool_images[l] = np.sort(pool_images[l])

group_index[1:,:] = pool_images

for g in range(len(group_index)):
    imgs = group_index[g]
    new_folder = os.path.join(ROOT, f'group{g}')
    mkdir(new_folder)
    for i in imgs:
        fname = f'{i:0>3}.jpg'
        src_fpath = os.path.join(SRC_FOLDER, fname)
        dst_fpath = os.path.join(new_folder, fname) 
        shutil.copy(src_fpath, dst_fpath)
    
np.save(os.path.join(ROOT, 'group_index.npy'), group_index)
print(group_index)
