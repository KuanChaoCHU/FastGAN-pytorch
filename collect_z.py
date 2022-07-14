# collect the calculated z from each group
import numpy as np

group_index_fpath = 'dataset/river_k_subset/group_index.npy'
group_index = np.load(group_index_fpath)  # image name, min = 1

DATASET_COUNT = 312
BATCH_SIZE = 12
FEATURE_DIM = 256

merged_data = np.full((DATASET_COUNT, FEATURE_DIM), fill_value=-100.0, dtype='float32')
img_names = set()

for i in range(len(group_index)):
    seg_fpath = f'train_results/realzG{i}/models/g{i}.npy'
    
    seg_data = np.load(seg_fpath)
    seg_img_name = group_index[i]
    for j in range(len(seg_img_name)):
        merged_data[seg_img_name[j]-1] = seg_data[j]
        assert seg_data[j].size == FEATURE_DIM
        print(f'Idx: {seg_img_name[j]} done')
        img_names.add(seg_img_name[j])

print(len(img_names), max(img_names), min(img_names))
np.save('realZ.npy', merged_data)


