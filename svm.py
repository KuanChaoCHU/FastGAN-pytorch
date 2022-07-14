import numpy as np
from sklearn import svm

# load data
x_fpath = 'realZ.npy'
x = np.load(x_fpath)
 
flood_img_names = [
    2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 
    13, 14, 27, 66, 94, 96, 100, 111, 115, 116,
    121, 127, 136, 137, 138, 139, 142, 144, 150, 161,
    165, 175, 176, 182, 183, 186, 190, 193, 200, 207, 
    208, 219, 221, 227, 231, 232, 237, 238, 250, 259, 
    284, 285, 287, 303, 304, 305, 306 
]
y = np.zeros((len(x),), dtype='int64')
for img_idx in flood_img_names:
    y[img_idx-1] = 1


# SVM
reg_C = 1000
clf = svm.SVC(kernel='linear', C=reg_C)
clf.fit(x, y)

vector_w = clf.coef_[0]
unit_normal = vector_w / ((vector_w ** 2).sum() ** 0.5)




# save normal
np.save('normal.npy', unit_normal)