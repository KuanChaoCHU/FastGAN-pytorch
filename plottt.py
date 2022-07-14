import matplotlib
matplotlib.use('Agg')
from util import utils
from plot import get_plot
import numpy as np


LRE = 5
COEF = 500

FPATH = f'train_results/IMV_n{LRE}_{COEF}/models/losslog.pkl'

raw = utils.read_from_pickle(FPATH)

lossV = []
lossM = []
loss = []
ep = np.arange(100)

for line in raw:
    data = line.split(']')
    
    if '45' in data[1]:
        losses = data[2].split()
        lossV.append(float(losses[1].split(',')[0]))
        lossM.append(float(losses[3].split(',')[0]))
        loss.append(float(losses[5]))


get_plot(
    ep, [lossV, lossM, loss],
    'epochs', '',
    ['loss_V', 'loss_M', 'loss_all'],
    grid=True,
    #axis=[0,40000,0.0,0.25],
    fpath=f'test_{LRE}_{COEF}.png'
)




"""
iters = []
pix = []
feat = []
adv = []
dist = []
z_max = []

raw = utils.read_from_pickle(FPATH)
for line in raw:
    iters.append(line[0])

    data = line[1].split()
    
    pix.append(float(data[1].split('=')[1].split(',')[0]))
    feat.append(float(data[2].split('=')[1].split(',')[0]))
    adv.append(float(data[3].split('=')[1].split(',')[0]))
    dist.append(float(data[4].split('=')[1].split(';')[0]))
    z_max.append(float(data[6].split('=')[1]))
"""

"""
get_plot(
    iters, [z_max],
    'iterations', 'z_max',
    ['z_max'],
    grid=True,
    #axis=[0,40000,0.0,0.25],
    fpath='testPPP2.png'
)
"""






