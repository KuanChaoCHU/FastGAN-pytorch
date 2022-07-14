import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def get_plot(x_axis, y_data_list, 
             x_label, y_label,
             labels,
             marker=None,
             line_style=None,
             title=None,
             axis=None,  # [x_min, x_max, y_min, y_max]
             xtick=False,
             grid=False,
             color=None,
             fpath=None):
    """ prototype...should be a complete class in the future 
    https://matplotlib.org/3.3.3/tutorials/intermediate/legend_guide.html
    https://stackoverflow.com/questions/10998621/rotate-axis-text-in-python-matplotlib
    line style and marker
    https://matplotlib.org/2.1.2/api/_as_gen/matplotlib.pyplot.plot.html
    """
    plot_kwargs = {}
    if marker is not None: plot_kwargs['marker'] = marker
    if line_style is not None: plot_kwargs['linestyle'] = line_style
    
    plt.figure()
    for idx, y_data in enumerate(y_data_list):
        if color is not None:
            plt.plot(x_axis, y_data, label=labels[idx], c=color[idx], **plot_kwargs)
        else:    
            plt.plot(x_axis, y_data, label=labels[idx], **plot_kwargs)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    if title is not None:
        plt.title(title)
    if axis:
        plt.xlim([axis[0], axis[1]])
        plt.ylim([axis[2], axis[3]])
    if xtick:
        plt.xticks(rotation=45)
    if grid:
        #https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.grid.html
        plt.grid(axis='both')
    plt.tight_layout()    
    if fpath:
        plt.savefig(fpath)
    plt.close()


exp_name = 'latent512_riverK'
FPATH = f'train_results/{exp_name}/fid_eval.npy'
fid = np.load(FPATH)
downarrow = u'\u2193'

get_plot(
    fid[:,0], [fid[:,1]], 'iteration', '', [f'{exp_name}'], 
    marker='x', title=f'FID({downarrow})', grid=True, fpath=f'fid_{exp_name}.png',
    axis=[0,50000,50,100]
)