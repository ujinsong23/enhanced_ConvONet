import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import torch
from src import config, data
import argparse

# Arguments
parser = argparse.ArgumentParser(
    description='Plot graphs(train error, iou)'
)
parser.add_argument('--plot-mode', type=str, default=None,
                    help='Specify the encoder type as one of "sum", "range", "mix", or "none".')

args = parser.parse_args()
assert args.plot_mode in ['sum','range','mix', 'none',None], f"Mode is invalid."

### Plot all runs
if args.plot_mode==None:
    cfg_dirs = ['configs/pointcloud/shapenet_3plane_indoor_weightnone.yaml',
                'configs/pointcloud/shapenet_3plane_indoor_weightsum.yaml',
                'configs/pointcloud/shapenet_3plane_indoor_weightrange.yaml',
                'configs/pointcloud/shapenet_3plane_indoor_weightmix.yaml',
                ]

    train_losses_all = dict()
    validation_iou_all =dict()
    for cfg_dir in cfg_dirs:
        cfg = config.load_config(cfg_dir, 'configs/default.yaml')
        mode_name = cfg_dir[42+6:-5]
        if mode_name=='none':
            mode_name = 'baseline'
        out_dir = cfg['training']['out_dir'] 

        with open(os.path.join(out_dir, 'train_losses.pkl'), 'rb') as fp:
            train_losses = pickle.load(fp)
            train_losses_all[mode_name]=train_losses
        with open(os.path.join(out_dir, 'validation_iou.pkl'), 'rb') as fp:
            validation_iou = pickle.load(fp)
            validation_iou_all[mode_name]=validation_iou

    train_losses_all.keys(), validation_iou_all.keys()

    kwargs = {'sum':{'color':'darkblue'},
              'range':{'color':'lightseagreen'},
              'mix':{'color':'plum'},
              'baseline':{'color':'darkgrey', 'linewidth':3, 'alpha':0.7}}

    fig, ax = plt.subplots(2,1,figsize=(6,10))
    for k in train_losses_all:
        train_losses = train_losses_all[k]
        validation_iou = validation_iou_all[k]
        ax[0].plot(train_losses.keys(),train_losses.values(),
                    label=k,**(kwargs[k]))
        ax[0].axis(ymax=50)
        ax[0].tick_params(axis='x', which='minor', bottom=False)
        ax[1].plot(validation_iou.keys(),validation_iou.values(),
                    label=k,**(kwargs[k]))
        
    ax[0].set_title('Train Error by iterations',fontsize=14)
    ax[1].set_title('Validation iou by iterations',fontsize=14)
    for i in range(2):
        ax[i].grid()
        ax[i].legend()
        ax[i].set_xlabel('iterations')

    filename = 'plot_vertical.png'
    plt.savefig(filename, bbox_inches='tight')
    print(f'Saved results from all run at : \'{filename}\'')


## Plot specific run
else: 
    mode = args.plot_mode
    cfg_dir = f'configs/pointcloud/shapenet_3plane_indoor_weight{mode}.yaml'
    cfg = config.load_config(cfg_dir, 'configs/default.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = cfg['training']['out_dir'] 

    with open(os.path.join(out_dir, 'train_losses.pkl'), 'rb') as fp:
        train_losses = pickle.load(fp)

    with open(os.path.join(out_dir, 'validation_iou.pkl'), 'rb') as fp:
        validation_iou = pickle.load(fp)

    # print(train_losses)
    # print(validation_iou)

    plt.figure(1)
    plt.plot(train_losses.keys(),train_losses.values())
    plt.xlabel('# iterations')
    plt.ylabel('loss')
    plt.grid()
    plt.savefig(os.path.join(out_dir, f'train_losses_{mode}.png'), bbox_inches='tight')
    print(f'Saved results from all run at : \'train_losses_{mode}.png\'')

    plt.figure(2)
    plt.plot(validation_iou.keys(),validation_iou.values(),label='validation iou')
    plt.xlabel('# iterations')
    plt.ylabel('iou')
    plt.grid()
    plt.savefig(os.path.join(out_dir, f'validation_iou_{mode}.png'), bbox_inches='tight')
    print(f'Saved results from all run at : \'validation_iou_{mode}.png\'')