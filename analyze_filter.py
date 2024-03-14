import torch
from src.encoder.unet import UNet
import matplotlib.pyplot as plt

# Load checkpoint
checkpoint = torch.load('out/pointcloud/shapenet_3plane_indoor_weightmix/model_best.pt')
ckpt = dict()
for k in checkpoint['model']:
    if 'conv_z' in k:
        ckpt[k[15:]]=checkpoint['model'][k]

conv_z = UNet(1, in_channels=4, depth=2, start_filts=4)
conv_z.load_state_dict(ckpt)


# Initialize 4 planes for comparison
plane = torch.zeros((1,4,64,64))
res = []
for i in range(4):
    plane[:,i,:,:] = torch.ones(1,64,64)
    with torch.no_grad():
        res.append(conv_z(plane)[0,0].detach().cpu())
    plane[:,i,:,:] = torch.zeros(1,64,64)

fig, ax = plt.subplots(1,4,figsize=(12,5))
names = ['max','min','mean','sum']
for i in range(4):
    im = ax[i].imshow(res[i], cmap='gray')
    ax[i].set_title(names[i])
fig.colorbar(im, ax=ax.ravel().tolist(),
                fraction=0.01, pad=0.04)

fig.savefig('filter_analysis.png',bbox_inches='tight')