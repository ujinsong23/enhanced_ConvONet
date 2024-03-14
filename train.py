from tqdm import tqdm
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
import argparse
import time, datetime
import matplotlib; matplotlib.use('Agg')
from src import config, data
from src.checkpoints import CheckpointIO
from collections import defaultdict
import shutil
import pickle

# Arguments
parser = argparse.ArgumentParser(
    description='Train a model using a different encoder.'
)
parser.add_argument('--encoder-type', type=str, 
                    help='Specify the encoder type as one of "sum", "range", "mix" or "none".')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')

args = parser.parse_args()
assert args.encoder_type in ['sum','range','none',"mix"], \
        "mode is invalid. encoder type is one of 'sum', 'range', 'mix' or 'none'."

# Load config
cfg_dir = f'configs/pointcloud/shapenet_3plane_indoor_weight{args.encoder_type}.yaml'
cfg = config.load_config(cfg_dir, 'configs/default.yaml')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

out_dir = cfg['training']['out_dir'] 
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
shutil.copyfile(cfg_dir, os.path.join(out_dir, 'config.yaml'))

batch_size = cfg['training']['batch_size'] # 32
backup_every = cfg['training']['backup_every'] # 10000
vis_n_outputs = cfg['generation']['vis_n_outputs'] # 2
exit_after = args.exit_after

# Set t0
t0 = time.time()

model_selection_metric = cfg['training']['model_selection_metric']
model_selection_sign = 1

train_dataset = config.get_dataset('train', cfg) 
val_dataset = config.get_dataset('val', cfg, return_idx=True) 

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=cfg['training']['n_workers'], shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=cfg['training']['n_workers_val'], shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

vis_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)


model_counter = defaultdict(int)
data_vis_list = []

# Build a data dictionary for visualization
iterator = iter(vis_loader)
for i in tqdm(range(len(vis_loader)), desc='Building a data dictionary for visualization'):
    data_vis = next(iterator)
    idx = data_vis['idx'].item()
    model_dict = val_dataset.get_model_dict(idx)
    category_id = model_dict.get('category', 'n/a')
    category_name = val_dataset.metadata[category_id].get('name', 'n/a')
    category_name = category_name.split(',')[0]
    if category_name == 'n/a':
        category_name = category_id

    c_it = model_counter[category_id]
    if c_it < vis_n_outputs:
        data_vis_list.append({'category': category_name, 'it': c_it, 'data': data_vis})

    model_counter[category_id] += 1

# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)

# Generator
generator = config.get_generator(model, cfg, device=device)

# Intialize training
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
trainer = config.get_trainer(model, optimizer, cfg, device=device)

checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
try:
    load_dict = checkpoint_io.load('model.pt')
except FileExistsError:
    load_dict = dict()

epoch_it = load_dict.get('epoch_it', 0)
it = load_dict.get('it', 0)
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf
print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))
logger = SummaryWriter(os.path.join(out_dir, 'logs'))

# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']
print(f'\nIteration Info:\n-print every {print_every} iter')
print(f'-checkpoint every {checkpoint_every} iter')
print(f'-validate every {validate_every} iter')
print(f'-visualize every {visualize_every} iter')

# Print model
print('\nModel/data Info:')
print(f'-Enocder type : {args.encoder_type}')
print(f'-Train data  {len(train_dataset)}, Validation data: {len(val_dataset)}')
print(f'-Batch size: {batch_size}, {len(train_dataset)/batch_size:.2f} iterations per epoch')
nparameters = sum(p.numel() for p in model.parameters())
print('-Total number of parameters: %d' % nparameters)
print('\nOutput path: ', cfg['training']['out_dir'])
print()


train_losses = dict()
validation_iou = dict()
losses = []
while (epoch_it<100):
    epoch_it += 1
    for batch in train_loader:
        it += 1
        loss = trainer.train_step(batch)
        logger.add_scalar('train/loss', loss, it)
        losses.append(loss)

        # Print output
        if print_every > 0 and (it % print_every) == 0:
            t = datetime.datetime.now()
            print('[Epoch %02d] it=%03d, loss=%.4f, time: %.2fs, %02d:%02d'
                     % (epoch_it, it, loss, time.time() - t0, t.hour, t.minute))
            train_losses[it]=np.mean(losses)
            losses = []

        # Visualize output
        if visualize_every > 0 and (it % visualize_every) == 0:
            print('Visualizing')
            for data_vis in data_vis_list:
                if cfg['generation']['sliding_window']:
                    out = generator.generate_mesh_sliding(data_vis['data'])    
                else:
                    out = generator.generate_mesh(data_vis['data'])
                # Get statistics
                try:
                    mesh, stats_dict = out
                except TypeError:
                    mesh, stats_dict = out, {}

                mesh.export(os.path.join(out_dir, 'vis', '{}_{}_{}.off'.format(it, data_vis['category'], data_vis['it'])))

        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print('Saving checkpoint')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            print('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
        # Run validation
        if validate_every > 0 and (it % validate_every) == 0:
            eval_dict = trainer.evaluate(val_loader)
            metric_val = eval_dict[model_selection_metric]
            print('Validation metric (%s): %.4f'
                  % (model_selection_metric, metric_val))
            validation_iou[it]=metric_val

            for k, v in eval_dict.items():
                logger.add_scalar('val/%s' % k, v, it)

            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                print('New best model (loss %.4f)' % metric_val_best)
                checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)

        # Exit if necessary
        if exit_after > 0 and (time.time() - t0) >= exit_after:
            print('Time limit reached. Exiting.')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            exit(3)

print(f'Train (it: {it}, epoch_it: {epoch_it}) ended')
print(f'Elapsed {time.time()-t0:.2f}sec')

# Save 
with open(os.path.join(out_dir, 'train_losses.pkl'), 'wb') as fp:
    pickle.dump(train_losses, fp)

with open(os.path.join(out_dir, 'validation_iou.pkl'), 'wb') as fp:
    pickle.dump(validation_iou, fp)