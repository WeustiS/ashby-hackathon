import torch
from torch import nn
import torchvision
from torchvision.transforms import transforms
from torchsummary import summary

from models.model import Model

from tqdm import tqdm
from config import CONFIG

from dataset import AshbyDataset
from utils import seed_everything

if CONFIG["wandb"]:
    import wandb
    wandb.init(project="ashby-hackathon")
seed_everything(42)

# train_transforms = transforms.Compose([])
# test_transforms = transforms.Compose([])
# T   D    H   W
# 133,39,157,167
# time_as_batch=False, train=True, transforms=None, slice_dims=(133, 39, 157, 167), padding=(0,0,0,0), context='before'                    
dataset = AshbyDataset(None, slice_dims=CONFIG['tubelet_dim'], padding=CONFIG['tubelet_pad'])
lengths = [int(len(dataset)*.85), len(dataset) - int(len(dataset)*.85)]
train_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)

# Model
input_size = tuple(CONFIG['tubelet_dim'][i] + CONFIG['tubelet_pad'][i] for i in range(len(4)))
model = Model((101, ) + input_size, [64, 64, 64, 64], [2, 2, 2, 2], CONFIG['tubelet_dim']).cuda()
summary(model, (101, ) + input_size)
# Loss/Optimizer
l2 = torch.nn.MSELoss()
l1 = torch.nn.L1Loss()
opt = torch.optim.AdamW(model.params())

for epoch in range(CONFIG['epochs']):
    # train loop
    model.train()
    train_l2 = 0
    train_l1 = 0
    test_l2 = 0
    test_l1 = 0
    for x,y in tqdm(train_dataloader):
        opt.zero_grad()
        x = x.cuda()
        y = y.cuda()
        
        pred = model(x)
        l2_loss = l2(pred, y)
        l1_loss = l1(pred, y)
        
        train_l2 += l2_loss.item()
        train_l1 += l1_loss.item()
        
        loss = l2_loss + l1_loss

        loss.backward()
        opt.step()
       
    model.eval()
    with torch.no_grad():
        
        for x,y in tqdm(test_dataloader):
            x = x.cuda()
            y = y.cuda()
            
            pred = model(x)
            l2_loss = l2(pred, y)
            l1_loss = l1(pred, y)

            test_l2 += l2_loss.item()
            test_l1 += l1_loss.item()
            
    # Logging
    if CONFIG["wandb"]:
        wandb.log({
            'train_l1': train_l1,
            'test_l1': test_l1,
            'train_l2': train_l2,
            'test_l2': test_l2
        })