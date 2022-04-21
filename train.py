import torch
from torch import nn
import torchvision
from torchvision.transforms import transforms

from tqdm import tqdm
from config import CONFIG

from dataset import AshbyDataset
from utils import seed_everything

if CONFIG["wandb"]:
    import wandb
    wandb.init(project="ashby-hackathon")
seed_everything(42)

train_transforms = transforms.Compose([])
test_transforms = transforms.Compose([])

train_dataset = AshbyDataset(r"", transform=train_transforms)
test_dataset = AshbyDataset(r"", train=False, transform=test_transforms)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)

# Model
model = None

# Loss/Optimizer
crit = None
opt = None

for epoch in range(CONFIG['epochs']):
    # train loop
    model.train()
    for x,y in tqdm(train_dataloader):
        opt.zero_grad()
        x = x.cuda()
        y = y.cuda()
        
        pred = model(x)
        loss = crit(pred, y)

        loss.backward()
        opt.step()
       
    model.eval()
    with torch.no_grad():
        
        for x,y in tqdm(test_dataloader):
            x = x.cuda()
            y = y.cuda()
            
            pred = model(x)
            loss = crit(pred, y)
            
    # Logging
    if CONFIG["wandb"]:
        wandb.log({
            # metrics
        })