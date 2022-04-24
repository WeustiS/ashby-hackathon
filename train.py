import torch
from torch import nn

from torchsummary import summary

from models.model import Model

from tqdm import tqdm
from config import CONFIG

from dataset import AshbyDataset
from utils import seed_everything

if CONFIG["wandb"]:
    import wandb
    wandb.init(project="ashby-hackathon", config=CONFIG)
seed_everything(42)

# train_transforms = transforms.Compose([])
# test_transforms = transforms.Compose([])
# T   D    H   W
# 133,39,157,167
# time_as_batch=False, train=True, transforms=None, slice_dims=(133, 39, 157, 167), padding=(0,0,0,0), context='before'                    

# Model
input_size = tuple(CONFIG['tubelet_dim'][i] + 2* CONFIG['tubelet_pad'][i] for i in range(4))
model = Model((102, ) + input_size, [32, 32, 32, 32], [2, 2, 2, 2], CONFIG['tubelet_dim']).cuda()
summary(model, (102, ) + input_size)


dataset = AshbyDataset(None, slice_dims=CONFIG['tubelet_dim'], padding=CONFIG['tubelet_pad'], context='all')
lengths = [int(len(dataset)*.85), len(dataset) - int(len(dataset)*.85)]
train_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)

model = torch.nn.DataParallel(model)
# Loss/Optimizer
l2 = torch.nn.MSELoss()
l1 = torch.nn.L1Loss()
opt = torch.optim.AdamW(model.parameters())

y_area = 1
for dim in CONFIG['tubelet_dim']:
    y_area *= dim
break_on = len(train_dataloader) // y_area
break_on_test = len(test_dataloader) // y_area

for epoch in range(CONFIG['epochs']):
    # train loop
    model.train()
    train_l2 = 0
    train_l1 = 0
    test_l2 = 0
    test_l1 = 0
    i=0
    for x,y in tqdm(train_dataloader):
        opt.zero_grad()
        x = x.cuda()
        y = y.cuda()
        
        pred = model(x)
     #   print(pred.shape, y.shape)
        l2_loss = l2(pred, y)
        l1_loss = l1(pred, y)
        
        train_l2 += l2_loss.item()
        train_l1 += l1_loss.item()
        
        loss = l2_loss + l1_loss

        loss.backward()
        opt.step()
        i+=1
        if i == break_on:
            break
       
    model.eval()
    i = 0
    with torch.no_grad():
        
        for x,y in tqdm(test_dataloader):
            x = x.cuda()
            y = y.cuda()
            
            pred = model(x)
            l2_loss = l2(pred, y)
            l1_loss = l1(pred, y)

            test_l2 += l2_loss.item()
            test_l1 += l1_loss.item()
            
            i += 1
            if i == break_on_test:
                break
            
    # Logging
    if CONFIG["wandb"]:
        wandb.log({
            'train_l1': train_l1,
            'test_l1': test_l1,
            'train_l2': train_l2,
            'test_l2': test_l2
        })
        
torch.save(model.state_dict(), f"model_{wandb.run.id}.pt")
#torch.save(otp.state_dict(), f"model_{wandb.run.id}.pt")
