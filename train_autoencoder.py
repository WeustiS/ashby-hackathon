import torch
from torch import nn

from torchsummary import summary

from models.coatnet import CoAtNet_3d

from tqdm import tqdm
from config import CONFIG_AUTOENC as CONFIG

from dataset import AshbyDataset
from dataset_eval import AshbyDataset as AshbyEvalDataset
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

c = 102 # 102
t, z, h, w = CONFIG['tubelet_dim']
emb_dim = 64
dims = [16, 32, 32, 64, emb_dim, 64, 32, 32, 16]
blocks = [1, 1, 1, 1, 1, 1, 1, 1, 1]
MODEL_SIZE =  (48, 160, 176)
ORIG_SIZE = (39, 157, 167)
model = CoAtNet_3d((c, 48, 160, 176), dims, blocks).cuda()
# 
# model = Model((c, t, 48, 160, 176), 
#         channels,
#         blocks,
#         tok_dim=256,
#         heads=8,
#         encoder_layers=8).cuda()
    

summary(model, ((102,) + (48, 160, 176)))

dataset = AshbyDataset(None,time_as_batch=True, slice_dims=CONFIG['tubelet_dim'], padding=CONFIG['tubelet_pad'], context='all')
lengths = [int(len(dataset)*.85), len(dataset) - int(len(dataset)*.85)]
train_dataset= dataset # torch.utils.data.random_split(dataset, lengths)
test_dataset = AshbyEvalDataset(None,time_as_batch=True, slice_dims=CONFIG['tubelet_dim'], padding=CONFIG['tubelet_pad'], context='middle')

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
print("Created Datasets")
# Loss/Optimizer
l2 = torch.nn.MSELoss()
l1 = torch.nn.L1Loss()
opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.1)

y_area = 1
for dim in CONFIG['tubelet_dim']:
    y_area *= dim
break_on = len(train_dataloader) // y_area
#break_on_test = len(test_dataloader) // y_area

for epoch in range(CONFIG['epochs']):
    # train loop
    model.train()
    train_l2 = 0
    train_l1 = 0
    test_l2 = 0
    test_l1 = 0
    i=0
    for x,y in tqdm(train_dataloader):

        b = len(x)
        opt.zero_grad()
        x = x.cuda(1)
        
        x= nn.functional.interpolate(x, size=MODEL_SIZE, mode='trilinear', align_corners=True)
        y = y.cuda()
        x = x.cuda(0)
        pred = model(x)
        pred = pred.cuda(2)
        pred = nn.functional.interpolate(pred, size=ORIG_SIZE, mode='trilinear', align_corners=True)
        pred = pred.cuda(0)
        l2_loss = l2(pred, y)
        l1_loss = l1(pred, y)
        
        train_l2 += l2_loss.item()/len(train_dataloader)
        train_l1 += l1_loss.item()/len(train_dataloader)
        
        loss = l2_loss + l1_loss

        loss.backward()
        opt.step()
        i+=1
#         if i == break_on:
#             break
       
    model.eval()
    i = 0
    with torch.no_grad():
        
        for x,y in tqdm(test_dataloader):
            y = y.cuda()
            
            x = x.cuda(1)
            x= nn.functional.interpolate(x, size=MODEL_SIZE, mode='trilinear', align_corners=True)
            x = x.cuda(0)
            
            pred = model(x)
            
            pred = pred.cuda(2)
            pred = nn.functional.interpolate(pred, size=ORIG_SIZE, mode='trilinear', align_corners=True)
            pred = pred.cuda(0)
            
            l2_loss = l2(pred, y)
            l1_loss = l1(pred, y)
            
            test_l2 += l2_loss.item()/len(test_dataloader)
            test_l1 += l1_loss.item()/len(test_dataloader)
            
            i += 1
            
    # Logging
    if CONFIG["wandb"]:
        wandb.log({
            'train_l1': train_l1,
             'test_l1': test_l1,
            'train_l2': train_l2,
             'test_l2': test_l2
        })
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"saves/model_{epoch}_{wandb.run.id}.pt")
        
torch.save(model.state_dict(), f"saves/model_final_{wandb.run.id}.pt")
#torch.save(otp.state_dict(), f"model_{wandb.run.id}.pt")
