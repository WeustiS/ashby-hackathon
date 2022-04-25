import torch
from torch import nn

from torchsummary import summary

from models.coatnet import CoAtNet_3d
from models.transf import Model

from tqdm import tqdm
from config import CONFIG_BIG as CONFIG

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
bn_model = CoAtNet_3d((c, 48, 160, 176), dims, blocks)
bn_model.load_state_dict({k[:]:v for k,v in torch.load("saves/model_final_hl79kxnu.pt").items()})

dims2 = [16, 32, 64, 128, emb_dim, 32, 16, 8, 4]
blocks2 = [2, 2, 3, 4, 5, 3, 3, 2, 2]
bn_model_dec = CoAtNet_3d((c, 48, 160, 176), dims2, blocks2, num_classes=1)

#bn_model.encoder.requires_grad = False # freeze encoder
encoder = nn.Sequential(
    bn_model.encoder.cuda(0)
)
#  img_size, dims, blocks, tok_dim, heads, encoder_layers, bn_model, 
model = Model((c, t,) + MODEL_SIZE, 
        dims,
        blocks,
        tok_dim=128,
        heads=4,
        encoder_layers=4,
        bn_model=bn_model_dec).cuda(0)
bn_dim = model.bn_dim

#summary(model, (t, model.tok_dim))

dataset = AshbyDataset(None,time_as_batch=False, slice_dims=CONFIG['tubelet_dim'], padding=CONFIG['tubelet_pad'], context='middle')
train_dataset= dataset 
test_dataset = AshbyEvalDataset(None,time_as_batch=False, slice_dims=CONFIG['tubelet_dim'], padding=CONFIG['tubelet_pad'], context='middle')

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True, prefetch_factor=1)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
# Loss/Optimizer
l2 = torch.nn.MSELoss()
l1 = torch.nn.L1Loss()
#opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0.01)
opt =  torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.03)

y_area = 1
for dim in CONFIG['tubelet_dim']:
    y_area *= dim
break_on = len(train_dataloader) // y_area
model.img_decoder = model.img_decoder.cuda(3)
for epoch in range(CONFIG['epochs']):
    # train loop
    model.train()
    train_l2 = 0
    train_l1 = 0
    test_l2 = 0
    test_l1 = 0
    i=0
    for x,y in tqdm(train_dataloader):
        b, c, t, z, h, w = x.shape
        opt.zero_grad()
        x = x.cuda(1)
        y = y.cuda()
        # b, c, t, z, h, w
        print(x.shape)
        x = x.permute([0,2,1,3,4,5]) # BTCZHW
        x = x.view(b*t, c, z, h, w) 
        x = nn.functional.interpolate(x, size=MODEL_SIZE, mode='trilinear', align_corners=True)
        x = x.cuda(0)
        x = encoder(x)
        
        # b*t, C, 1, 1, 1
        # b, t, c, z, h, w
        x = x.view(b,t,-1)
        
        # b, t, E
        pred = model(x)
        # b c z h w
       # pred = torch.mean(pred, dim=1)  # mean across the T (View) dimension
        #pred = pred[:, :, :39, :157, :167]
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
    with torch.no_grad():

        for x,y in tqdm(test_dataloader):
            y = y.cuda()

            x = x.cuda(1)
            x = x.permute([0,2,1,3,4,5]) # BTCZHW
            x = x.view(b*t, c, z, h, w) 
            x= nn.functional.interpolate(x, size=MODEL_SIZE, mode='trilinear', align_corners=True)
            x = x.cuda(0)
            x = encoder(x)
            x = x.view(b,t,-1)
            pred = model(x)
    
            pred = pred.cuda(2)
            pred = nn.functional.interpolate(pred, size=ORIG_SIZE, mode='trilinear', align_corners=True)
            pred = pred.cuda(0)

            l2_loss = l2(pred, y)
            l1_loss = l1(pred, y)

            test_l2 += l2_loss.item()/len(test_dataloader)
            test_l1 += l1_loss.item()/len(test_dataloader)

            
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
