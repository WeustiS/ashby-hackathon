CONFIG = {
    "batch_size": 10,
    "num_workers": 10,
    "epochs": 10, 
    "tubelet_dim": (10, 10, 22, 22), # output shape
    "tubelet_pad": (3, 3, 5, 5), # 20, 20, 40, 40 | 10 10 20 20 | 5 5 10 10 | 
    "wandb": True
} 
# 16 16 32 32 
# 10 10 22 22