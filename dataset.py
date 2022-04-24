import torch
import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm
import xarray as xr

class AshbyDataset(torch.utils.data.Dataset):
    def __init__(self, features, time_as_batch=False, train=True, transforms=None, slice_dims=(133, 39, 157, 167), padding=(0,0,0,0), context='before'):
        
        '''
        features: a list of the names of all features you want to keep 
        time_as_batch: if we treat time as the batch dim. basically, if true then you get 3d outputs, if false then you get 4d outputs (with time as a dim)
        train: TODO- custom splits for train/test
        transforms: TODO - not yet impl
        slice_dims: the shape of the 'tubelet' that we extract. shape is TZHW. if time_as_batch then T is ignored (but note that T==1 is basically time_as_batch == True)
        padding: we may want to look at some input data 'around' our target output data. This controls that. order is  TZHW
        context: IFF not time_as_batch, are we trying to predict the:
            "before": give only last frame (all frames EXCEPT PADDING provided come before Y)
            "middle": give middle most frame 
            "after": give only first frame (all frames EXCEPT PADDING provided come after Y)
            "all": give all Y frames for t range
        '''
        assert context in ['before', 'middle', 'after', 'all'], "ask Eustis to add whatever context you want or use all and directly slice out"
        self.context = context
        self.all_x_features = self.all_x_features
        self.tube_t, self.tube_z, self.tube_h, self.tube_w = slice_dims
        self.max_shape = (0, 133, 0, 39, 1, 158, 1, 168)
        # (133, 39, 157, 167)
        self.padding = padding # t, z, h, w 
        
        self.all_x_features.sort() # for consistency 
        self.all_y_features.sort() # for consistency 
        
        self.features = self.all_x_features # todo support features as input
        
        data_dir = '/home/jcurtis2/hackathon_data/'
        wrf_filename = '%straining.nc' % data_dir
        xr_data = xr.open_dataset(wrf_filename)
        
        pt, pz, ph, pw = self.padding
        if time_as_batch:
            pt = 0
        print("Converting Dataset into Padded Tensor")
        
        # TODO 101 is hardcoded rn... needs to be len of features + 2 for lat/long
        x_data = torch.zeros(101, 133+2*pt, 39+2*pz, 157+2*ph, 167+2*pw)
        for i, feat in enumerate(tqdm(self.all_x_features)):
            x_data[i, pt:133+pt, pz:39+pz, ph:157+ph, pw:167+pw] = torch.as_tensor(xr_data[feat][0:133, 0:39, 1:158, 1:168].values)
        
        x_data[99, pt:133+pt, pz:39+pz, ph:157+ph, pw:167+pw] = torch.as_tensor(xr_data['XLAT'][0:133, 1:158, 1:168].values).unsqueeze(1).repeat(1,39,1,1)
        x_data[100, pt:133+pt, pz:39+pz, ph:157+ph, pw:167+pw] = torch.as_tensor(xr_data['XLONG'][0:133, 1:158, 1:168].values).unsqueeze(1).repeat(1,39,1,1)
        
        self.x_data = x_data
        
        y_data = torch.zeros(10, 133, 39, 157, 167)
        for i, feat in enumerate(tqdm(self.all_y_features)):
            y_data[i] = torch.as_tensor(xr_data[feat][0:133, 0:39, 1:158, 1:168].values)
        self.y_data = y_data
            
        self.time_as_batch = time_as_batch 
        self.train = train
        self.transforms = (lambda x: x) if transforms is None else transforms
        
        self.t_positions = 133 if time_as_batch else (self.max_shape[1]) - self.tube_t + 1
        self.z_positions = (self.max_shape[3]) - self.tube_z + 1
        self.h_positions = (self.max_shape[5]) - self.tube_h # + 1 - 1 (rm 1st idx)
        self.w_positions = (self.max_shape[7]) - self.tube_w # + 1 -1 (rm 1st idx)
         
        print("Found _ Positions for (T,Z,H,W)", self.t_positions, self.z_positions, self.h_positions, self.w_positions)
        
    def __len__(self):
        if self.time_as_batch:
            return self.max_shape[1] * self.z_positions * self.h_positions * self.w_positions # t z h w 
        else:
            return self.t_positions * self.z_positions * self.h_positions * self.w_positions
        
    def __getitem__(self, idx):
        t_idx = idx % self.t_positions 
        z_idx = (idx // self.t_positions) % self.z_positions
        h_idx = ((idx // self.t_positions) // self.z_positions) % self.h_positions
        w_idx = (((idx // self.t_positions) // self.z_positions) // self.h_positions) % self.w_positions
       # print(f"RAW IDX ({idx}) to TZHW:", t_idx, z_idx, h_idx, w_idx)
        pt, pz, ph, pw = self.padding
        if self.time_as_batch:
           # print(f"Loading X from \nT={t_idx} \nZ={z_idx} to {z_idx+self.tube_z+2*pz} \nH={1+h_idx} to {1+h_idx+self.tube_h+2*ph}\nW={1+w_idx} to {1+w_idx+self.tube_w+2*pw}")
           # print(f"Loading Y from \nT={t_idx} \nZ={z_idx} to {z_idx+self.tube_z} \nH={1+h_idx} to {1+h_idx+self.tube_h}\nW={1+w_idx} to {1+w_idx+self.tube_w}")
            x = self.x_data[  :,
                              t_idx, 
                              z_idx  :    z_idx+self.tube_z+2*pz, 
                            1+h_idx  :  1+h_idx+self.tube_h+2*ph, 
                            1+w_idx  :  1+w_idx+self.tube_w+2*pw
            ]

            y = self.y_data[               :,
                                    t_idx,
                              z_idx  :   z_idx+self.tube_z, 
                            1+h_idx  : 1+h_idx+self.tube_h, 
                            1+w_idx  : 1+w_idx+self.tube_w
            ]
        else:
           # print(f"Loading X from \nT={t_idx} to {t_idx+self.tube_t+2*pt} \nZ={z_idx} to {z_idx+self.tube_z+2*pz} \nH={1+h_idx} to {1+h_idx+self.tube_h+2*ph}\nW={1+w_idx} to {1+w_idx+self.tube_w+2*pw}")
            
            x = self.x_data[:,
                              t_idx:  t_idx+self.tube_t+2*pt, 
                              z_idx:  z_idx+self.tube_z+2*pz, 
                            1+h_idx:1+h_idx+self.tube_h+2*ph, 
                            1+w_idx:1+w_idx+self.tube_w+2*pw
            ]
            
            if self.context == "after":
             #   print(f"Loading Y from \nT={t_idx} \nZ={z_idx} to {z_idx+self.tube_z} \nH={1+h_idx} to {1+h_idx+self.tube_h}\nW={1+w_idx} to {1+w_idx+self.tube_w}")
                y = self.y_data[:,
                              t_idx,
                              z_idx:  z_idx+self.tube_z, 
                            1+h_idx:1+h_idx+self.tube_h, 
                            1+w_idx:1+w_idx+self.tube_w
                ]
            elif self.context == "before":
              #  print(f"Loading Y from \nT={t_idx+self.tube_t-1} \nZ={z_idx} to {z_idx+self.tube_z} \nH={1+h_idx} to {1+h_idx+self.tube_h}\nW={1+w_idx} to {1+w_idx+self.tube_w}")
                y = self.y_data[:,
                              t_idx+self.tube_t-1,
                              z_idx:  z_idx+self.tube_z, 
                            1+h_idx:1+h_idx+self.tube_h, 
                            1+w_idx:1+w_idx+self.tube_w
                ]
            elif self.context == "middle":
             #   print(f"Loading Y from \nT={(2*t_idx+self.tube_t)//2} \nZ={z_idx} to {z_idx+self.tube_z} \nH={1+h_idx} to {1+h_idx+self.tube_h}\nW={1+w_idx} to {1+w_idx+self.tube_w}")
                y = self.y_data[:,
                            (2*t_idx+self.tube_t)//2,
                            z_idx:  z_idx+self.tube_z, 
                            1+h_idx:1+h_idx+self.tube_h, 
                            1+w_idx:1+w_idx+self.tube_w
                ]
            else: # self.context == "all":
              #  print(f"Loading Y from \nT={t_idx} to {t_idx+self.tube_t}\nZ={z_idx} to {z_idx+self.tube_z} \nH={1+h_idx} to {1+h_idx+self.tube_h}\nW={1+w_idx} to {1+w_idx+self.tube_w}")
                y = self.y_data[:,
                            t_idx:  t_idx+self.tube_t,
                            z_idx:  z_idx+self.tube_z, 
                            1+h_idx:1+h_idx+self.tube_h, 
                            1+w_idx:1+w_idx+self.tube_w
                ]
        return x, y 
    
    all_y_features = [
        'ccn_001',
        'ccn_003',
        'ccn_006',
        'CHI',
        'CHI_CCN',
        'D_ALPHA',
        'D_GAMMA',
        "D_ALPHA_CCN",
        "D_GAMMA_CCN",
        "PM25"
    ]
    all_x_features = [
            "TOT_NUM_CONC",
            "TOT_MASS_CONC",
            "pmc_SO4",
            "pmc_NO3",
            "pmc_Cl",
            "pmc_NH4",
            "pmc_ARO1",
            "pmc_ARO2",
            "pmc_ALK1",
            "pmc_OLE1",
            "pmc_API1",
            "pmc_API2",
            "pmc_LIM1",
            "pmc_LIM2",
            "pmc_OC",
            "pmc_BC",
            "pmc_H2O",
            "PB", # TODO CHECK IF THIS IS RIGHT
            "TEMPERATURE",
            "REL_HUMID",
            "ALT",
            "Z",
            "h2so4",
            "hno3",
            "hcl",
            "nh3",
            "no",
            "no2",
            "no3",
            "n2o5",
            "hono",
            "hno4",
            "o3",
            "o1d",
            "O3P",
            "oh",
            "ho2",
            "h2o2",
            "co",
            "so2",
            "ch4",
            "c2h6",
            "ch3o2",
            "ethp",
            "hcho",
            "ch3oh",
            "ANOL",
            "ch3ooh",
            "ETHOOH",
            "ald2",
            "hcooh",
            "RCOOH",
            "c2o3",
            "pan",
            "aro1",
            "aro2",
            "alk1",
            "ole1",
            "api1",
            "api2",
            "lim1",
            "lim2",
            "par",
            "AONE",
            "mgly",
            "eth",
            "OLET",
            "OLEI",
            "tol",
            "xyl",
            "cres",
            "to2",
            "cro",
            "open",
            "onit",
            "rooh",
            "ro2",
            "ano2",
            "nap",
            "xo2",
            "xpar",
            "isop",
            "isoprd",
            "isopp",
            "isopn",
            "isopo2",
            "api",
            "lim",
            "dms",
            "msa",
            "dmso",
            "dmso2",
            "ch3so2h",
            "ch3sch2oo",
            "ch3so2",
            "ch3so3",
            "ch3so2oo",
            "ch3so2ch2oo",
            "SULFHOX",
       #     "XLAT",
         #   "XLONG"
        ]
    
if __name__ == "__main__":
    ds = AshbyDataset(None)