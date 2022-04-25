import torch
import numpy as np
from netCDF4 import Dataset
import cartopy
from tqdm import tqdm

class AshbyDataset(torch.utils.data.Dataset):
    def __init__(self, features, time_as_batch=False, train=True, transforms=None, slice_dims=(133, 39, 157, 167), padding=(0,0,0,0)):

        self.tube_t, self.tube_z, self.tube_h, self.tube_w = slice_dims
        self.max_shape = (0, 133, 0, 39, 1, 158, 1, 168)
        # (133, 39, 157, 167)
        self.padding = padding # t, z, h, w 
        
        self.all_x_features.sort() # for consistency 
        self.all_y_features.sort() # for consistency 
        
        self.features = self.all_x_features # todo support features as input
        
        data_dir = '/home/jcurtis2/hackathon_data/'
        wrf_filename = '%straining.nc' % data_dir
        ncf = Dataset(wrf_filename, "r", format="NETCDF4")
        
        print("Loading Dataset as Tensor")
        tensor_dataset = torch.empty(101, 133, 39, 157, 167)
        for i, feat in tqdm(self.all_x_features):
            tensor_dataset[:] = torch.as_tensor(ncf[feat][0:133, 0:39, 1:158, 1:168])
        
        # Added to the all_x_features
#         self.lats = ncf.variables['XLAT'][0,:,:]
#         self.lons = ncf.variables['XLONG'][0,:,:]
        
        self.ncf = ncf
        # TODO convert to tensor ahead of time
        
            
        self.time_as_batch = time_as_batch 
        self.train = train
        self.transforms = (lambda x: x) if transforms is None else transforms
        
        self.t_positions = 133 if time_as_batch else (self.max_shape[1]) - self.tube_t + 1
        self.z_positions = (self.max_shape[3]) - self.tube_z + 1
        self.h_positions = (self.max_shape[5]) - self.tube_h # + 1 - 1 (rm 1st idx)
        self.w_positions = (self.max_shape[7]) - self.tube_w # + 1 -1 (rm 1st idx)
         
        
        print("TZHW Positions", self.t_positions, self.z_positions, self.h_positions, self.w_positions)
        
        
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
        if self.time_as_batch:
            
            x = torch.zeros(len(self.features), self.tube_z, self.tube_h, self.tube_w) 
            for i, feature in enumerate(self.features):
                x[i, :] = torch.as_tensor(self.ncf.variables[feature][t_idx, 
                                                      z_idx:z_idx+self.tube_z, 
                                                      1+h_idx:1+h_idx+self.tube_h, 
                                                      1+w_idx:1+w_idx+self.tube_w])
            # TODO support x tubelet that has padding around y tubelet 
            y = torch.zeros(len(self.all_y_features), self.tube_z, self.tube_h, self.tube_w)
            pad_t, pad_z, pad_h, pad_w = self.padding
            for i, feature in enumerate(self.all_y_features):
                y[i, :] = torch.as_tensor(self.ncf.variables[feature][t_idx, 
                                                        pad_z+z_idx:  z_idx+self.tube_z-pad_z, 
                                                      1+pad_h+h_idx:1+h_idx+self.tube_h-pad_h, 
                                                      1+pad_w+w_idx:1+w_idx+self.tube_w-pad_w])
        else:
            x = torch.zeros(len(self.features), self.tube_t, self.tube_z, self.tube_h, self.tube_w) 
            for i, feature in enumerate(self.features):
                x[i, :] = torch.as_tensor(self.ncf.variables[feature][t_idx:t_idx+self.tube_t, 
                                                      z_idx:z_idx+self.tube_z, 
                                                      1+h_idx:1+h_idx+self.tube_h, 
                                                      1+w_idx:1+w_idx+self.tube_w])

            y = torch.zeros(len(self.all_y_features), self.tube_t, self.tube_z, self.tube_h, self.tube_w)
            pad_t, pad_z, pad_h, pad_w = self.padding
            for i, feature in enumerate(self.all_y_features):
                y[i, :] = torch.as_tensor(self.ncf.variables[feature][t_idx, 
                                                        pad_z+z_idx:  z_idx+self.tube_z-pad_z, 
                                                      1+pad_h+h_idx:1+h_idx+self.tube_h-pad_h, 
                                                      1+pad_w+w_idx:1+w_idx+self.tube_w-pad_w])
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
            "XLAT",
            "XLONG"
        ]
    
if __name__ == "__main__":
    ds = AshbyDataset(None)