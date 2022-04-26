import torch
import torch.nn as nn

#from torchsummary import summary

from models.model import Model
from config import CONFIG
from dataset import AshbyDataset

import shap

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
            "P",
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

# class shap_explainer(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def explainer(self, train_data):
#         explainer = shap.Explainer(self.model)
#         values = explainer(train_data)
#         return (explainer, values)
    
if __name__ == '__main__':
    
    selected_features = ["TOT_NUM_CONC", "TOT_MASS_CONC","P","PB","TEMPERATURE","REL_HUMID","Z", "XLAT","XLON","dms"] #must have features
    
    dataset = AshbyDataset(features=selected_features)
    lengths = [int(len(dataset)*.85), len(dataset) - int(len(dataset)*.85)]
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths)
    print(train_dataset)
    #train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
    
#     model = Model((102, 16, 16, 32, 32), [32, 32,32,32], [2,2,2,2], [10, 10, 22, 22])
#     state_dict = torch.load("model_3jl5muan.pt", map_location=torch.device('cpu'))
#     state_dict = {k[7:]:v for k,v in state_dict.items()}
#     model.load_state_dict(state_dict)
#     model.eval()
    
    
#     explainer = shap.Explainer(model)
#     shap_values = explainer(train_dataset)
#     print(explainer)
    #print(shap_values)
#     s = shap_explainer(model)
#     (e, v) = s.explainer(data_set)
#     print(e)
#     print(v)
    
