import torch
import numpy as np
from tqdm import tqdm
import torchvision

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
        
        norm_const_arr = [(k, v) for k, v in self.norm_consts.items()]
        norm_const_arr.sort(key=lambda x: x[0])
        
        norm_const_arr_test = [(k, v) for k, v in self.test_norm_consts.items()]
        norm_const_arr_test.sort(key=lambda x: x[0])
        
        x_raw = torch.load("x.pt")

        for i in range(len(norm_const_arr)):
            x_raw[i, :] = (x_raw[i] - norm_const_arr[i][1][0])/norm_const_arr[i][1][1]
        

        y_data = torch.load("y.pt")
        for i in range(len(norm_const_arr_test)):
            y_data[i, :] -= norm_const_arr_test[i][1][0]
            y_data[i, :] /= norm_const_arr_test[i][1][1]
        
        pt, pz, ph, pw = self.padding
        if time_as_batch:
            pt = 0
        print("Converting Dataset into Padded Tensor")
        

        x_data = torch.zeros(102, 133+2*pt, 39+2*pz, 157+2*ph, 167+2*pw)
        x_data[:, pt:133+pt, pz:39+pz, ph:157+ph, pw:167+pw] = x_raw[:, :, :, 1:-1, 1:-1] # TODO CROP
        self.x_data = x_data
        
        self.y_data = y_data[:, :, :, 1:-1, 1:-1]
        
        self.time_as_batch = time_as_batch 
        self.train = train
        self.transforms = (lambda x: x) if transforms is None else transforms
        
        self.t_positions = 133 if time_as_batch else (self.max_shape[1]) - self.tube_t + 1
        self.z_positions = (self.max_shape[3]) - self.tube_z + 1
        self.h_positions = (self.max_shape[5]) - self.tube_h # + 1 - 1 (rm 1st idx)
        self.w_positions = (self.max_shape[7]) - self.tube_w # + 1 -1 (rm 1st idx)
         
        print("Found the following number of positions for (T,Z,H,W)", self.t_positions, self.z_positions, self.h_positions, self.w_positions)
        
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
    #    print(f"RAW IDX ({idx}) to TZHW:", t_idx, z_idx, h_idx, w_idx)
        pt, pz, ph, pw = self.padding
        if self.time_as_batch:
          #  print(f"Loading X from \nT={t_idx} \nZ={z_idx} to {z_idx+self.tube_z+2*pz} \nH={1+h_idx} to {1+h_idx+self.tube_h+2*ph}\nW={1+w_idx} to {1+w_idx+self.tube_w+2*pw}")
          #  print(f"Loading Y from \nT={t_idx} \nZ={z_idx} to {z_idx+self.tube_z} \nH={1+h_idx} to {1+h_idx+self.tube_h}\nW={1+w_idx} to {1+w_idx+self.tube_w}")
            x = self.x_data[  :,
                              t_idx, 
                              z_idx  :    z_idx+self.tube_z+2*pz, 
                            h_idx  :  h_idx+self.tube_h+2*ph, 
                            w_idx  :  w_idx+self.tube_w+2*pw
            ]

            y = self.y_data[               :,
                                    t_idx,
                              z_idx  :   z_idx+self.tube_z, 
                            h_idx  :   h_idx+self.tube_h, 
                            w_idx  : w_idx+self.tube_w
            ]
        else:
         #   print(f"Loading X from \nT={t_idx} to {t_idx+self.tube_t+2*pt} \nZ={z_idx} to {z_idx+self.tube_z+2*pz} \nH={1+h_idx} to {1+h_idx+self.tube_h+2*ph}\nW={1+w_idx} to {1+w_idx+self.tube_w+2*pw}")
            
            x = self.x_data[:,
                              t_idx:  t_idx+self.tube_t+2*pt, 
                              z_idx:  z_idx+self.tube_z+2*pz, 
                            h_idx:h_idx+self.tube_h+2*ph, 
                            w_idx:w_idx+self.tube_w+2*pw
            ]
            
            if self.context == "after":
     #           print(f"Loading Y from \nT={t_idx} \nZ={z_idx} to {z_idx+self.tube_z} \nH={1+h_idx} to {1+h_idx+self.tube_h}\nW={1+w_idx} to {1+w_idx+self.tube_w}")
                y = self.y_data[:,
                              t_idx,
                              z_idx:  z_idx+self.tube_z, 
                            h_idx:h_idx+self.tube_h, 
                            w_idx:w_idx+self.tube_w
                ]
            elif self.context == "before":
      #          print(f"Loading Y from \nT={t_idx+self.tube_t-1} \nZ={z_idx} to {z_idx+self.tube_z} \nH={1+h_idx} to {1+h_idx+self.tube_h}\nW={1+w_idx} to {1+w_idx+self.tube_w}")
                y = self.y_data[:,
                              t_idx+self.tube_t-1,
                              z_idx:  z_idx+self.tube_z, 
                            h_idx:h_idx+self.tube_h, 
                            w_idx:w_idx+self.tube_w
                ]
            elif self.context == "middle":
       #         print(f"Loading Y from \nT={(2*t_idx+self.tube_t)//2} \nZ={z_idx} to {z_idx+self.tube_z} \nH={1+h_idx} to {1+h_idx+self.tube_h}\nW={1+w_idx} to {1+w_idx+self.tube_w}")
                y = self.y_data[:,
                            (2*t_idx+self.tube_t)//2,
                            z_idx:  z_idx+self.tube_z, 
                            h_idx:h_idx+self.tube_h, 
                            w_idx:w_idx+self.tube_w
                ]
            else: # self.context == "all":
          #      print(f"Loading Y from \nT={t_idx} to {t_idx+self.tube_t}\nZ={z_idx} to {z_idx+self.tube_z} \nH={1+h_idx} to {1+h_idx+self.tube_h}\nW={1+w_idx} to {1+w_idx+self.tube_w}")
                y = self.y_data[:,
                            t_idx:  t_idx+self.tube_t,
                            z_idx:  z_idx+self.tube_z, 
                            h_idx:h_idx+self.tube_h, 
                            w_idx:w_idx+self.tube_w
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
    test_norm_consts = {
        'ccn_001': (8394700.0, 12820105.0),
         'ccn_003': (27170852.0, 35127944.0),
         'ccn_006': (41711900.0, 54998000.0),
         'CHI': (0.7079330086708069, 0.05430074781179428),
         'CHI_CCN': (0.7251064777374268, 0.06840328127145767),
         'D_ALPHA': (2.1322059631347656, 0.4754127860069275),
         'D_GAMMA': (2.6306188106536865, 0.7740411758422852),
         'D_ALPHA_CCN': (1.4175677299499512, 0.09679309278726578),
         'D_GAMMA_CCN': (1.5800429582595825, 0.13608603179454803),
         'PM25': (3.3328231840989986e-10, 6.413445885478097e-10)
    }
    norm_consts = {
 'TOT_NUM_CONC': (122243312.0, 115225288.0),
 'TOT_MASS_CONC': (5.242313250164443e-10, 1.0374475722940701e-09),
 'pmc_SO4': (1.6618799392187356e-10, 1.7388031292586703e-10),
 'pmc_NO3': (5.9459021907459775e-12, 8.737478102149865e-11),
 'pmc_Cl': (6.25499262015141e-16, 6.634626794352685e-14),
 'pmc_NH4': (1.4210171234152469e-11, 4.621476812349812e-11),
 'pmc_ARO1': (3.87463092209675e-13, 1.7908442740896535e-12),
 'pmc_ARO2': (8.313421701261828e-15, 4.7049641065761635e-14),
 'pmc_ALK1': (2.7265067373966057e-12, 1.026210279947426e-11),
 'pmc_OLE1': (9.025159634391822e-13, 3.3690296646210482e-12),
 'pmc_API1': (1.5405897044185046e-11, 1.3261887248094961e-11),
 'pmc_API2': (1.0754304062606488e-25, 1.311163245377515e-25),
 'pmc_LIM1': (3.073970066592821e-25, 4.9992161671828435e-25),
 'pmc_LIM2': (1.999355268552461e-25, 3.1528140023602996e-25),
 'pmc_OC': (3.0542110507347786e-11, 5.430560168218079e-11),
 'pmc_BC': (1.4362456536576307e-11, 2.735458387581602e-11),
 'pmc_H2O': (2.735514037510711e-10, 8.615053670446571e-10),
 'PB': (42643.22265625, 30902.26953125),
 'P': (563.5490112304688, 488.75616455078125),
 'TEMPERATURE': (247.28065490722656, 29.46735191345215),
 'REL_HUMID': (0.26973065733909607, 0.21420344710350037),
 'ALT': (3.3093812465667725, 2.9574248790740967),
 'Z': (8953.72265625, 6435.02197265625),
 'h2so4': (0.0002380251098657027, 0.0009742539841681719),
 'hno3': (0.4889284670352936, 0.7151858806610107),
 'hcl': (4.993791299057193e-05, 0.0012396466918289661),
 'nh3': (0.3344135880470276, 2.24568247795105),
 'no': (0.059019263833761215, 0.2569088339805603),
 'no2': (0.19072215259075165, 0.5479520559310913),
 'no3': (0.0017606603214517236, 0.008451168425381184),
 'n2o5': (0.022702232003211975, 0.054008692502975464),
 'hono': (0.0007483834051527083, 0.004376617260277271),
 'hno4': (0.023550625890493393, 0.029744165018200874),
 'o3': (296.68341064453125, 466.1495666503906),
 'o1d': (3.2373572572685916e-11, 1.311246528024057e-10),
 'O3P': (4.0348149923374876e-05, 0.0001572183973621577),
 'oh': (0.00015794720093254, 0.0003126475203316659),
 'ho2': (0.0047643897123634815, 0.006401075981557369),
 'h2o2': (0.6846633553504944, 0.6319964528083801),
 'co': (78.20901489257812, 35.1939697265625),
 'so2': (0.012718993239104748, 0.06387235969305038),
 'ch4': (0.48198652267456055, 3.120267152786255),
 'c2h6': (0.0041136713698506355, 0.02158227190375328),
 'ch3o2': (0.00023352899006567895, 0.00048250085092149675),
 'ethp': (1.4215769624570385e-05, 7.255982927745208e-05),
 'hcho': (0.12044161558151245, 0.26350897550582886),
 'ch3oh': (0.06676128506660461, 0.18575364351272583),
 'ANOL': (0.012000726535916328, 0.045800067484378815),
 'ch3ooh': (0.004237775225192308, 0.009050405584275723),
 'ETHOOH': (0.00018857508257497102, 0.0005879862001165748),
 'ald2': (0.03090810589492321, 0.09336494654417038),
 'hcooh': (0.0016189146554097533, 0.004720032215118408),
 'RCOOH': (0.006248194258660078, 0.014132672920823097),
 'c2o3': (9.179675544146448e-05, 0.0001780444581527263),
 'pan': (0.13701358437538147, 0.1012202650308609),
 'aro1': (0.00010954443860100582, 0.0004766401252709329),
 'aro2': (4.135400376981124e-05, 0.00018421222921460867),
 'alk1': (0.00030477758264169097, 0.0010700338752940297),
 'ole1': (9.355915244668722e-05, 0.000306515721604228),
 'api1': (0.00016052905994001776, 0.00031206593848764896),
 'api2': (1.156230148550866e-16, 3.951551966623405e-17),
 'lim1': (1.169022595985672e-16, 4.293164324718005e-17),
 'lim2': (1.457666939863806e-16, 1.0584515767955857e-16),
 'par': (0.5630608797073364, 2.028073787689209),
 'AONE': (0.02213534153997898, 0.07549010962247849),
 'mgly': (0.003164333524182439, 0.01070734579116106),
 'eth': (0.04410184547305107, 0.14217419922351837),
 'OLET': (0.014296816661953926, 0.06587082892656326),
 'OLEI': (0.00196642754599452, 0.013290057890117168),
 'tol': (0.003926633857190609, 0.029529329389333725),
 'xyl': (0.002022645203396678, 0.0220699030905962),
 'cres': (2.6826164685189724e-05, 0.00021710287546738982),
 'to2': (1.0505578757147305e-05, 6.920799205545336e-05),
 'cro': (3.711737051048658e-08, 1.6965157101367367e-07),
 'open': (0.00021982230828143656, 0.0013298847479745746),
 'onit': (0.020106792449951172, 0.06660513579845428),
 'rooh': (0.014537632465362549, 0.042882487177848816),
 'ro2': (0.00010992176248691976, 0.0003908840590156615),
 'ano2': (1.01118048405624e-05, 6.336103251669556e-05),
 'nap': (7.543396350229159e-05, 0.0005537538090720773),
 'xo2': (0.00019939153571613133, 0.0006941959145478904),
 'xpar': (0.0002438412484480068, 0.0023324955254793167),
 'isop': (0.005643931683152914, 0.04789058864116669),
 'isoprd': (0.011829123832285404, 0.04617618769407272),
 'isopp': (0.000108394815470092, 0.0007807417423464358),
 'isopn': (0.00013015361037105322, 0.0008361282525584102),
 'isopo2': (3.188422488165088e-05, 0.000132248955196701),
 'api': (9.567690434213001e-17, 1.254190848933418e-17),
 'lim': (9.244174140943102e-17, 1.7572444607702983e-17),
 'dms': (9.999988257223015e-17, 1.5667591793215098e-19),
 'msa': (1.0249362036084071e-16, 5.248435310962747e-16),
 'dmso': (9.999832085523364e-17, 2.3487956344626317e-19),
 'dmso2': (1.0003128234828266e-16, 7.147608736452068e-19),
 'ch3so2h': (1.0003790641062798e-16, 8.981920001891737e-19),
 'ch3sch2oo': (1.000194900614701e-16, 3.9957306905231274e-18),
 'ch3so2': (9.993407870014033e-17, 4.014714071990913e-18),
 'ch3so3': (9.99398491120935e-17, 4.44397572856255e-18),
 'ch3so2oo': (9.998742192348265e-17, 4.0522459860784654e-17),
 'ch3so2ch2oo': (9.995455307466224e-17, 2.1401379290788468e-18),
 'SULFHOX': (1.653789922784199e-06, 4.9455411499366164e-05)}
    
if __name__ == "__main__":
    ds = AshbyDataset(None)