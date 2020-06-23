from nets_trainer_LHPP2V2 import *
from nets_trainer_LHPP2V3 import *
#from nets_trainer_LHPP2V32 import *
#from nets_trainer_LHPP2V33 import *
#from nets_trainer_LHPP2V4 import *
#from nets_trainer_LHPP2V5 import *
#from nets_trainer_LHPP2V6 import *
#from nets_trainer_LHPP2V61 import *
#from nets_trainer_LHPP2V7 import *
from nets_trainer_LHPP2V8 import *
class nets_conf:
    def __init__(self):
        # net config
        self.lv_shape = (20, 17)
        self.sv_shape = (20, 25, 2)

def init_trainer_config(input_lc):
    global lc, nc
    lc=input_lc
    nc=nets_conf()
    for key in list(nc.__dict__.keys()):
        nc.__dict__[key] = lc.net_config[key]
    nc.lv_shape = tuple(nc.lv_shape)
    nc.sv_shape = tuple(nc.sv_shape)
    init_nets_trainer_LHPP2V2(lc,nc)
    init_nets_trainer_LHPP2V3(lc, nc)
    init_nets_trainer_LHPP2V8(lc, nc)

