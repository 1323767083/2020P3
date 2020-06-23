from nets_agent_LHPP2V3 import *
from nets_agent_LHPP2V2 import *
#from nets_agent_LHPP2V4 import *
#from nets_agent_LHPP2V5 import *
#from nets_agent_LHPP2V6 import *
#from nets_agent_LHPP2V61 import *
#from nets_agent_LHPP2V7 import *
from nets_agent_LHPP2V8 import *
class net_conf:
    """
    @DynamicAttrs
    """

def init_agent_config(input_lc):
    global nc,lc
    lc=input_lc
    nc=net_conf()


    N_item_list=["lv_shape","sv_shape"]

    CNN_sv_list=["flag_s_level","s_kernel_l","s_filter_l","s_maxpool_l"]
    RNN_sv_list=["ms_param_TimeDistributed"]

    CNN_lvsv_list=["flag_l_level","l_kernel_l","l_filter_l","l_maxpool_l"]
    RNN_lvsv_list=["ms_param_CuDNNLSTM"]

    D_list=["dense_l","dense_prob","dense_advent"]

    nc_item_list=[]
    nc_item_list += N_item_list
    nc_item_list += D_list
    if lc.agent_method_sv=="RNN":
        nc_item_list += RNN_sv_list
    elif lc.agent_method_sv=="CNN":
        nc_item_list += CNN_sv_list
    else:
        assert lc.agent_method_sv=="RCN"
        nc_item_list += RNN_sv_list
        nc_item_list += CNN_sv_list

    if lc.agent_method_joint_lvsv=="RNN":
        nc_item_list += RNN_lvsv_list
    elif lc.agent_method_joint_lvsv=="CNN":
        nc_item_list += CNN_lvsv_list
    else:
        assert lc.agent_method_joint_lvsv=="RCN"
        nc_item_list += RNN_lvsv_list
        nc_item_list += CNN_lvsv_list


    for item_title in nc_item_list:
        assert item_title in list(lc.net_config.keys())
        setattr(nc, item_title, lc.net_config[item_title])

    nc.lv_shape = tuple(nc.lv_shape)
    nc.sv_shape = tuple(nc.sv_shape)

    global LNM_LV_SV_joint,LNM_P,LNM_V
    LNM_LV_SV_joint = "State_LSV"
    LNM_P = "Act_prob"
    LNM_V = "State_value"
    init_nets_agent_base(lc, nc, LNM_LV_SV_joint, LNM_P, LNM_V)
    init_nets_agent_LHPP2V2(lc, nc, LNM_LV_SV_joint, LNM_P, LNM_V)
    init_nets_agent_LHPP2V3(lc, nc, LNM_LV_SV_joint, LNM_P, LNM_V)
    init_nets_agent_LHPP2V8(lc, nc, LNM_LV_SV_joint, LNM_P, LNM_V)