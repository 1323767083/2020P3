#from keras.layers.core import Dense
#from keras.layers import Input, Lambda
#from keras.layers import Concatenate
import tensorflow as tf
from nets_agent_base import *


def init_nets_agent_LHPP2V4(ilc, inc,iLNM_LV_SV_joint):
    global lc,nc
    lc,nc = ilc, inc
    global LNM_LV_SV_joint,cc
    LNM_LV_SV_joint = iLNM_LV_SV_joint
    cc = common_component()

class LHPP2V4:
    def __init__(self):
        self.DC = {
            "method_SV_state": "{0}_get_SV_state".format(lc.agent_method_sv),  # "RNN_get_SV_state",
            "method_LV_SV_joint_state": "{0}_get_LV_SV_joint_state".format(lc.agent_method_joint_lvsv),
            "method_ap_sv": "get_ap_av_{0}".format(lc.agent_method_apsv)
        }

    def build_predict_model(self, name):
        input_lv = Input(shape=nc.lv_shape, dtype='float32', name="{0}_input_lv".format(name))
        input_sv = Input(shape=nc.sv_shape, dtype='float32', name="{0}_input_sv".format(name))
        i_SV = SV_component()
        i_LV_SV = LV_SV_joint_component()

        sv_state = getattr(i_SV, self.DC["method_SV_state"])(input_sv, name)
        lv_sv_state = getattr(i_LV_SV, self.DC["method_LV_SV_joint_state"])([input_lv, sv_state], name + "for_ap")

        assert not lc.flag_sv_joint_state_stop_gradient, "{0} only support not lc.flag_sv_joint_state_stop_gradient".format(self.__class__.__name__)
        l_agent_output = getattr(self, self.DC["method_ap_sv"])(lv_sv_state, name)
        predict_model = Model(inputs=[input_lv, input_sv], outputs=l_agent_output, name=name)
        return predict_model

    def get_ap_av_HP(self, input_state, name):
        label = name + "Q"
        state = cc.construct_denses(nc.dense_l, input_state,            name=label + "_commonD")

        Pre_Q = cc.construct_denses(nc.dense_prob[:-1], state,          name=label + "_Pre_Q")
        Qs = Dense(nc.dense_prob[-1], activation='linear',             name=label + "Qs")(Pre_Q)
        return Qs



