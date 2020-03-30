from nets_agent_base import *
import tensorflow as tf

def init_nets_agent_LHPP2V5(ilc, inc,iLNM_LV_SV_joint,iLNM_P, iLNM_V):
    global lc,nc
    lc,nc = ilc, inc
    global LNM_LV_SV_joint,LNM_P,LNM_V, cc
    LNM_LV_SV_joint = iLNM_LV_SV_joint
    LNM_P = iLNM_P
    LNM_V = iLNM_V
    cc = common_component()
#LHPP2V5 is same as LHPP2V3
class LHPP2V5:
    def __init__(self):
        self.DC = {
            "method_SV_state": "{0}_get_SV_state".format(lc.agent_method_sv),  # "RNN_get_SV_state",
            "method_LV_SV_joint_state": "{0}_get_LV_SV_joint_state".format(lc.agent_method_joint_lvsv),
            "method_ap_sv": "get_ap_av_{0}".format(lc.agent_method_apsv)
        }

    def build_predict_model(self, name):
        input_lv = keras.Input(shape=nc.lv_shape, dtype='float32', name="{0}_input_lv".format(name))
        input_sv = keras.Input(shape=nc.sv_shape, dtype='float32', name="{0}_input_sv".format(name))
        i_SV = SV_component()
        i_LV_SV = LV_SV_joint_component()

        sv_state = getattr(i_SV, self.DC["method_SV_state"])(input_sv, name)
        lv_sv_state = getattr(i_LV_SV, self.DC["method_LV_SV_joint_state"])([input_lv, sv_state], name + "for_ap")

        if not lc.flag_sv_joint_state_stop_gradient:
            input_method_ap_sv = lv_sv_state
        else:
            sv_state_stop_gradient = keras.layers.Lambda(lambda x: tf.stop_gradient(x), name="stop_gradiant_SV_state")(sv_state)
            lv_sv_state_stop_gradient = getattr(i_LV_SV, self.DC["method_LV_SV_joint_state"])(
                [input_lv, sv_state_stop_gradient], name + "for_sv")
            input_method_ap_sv = [lv_sv_state, lv_sv_state_stop_gradient]

        l_agent_output = getattr(self, self.DC["method_ap_sv"])(input_method_ap_sv, name)

        predict_model = keras.Model(inputs=[input_lv, input_sv], outputs=l_agent_output, name=name)
        return predict_model

    #HP means status include holding period
    def get_ap_av_HP(self, input_state, name):
        label = name + "_OB"
        state = cc.construct_denses(nc.dense_l, input_state,            name=label + "_commonD")

        Pre_aBuy = cc.construct_denses(nc.dense_prob[:-1], state,       name=label + "_Pre_aBuy")
        ap = keras.layers.Dense(nc.dense_prob[-1], activation='softmax',             name=LNM_P)(Pre_aBuy)

        if lc.flag_sv_stop_gradient:
            sv_state=keras.layers.Lambda(lambda x: tf.stop_gradient(x),              name=label + "_stop_gradiant_sv")(state)
        else:
            sv_state = keras.layers.Lambda(lambda x: x,                              name=label + "_not_stop_gradiant_sv")(state)
        Pre_sv = cc.construct_denses(nc.dense_advent[:-1], sv_state,    name=label + "_Pre_sv")
        sv = keras.layers.Dense(nc.dense_advent[-1], activation='linear',            name=LNM_V)(Pre_sv)
        return ap, sv



    #SP means seperate ap sv 's lv_sv_jiong_state
    def get_ap_av_HP_SP(self, inputs, name):
        ap_input_state, sv_input_state = inputs
        aplabel = name + "_OB_ap"
        svlabel = name + "_OB_sv"

        ap_state = cc.construct_denses(nc.dense_l, ap_input_state,      name=aplabel + "_commonD")
        Pre_aBuy = cc.construct_denses(nc.dense_prob[:-1], ap_state,    name=aplabel + "_Pre_aBuy")
        ap = Dense(nc.dense_prob[-1], activation='softmax',             name=LNM_P)(Pre_aBuy)

        sv_state_com = cc.construct_denses(nc.dense_l, sv_input_state,  name=svlabel + "_commonD")
        Pre_sv = cc.construct_denses(nc.dense_advent[:-1], sv_state_com,name=svlabel + "_Pre_sv")
        sv = Dense(nc.dense_advent[-1], activation='linear',            name=LNM_V)(Pre_sv)
        return ap, sv


