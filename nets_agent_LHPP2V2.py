import tensorflow as tf
from nets_agent_base import *

def init_nets_agent_LHPP2V2(ilc, inc,iLNM_LV_SV_joint,iLNM_P, iLNM_V):
    global lc,nc
    lc,nc = ilc, inc
    global LNM_LV_SV_joint,LNM_P,LNM_V, cc
    LNM_LV_SV_joint = iLNM_LV_SV_joint
    LNM_P = iLNM_P
    LNM_V = iLNM_V
    cc=common_component()
class LHPP2V2:
    def __init__(self):
        self.DC = {
            "method_SV_state": "{0}_get_SV_state".format(lc.agent_method_sv),  # "RNN_get_SV_state",
            "method_LV_SV_joint_state": "{0}_get_LV_SV_joint_state".format(lc.agent_method_joint_lvsv),
            "method_ap_sv": "get_ap_av_{0}".format(lc.agent_method_apsv)
        }

    def build_predict_model(self, name):
        input_lv = Input(shape=nc.lv_shape, dtype='float32', name="{0}_input_lv".format(name))
        input_sv = Input(shape=nc.sv_shape, dtype='float32', name="{0}_input_sv".format(name))
        input_av = Input(shape=lc.specific_param.OS_AV_shape, dtype='float32', name="{0}_input_av".format(name))
        i_SV = SV_component()
        i_LV_SV = LV_SV_joint_component()

        sv_state = getattr(i_SV, self.DC["method_SV_state"])(input_sv, name)
        lv_sv_state = getattr(i_LV_SV, self.DC["method_LV_SV_joint_state"])([input_lv, sv_state], name + "for_ap")

        if not lc.flag_sv_joint_state_stop_gradient:
            input_method_ap_sv = [lv_sv_state, input_av]
        else:
            sv_state_stop_gradient = Lambda(lambda x: tf.stop_gradient(x), name="stop_gradiant_SV_state")(sv_state)
            lv_sv_state_stop_gradient = getattr(i_LV_SV, self.DC["method_LV_SV_joint_state"])(
                [input_lv, sv_state_stop_gradient], name + "for_sv")
            input_method_ap_sv = [lv_sv_state, lv_sv_state_stop_gradient, input_av]

        l_agent_output = getattr(self, self.DC["method_ap_sv"])(input_method_ap_sv, name)
        predict_model = Model(inputs=[input_lv, input_sv, input_av], outputs=l_agent_output, name=name)
        return predict_model

    #HP means status include holding period
    def get_ap_av_HP(self, inputs, name):
        js, input_av = inputs
        label = name + "_OS"

        input_state = Concatenate(axis=-1,                          name=label + "_input")([js, input_av])
        state = cc.construct_denses(nc.dense_l, input_state,        name=label + "_commonD")

        Pre_aSell = cc.construct_denses(nc.dense_prob[:-1], state,  name=label + "_Pre_aSell")
        ap = Dense(nc.dense_prob[-1], activation='softmax',      name=LNM_P)(Pre_aSell)

        if lc.flag_sv_stop_gradient:
            sv_state=Lambda(lambda x: tf.stop_gradient(x),          name=label + "_stop_gradiant_sv")(state)
        else:
            sv_state = Lambda(lambda x: x,                          name=label + "_not_stop_gradiant_sv")(state)
        Pre_sv = cc.construct_denses(nc.dense_advent[:-1], sv_state,name=label + "_Pre_sv")
        sv = Dense(nc.dense_advent[-1], activation='linear',        name=LNM_V)(Pre_sv)
        return ap, sv

    #SP means seperate ap sv 's lv_sv_jiong_state
    def get_ap_av_HP_SP(self, inputs, name):
        ap_js, sv_js, input_av = inputs

        aplabel = name + "_OS_ap"
        svlabel = name + "_OS_sv"
        ap_input_state = Concatenate(axis=-1, name=aplabel + "_input")([ap_js, input_av])
        sv_input_state = Concatenate(axis=-1, name=svlabel + "_input")([sv_js, input_av])


        ap_state = cc.construct_denses(nc.dense_l, ap_input_state,    name=aplabel + "_commonD")


        Pre_aSell = cc.construct_denses(nc.dense_prob[:-1], ap_state, name=aplabel + "_Pre_aSell")
        ap = Dense(nc.dense_prob[-1], activation='softmax',        name=LNM_P)(Pre_aSell)

        sv_state = cc.construct_denses(nc.dense_l, sv_input_state,    name=svlabel + "_commonD")
        Pre_sv = cc.construct_denses(nc.dense_advent[:-1], sv_state,  name=svlabel + "_Pre_sv")
        sv = Dense(nc.dense_advent[-1], activation='linear',          name=LNM_V)(Pre_sv)

        return ap,sv

    def get_ap_av_HP_DAV(self, inputs, name):#DAV means deep down av_input
        label = name + "_OS"


        input_state, input_av = inputs


        #input_state = Concatenate(axis=-1,                          name=label + "_input")([js, input_av])
        state = cc.construct_denses(nc.dense_l, input_state,        name=label + "_commonD")

        itermediate_state = Concatenate(axis=-1, name=label + "_input")([state, input_av])

        Pre_aSell = cc.construct_denses(nc.dense_prob[:-1], itermediate_state,  name=label + "_Pre_aSell")
        ap = Dense(nc.dense_prob[-1], activation='softmax',      name=LNM_P)(Pre_aSell)

        if lc.flag_sv_stop_gradient:
            sv_state=Lambda(lambda x: tf.stop_gradient(x),          name=label + "_stop_gradiant_sv")(itermediate_state)
        else:
            sv_state = Lambda(lambda x: x,                          name=label + "_not_stop_gradiant_sv")(itermediate_state)
        Pre_sv = cc.construct_denses(nc.dense_advent[:-1], sv_state,name=label + "_Pre_sv")
        sv = Dense(nc.dense_advent[-1], activation='linear',        name=LNM_V)(Pre_sv)
        return ap, sv