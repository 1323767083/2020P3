import numpy as np
from nets_agent_base import *
from action_comm import actionOBOS

def init_nets_agent_LHPP2V3(ilc, inc,iLNM_LV_SV_joint,iLNM_P, iLNM_V):
    global lc,nc
    lc,nc = ilc, inc
    global LNM_LV_SV_joint,LNM_P,LNM_V, cc
    LNM_LV_SV_joint = iLNM_LV_SV_joint
    LNM_P = iLNM_P
    LNM_V = iLNM_V
    cc = common_component()

class LHPP2V3_Agent:
    def __init__(self):
        keras.backend.set_learning_phase(0)  # add by john for error solved by
        self.DC = {
            "method_SV_state": "{0}_get_SV_state".format(lc.agent_method_sv),  # "RNN_get_SV_state",
            "method_LV_SV_joint_state": "{0}_get_LV_SV_joint_state".format(lc.agent_method_joint_lvsv),
            "method_ap_sv": "get_ap_av_{0}".format(lc.agent_method_apsv)
        }
        self.i_action = actionOBOS(lc.train_action_type)
        #self.check_holding_fun=LHPP2V2_check_holding
        if  hasattr(lc.specific_param,"CLN_AV"):
            i_cav=globals()[lc.specific_param.CLN_AV]()
            self.check_holding_fun = i_cav.check_holding_item
            self.get_OB_AV = i_cav.get_OB_av
        else:
            self.check_holding_fun = LHPP2V2_check_holding
            self.get_OB_AV = Train_Buy_get_AV_2


    def build_predict_model(self, name):
        input_lv = keras.Input(shape=nc.lv_shape, dtype='float32', name="{0}_input_lv".format(name))
        input_sv = keras.Input(shape=nc.sv_shape, dtype='float32', name="{0}_input_sv".format(name))
        input_av = keras.Input(shape=lc.specific_param.OB_AV_shape, dtype='float32', name="{0}_input_av".format(name))
        i_SV = SV_component()
        i_LV_SV = LV_SV_joint_component()

        sv_state = getattr(i_SV, self.DC["method_SV_state"])(input_sv, name)
        lv_sv_state = getattr(i_LV_SV, self.DC["method_LV_SV_joint_state"])([input_lv, sv_state], name + "for_ap")

        if not lc.flag_sv_joint_state_stop_gradient:
            input_method_ap_sv = [lv_sv_state,input_av]
        else:
            sv_state_stop_gradient = keras.layers.Lambda(lambda x: tf.stop_gradient(x), name="stop_gradiant_SV_state")(sv_state)
            lv_sv_state_stop_gradient = getattr(i_LV_SV, self.DC["method_LV_SV_joint_state"])(
                [input_lv, sv_state_stop_gradient], name + "for_sv")
            input_method_ap_sv = [lv_sv_state, lv_sv_state_stop_gradient, input_av]

        l_agent_output = getattr(self, self.DC["method_ap_sv"])(input_method_ap_sv, name)

        #predict_model = keras.Model(inputs=[input_lv, input_sv], outputs=l_agent_output, name=name)
        self.OB_model = keras.Model(inputs=[input_lv, input_sv,input_av], outputs=l_agent_output, name=name)
        return self.OB_model

    def load_weight(self, weight_fnwp):
        self.OB_model.load_weights(weight_fnwp)

    def get_ap_av_HP(self, inputs, name):
        label = name + "_OB"
        lv_sv_state,input_av = inputs

        input_state = keras.layers.Concatenate(axis=-1, name=label + "_input_state")([lv_sv_state, input_av])


        state = cc.construct_denses(nc.dense_l, input_state,            name=label + "_commonD")

        Pre_aBuy = cc.construct_denses(nc.dense_prob[:-1], state,       name=label + "_Pre_aBuy")
        ap = keras.layers.Dense(nc.dense_prob[-1], activation='softmax',name=label + LNM_P)(Pre_aBuy)

        if lc.flag_sv_stop_gradient:
            sv_state=keras.layers.Lambda(lambda x: tf.stop_gradient(x),              name=label + "_stop_gradiant_sv")(state)
        else:
            sv_state = keras.layers.Lambda(lambda x: x,                              name=label + "_not_stop_gradiant_sv")(state)
        Pre_sv = cc.construct_denses(nc.dense_advent[:-1], sv_state,    name=label + "_Pre_sv")
        sv = keras.layers.Dense(nc.dense_advent[-1], activation='linear', name=label + LNM_V)(Pre_sv)
        return ap, sv



    #SP means seperate ap sv 's lv_sv_jiong_state
    def get_ap_av_HP_SP(self, inputs, name):

        ap_input, sv_input,input_av = inputs
        aplabel = name + "_OB_ap"
        svlabel = name + "_OB_sv"

        ap_input_state = keras.layers.Concatenate(axis=-1, name=aplabel + "_input_state")([ap_input, input_av])


        ap_state = cc.construct_denses(nc.dense_l, ap_input_state,      name=aplabel + "_commonD")
        Pre_aBuy = cc.construct_denses(nc.dense_prob[:-1], ap_state,    name=aplabel + "_Pre_aBuy")
        ap = keras.layers.Dense(nc.dense_prob[-1], activation='softmax',name=aplabel + LNM_P)(Pre_aBuy)

        sv_input_state = keras.layers.Concatenate(axis=-1, name=svlabel + "_input_state")([sv_input, input_av])
        sv_state_com = cc.construct_denses(nc.dense_l, sv_input_state,  name=svlabel + "_commonD")
        Pre_sv = cc.construct_denses(nc.dense_advent[:-1], sv_state_com,name=svlabel + "_Pre_sv")
        sv = keras.layers.Dense(nc.dense_advent[-1], activation='linear',name=svlabel + LNM_V)(Pre_sv)
        return ap, sv

    def predict(self,state):
        assert lc.P2_current_phase == "Train_Buy"
        lv, sv,av = state
        if not hasattr(self, "OB_model"):
            assert False, "should build or load model before"
        p, v = self.OB_model.predict({'P_input_lv': lv, 'P_input_sv': sv,"P_input_av":self.get_OB_AV(av)})
        return p, v

    def choose_action(self,state):
        assert lc.P2_current_phase == "Train_Buy"
        assert not lc.flag_multi_buy
        lv, sv, av = state
        buy_probs, buy_SVs = self.predict([lv, sv,av])
        if not hasattr(self, "OS_agent"):
            self.OS_agent = V2OS_4_OB_agent(lc.P2_sell_system_name, lc.P2_sell_model_tc)
            self.i_OS_action=actionOBOS("OS")
        sel_probs, sell_SVs = self.OS_agent.predict(state)
        l_a = []
        l_ap = []
        l_sv = []
        for buy_prob, sell_prob, buy_sv, sell_sv, av_item in zip(buy_probs,sel_probs,buy_SVs,sell_SVs,av):
            assert len(buy_prob)==2
            assert len(sell_prob) == 2
            flag_holding=self.check_holding_fun(av_item)
            if flag_holding:
                #action = np.random.choice([2, 3], p=sell_prob)
                action = self.i_OS_action.I_nets_choose_action(sell_prob)
                l_a.append(action)
                l_ap.append(np.zeros_like(sell_prob))  # this is add zero and this record will be removed by TD_buffer before send to server for train
                l_sv.append(sell_sv[0])
            else: # not have holding
                #action = np.random.choice([0, 1], p=buy_prob)
                action = self.i_action.I_nets_choose_action(buy_prob)
                l_a.append(action)
                l_ap.append(buy_prob)
                l_sv.append(buy_sv[0])
        return l_a, l_ap,l_sv

LHPP2V32_Agent=LHPP2V3_Agent
LHPP2V33_Agent=LHPP2V3_Agent

