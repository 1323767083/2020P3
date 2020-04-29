from nets_agent_base import *
import tensorflow as tf
import numpy as np
from action_comm import actionOBOS

def init_nets_agent_LHPP2V61(ilc, inc,iLNM_LV_SV_joint,iLNM_P, iLNM_V):
    global lc,nc
    lc,nc = ilc, inc
    global LNM_LV_SV_joint,LNM_P,LNM_V, cc
    LNM_LV_SV_joint = iLNM_LV_SV_joint
    LNM_P = iLNM_P
    LNM_V = iLNM_V
    cc = common_component()
#LHPP2V5 is same as LHPP2V3
class LHPP2V61_Agent:
    def __init__(self):
        keras.backend.set_learning_phase(0)  # add by john for error solved by
        self.DC = {
            "method_SV_state": "{0}_get_SV_state".format(lc.agent_method_sv),  # "RNN_get_SV_state",
            "method_LV_SV_joint_state": "{0}_get_LV_SV_joint_state".format(lc.agent_method_joint_lvsv),
            "method_ap_sv": "get_ap_av_{0}".format(lc.agent_method_apsv)
        }
        self.i_action = actionOBOS(lc.train_action_type)
        self.check_holding_fun=LHPP2V2_check_holding

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

        self.OB_model = keras.Model(inputs=[input_lv, input_sv], outputs=l_agent_output, name=name)
        return self.OB_model

    def load_weight(self, weight_fnwp):
        self.OB_model.load_weights(weight_fnwp)

    #HP means status include holding period
    def get_ap_av_HP(self, input_state, name):
        label = name + "_OB"
        state = cc.construct_denses(nc.dense_l, input_state,            name=label + "_commonD")

        Pre_apTNT = cc.construct_denses(nc.dense_prob[:-1], state,       name=label + "_Pre_apTNT")
        apTNT = keras.layers.Dense(nc.dense_prob[-1], activation='softmax',  name=label + "_apTNT")(Pre_apTNT)

        Pre_apBNB = cc.construct_denses(nc.dense_prob[:-1], state,       name=label + "_Pre_apBNB")
        apBNB = keras.layers.Dense(nc.dense_prob[-1], activation='softmax',  name=label + "_apBNB")(Pre_apBNB)

        ap = keras.layers.Concatenate(axis=-1, name=name + LNM_P)([apTNT, apBNB])

        if lc.flag_sv_stop_gradient:
            sv_state=keras.layers.Lambda(lambda x: tf.stop_gradient(x),              name=label + "_stop_gradiant_sv")(state)
        else:
            sv_state = keras.layers.Lambda(lambda x: x,                              name=label + "_not_stop_gradiant_sv")(state)

        Pre_sv_TNT = cc.construct_denses(nc.dense_advent[:-1], sv_state,    name=label + "_Pre_sv_TNT")
        sv_TNT = keras.layers.Dense(nc.dense_advent[-1], activation='linear',            name=label + "_sv_TNT")(Pre_sv_TNT)

        Pre_sv_BNB = cc.construct_denses(nc.dense_advent[:-1], sv_state,    name=label + "_Pre_sv_BNB")
        sv_BNB = keras.layers.Dense(nc.dense_advent[-1], activation='linear',            name=label + "_sv_BNB")(Pre_sv_BNB)

        sv = keras.layers.Concatenate(axis=-1, name=name + LNM_V)([sv_TNT, sv_BNB])
        return ap, sv

    #SP means seperate ap sv 's lv_sv_jiong_state
    def get_ap_av_HP_SP(self, inputs, name):
        ap_input_state, sv_input_state = inputs
        aplabel = name + "_OB_ap"
        svlabel = name + "_OB_sv"

        ap_state = cc.construct_denses(nc.dense_l, ap_input_state, name=aplabel + "_commonD")

        Pre_apTNT = cc.construct_denses(nc.dense_prob[:-1], ap_state,       name=aplabel + "_Pre_apTNT")
        apTNT = keras.layers.Dense(nc.dense_prob[-1], activation='softmax',  name=aplabel + "_apTNT")(Pre_apTNT)

        Pre_apBNB = cc.construct_denses(nc.dense_prob[:-1], ap_state,       name=aplabel + "_Pre_apBNB")
        apBNB = keras.layers.Dense(nc.dense_prob[-1], activation='softmax',  name=aplabel + "_apBNB")(Pre_apBNB)

        ap = keras.layers.Concatenate(axis=-1, name=name + LNM_P)([apTNT, apBNB])

        sv_state_com = cc.construct_denses(nc.dense_l, sv_input_state, name=svlabel + "_commonD")

        Pre_sv_TNT = cc.construct_denses(nc.dense_advent[:-1], sv_state_com,    name=svlabel + "_Pre_sv_TNT")
        sv_TNT = keras.layers.Dense(nc.dense_advent[-1], activation='linear',            name=svlabel + "_sv_TNT")(Pre_sv_TNT)

        Pre_sv_BNB = cc.construct_denses(nc.dense_advent[:-1], sv_state_com,    name=svlabel + "_Pre_sv_BNB")
        sv_BNB = keras.layers.Dense(nc.dense_advent[-1], activation='linear',            name=svlabel + "_sv_BNB")(Pre_sv_BNB)

        sv = keras.layers.Concatenate(axis=-1, name=name + LNM_V)([sv_TNT, sv_BNB])

        return ap, sv

    def predict(self,state):
        assert lc.P2_current_phase == "Train_Buy"
        lv, sv = state
        if not hasattr(self, "OB_model"):
            assert False, "should build or load model before"
        p, v = self.OB_model.predict({'P_input_lv': lv, 'P_input_sv': sv})
        return p, v

    def choose_action(self,state):
        assert lc.P2_current_phase == "Train_Buy"
        assert not lc.flag_multi_buy
        lv, sv, av = state
        buy_probs, buy_SVs = self.predict([lv, sv])
        if not hasattr(self, "OS_agent"):
            self.OS_agent = V2OS_4_OB_agent(lc.P2_sell_system_name, lc.P2_sell_model_tc)
            self.i_OS_action=actionOBOS("OS")
        sel_probs, sell_SVs = self.OS_agent.predict(state)
        l_a = []
        l_ap = []
        l_sv = []
        for buy_prob, sell_prob, buy_sv, sell_sv, av_item in zip(buy_probs,sel_probs,buy_SVs,sell_SVs,av):
            assert len(buy_prob)==4
            assert len(sell_prob) == 2
            flag_holding=self.check_holding_fun(av_item)
            if flag_holding:
                #action = np.random.choice([2, 3], p=sell_prob)
                action = self.i_OS_action.I_nets_choose_action(sell_prob)
                l_a.append(action)
                #l_ap.append(np.zeros(len(sell_prob)+1))  # this is add zero and this record will be removed by TD_buffer before send to server for train
                l_ap.append(np.zeros(len(buy_prob),dtype=np.float32))  # this is add zero and this record will be removed by TD_buffer before send to server for train
                l_sv.append(sell_sv[0])
            else: # not have holding
                #action = np.random.choice([0, 1], p=buy_prob)
                action = self.i_action.I_nets_choose_action(buy_prob)
                l_a.append(action)
                l_ap.append(buy_prob)
                l_sv.append(buy_sv[0])
        return l_a, l_ap,l_sv

LHPP2V62_Agent=LHPP2V61_Agent