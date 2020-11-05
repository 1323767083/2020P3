from nets_agent_base import *
import tensorflow as tf
from action_comm import actionOBOS
import numpy as np
class LHPP2V8_Agent:
    def __init__(self,lc):
        self.lc=lc
        self.nc = get_agent_nc(lc)
        self.cc= common_component()

        keras.backend.set_learning_phase(0)  # add by john for error solved by
        self.DC = {
            "method_SV_state": "{0}_get_SV_state".format(lc.agent_method_sv),  # "RNN_get_SV_state",
            "method_LV_SV_joint_state": "{0}_get_LV_SV_joint_state".format(lc.agent_method_joint_lvsv),
            "method_ap_sv": "get_ap_av_{0}".format(lc.agent_method_apsv)
        }
        self.i_action = actionOBOS(lc.train_action_type)
        self.i_cav = globals()[lc.CLN_AV_Handler](lc)


    def build_predict_model(self, name):
        input_lv = keras.Input(shape=self.nc.lv_shape, dtype='float32', name="{0}_input_lv".format(name))
        input_sv = keras.Input(shape=self.nc.sv_shape, dtype='float32', name="{0}_input_sv".format(name))
        #input_av = keras.Input(shape=(1+lc.specific_param.LNT+1+ lc.specific_param.LNB+1,), dtype='float32', name="{0}_input_av".format(name))
        input_av = keras.Input(shape=self.lc.OB_AV_shape, dtype='float32',name="{0}_input_av".format(name))
        i_SV = SV_component(self.nc)
        i_LV_SV = LV_SV_joint_component(self.nc, self.cc)

        sv_state = getattr(i_SV, self.DC["method_SV_state"])(input_sv, name)
        lv_sv_state = getattr(i_LV_SV, self.DC["method_LV_SV_joint_state"])([input_lv, sv_state], name + "for_ap")

        if not self.lc.flag_sv_joint_state_stop_gradient:
            input_method_ap_sv = [lv_sv_state,input_av]
        else:
            sv_state_stop_gradient = keras.layers.Lambda(lambda x: tf.stop_gradient(x), name="stop_gradiant_SV_state")(sv_state)
            lv_sv_state_stop_gradient = getattr(i_LV_SV, self.DC["method_LV_SV_joint_state"])(
                [input_lv, sv_state_stop_gradient], name + "for_sv")
            input_method_ap_sv = [lv_sv_state, lv_sv_state_stop_gradient, input_av]

        l_agent_output = getattr(self, self.DC["method_ap_sv"])(input_method_ap_sv, name)

        self.OB_model = keras.Model(inputs=[input_lv, input_sv,input_av], outputs=l_agent_output, name=name)
        return self.OB_model

    def load_weight(self, weight_fnwp):
        self.OB_model.load_weights(weight_fnwp)

    #HP means status include holding period
    def get_ap_av_HP(self, inputs, name):
        label = name + "_OB"
        lv_sv_state,input_av = inputs

        input_state = keras.layers.Concatenate(axis=-1, name=label + "_input_state")([lv_sv_state, input_av])


        state = self.cc.construct_denses(self.nc.dense_l, input_state,            name=label + "_commonD")

        Pre_aBuy = self.cc.construct_denses(self.nc.dense_prob[:-1], state,       name=label + "_Pre_aBuy")
        ap = keras.layers.Dense(self.nc.dense_prob[-1], activation='softmax',name=label + self.nc.LNM_P)(Pre_aBuy)

        if self.lc.flag_sv_stop_gradient:
            sv_state=keras.layers.Lambda(lambda x: tf.stop_gradient(x),              name=label + "_stop_gradiant_sv")(state)
        else:
            sv_state = keras.layers.Lambda(lambda x: x,                              name=label + "_not_stop_gradiant_sv")(state)
        Pre_sv = self.cc.construct_denses(self.nc.dense_advent[:-1], sv_state,    name=label + "_Pre_sv")
        sv = keras.layers.Dense(self.nc.dense_advent[-1], activation='linear', name=label + self.nc.LNM_V)(Pre_sv)
        return ap, sv



    #SP means seperate ap sv 's lv_sv_jiong_state
    def get_ap_av_HP_SP(self, inputs, name):

        ap_input, sv_input,input_av = inputs
        aplabel = name + "_OB_ap"
        svlabel = name + "_OB_sv"

        ap_input_state = keras.layers.Concatenate(axis=-1, name=aplabel + "_input_state")([ap_input, input_av])


        ap_state = self.cc.construct_denses(self.nc.dense_l, ap_input_state,      name=aplabel + "_commonD")
        Pre_aBuy = self.cc.construct_denses(self.nc.dense_prob[:-1], ap_state,    name=aplabel + "_Pre_aBuy")
        ap = keras.layers.Dense(self.nc.dense_prob[-1], activation='softmax',name=aplabel + self.nc.LNM_P)(Pre_aBuy)

        sv_input_state = keras.layers.Concatenate(axis=-1, name=svlabel + "_input_state")([sv_input, input_av])
        sv_state_com = self.cc.construct_denses(self.nc.dense_l, sv_input_state,  name=svlabel + "_commonD")
        Pre_sv = self.cc.construct_denses(self.nc.dense_advent[:-1], sv_state_com,name=svlabel + "_Pre_sv")
        sv = keras.layers.Dense(self.nc.dense_advent[-1], activation='linear',name=svlabel + self.nc.LNM_V)(Pre_sv)
        return ap, sv


    def predict(self,state):
        assert self.lc.P2_current_phase == "Train_Buy"
        lv, sv,av = state
        if not hasattr(self, "OB_model"):
            assert False, "should build or load model before"
        p, v = self.OB_model.predict({'P_input_lv': lv, 'P_input_sv': sv,"P_input_av":self.i_cav.get_OB_AV(av)})
        return p, v

    def choose_action(self,state,calledby="Eval"):
        assert self.lc.P2_current_phase == "Train_Buy"
        lv, sv, av = state
        buy_probs, buy_SVs = self.predict([lv, sv, av])
        if not hasattr(self, "OS_agent"):
            self.OS_agent = V2OS_4_OB_agent(self.lc,self.lc.P2_sell_system_name, self.lc.P2_sell_model_tc)
            self.i_OS_action=actionOBOS("OS")
        sel_probs, sell_SVs = self.OS_agent.predict(state)
        l_a = []
        l_ap = []
        l_sv = []
        for buy_prob, sell_prob, buy_sv, sell_sv, av_item in zip(buy_probs,sel_probs,buy_SVs,sell_SVs,av):
            assert len(buy_prob)==3
            assert len(sell_prob) == 2
            flag_holding=self.i_cav.Is_Holding_Item(av_item)
            if flag_holding:
                #action = np.random.choice([2, 3], p=sell_prob)
                action = self.i_OS_action.I_nets_choose_action(sell_prob)
                l_a.append(action)
                l_ap.append(np.zeros(len(sell_prob)+1,dtype=np.float32))  # this is add zero and this record will be removed by TD_buffer before send to server for train
                l_sv.append(sell_sv[0])
            else: # not have holding
                #action = np.random.choice([0, 1], p=buy_prob)
                #action = self.i_action.I_nets_choose_action_V8([buy_prob, av_item,lc.specific_param.LNB])
                action = self.i_action.I_nets_choose_action([buy_prob, av_item, self.lc.specific_param.LNB])
                l_a.append(action)
                l_ap.append(buy_prob)
                l_sv.append(buy_sv[0])
        return l_a, l_ap,l_sv
