import numpy as np
from nets_agent_base import *
from action_comm import actionOBOS

def init_nets_agent_LHPP2V4(ilc, inc,iLNM_LV_SV_joint):
    global lc,nc
    lc,nc = ilc, inc
    global LNM_LV_SV_joint,cc
    LNM_LV_SV_joint = iLNM_LV_SV_joint
    cc = common_component()

class LHPP2V4_Agent:
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

        assert not lc.flag_sv_joint_state_stop_gradient, "{0} only support not lc.flag_sv_joint_state_stop_gradient".format(self.__class__.__name__)
        l_agent_output = getattr(self, self.DC["method_ap_sv"])(lv_sv_state, name)
        self.OB_model = keras.Model(inputs=[input_lv, input_sv], outputs=l_agent_output, name=name)
        return self.OB_model

    def load_weight(self, weight_fnwp):
        self.OB_model.load_weights(weight_fnwp)

    def get_ap_av_HP(self, input_state, name):
        label = name + "Q"
        state = cc.construct_denses(nc.dense_l, input_state,            name=label + "_commonD")

        Pre_Q = cc.construct_denses(nc.dense_prob[:-1], state,          name=label + "_Pre_Q")
        Qs = keras.layers.Dense(nc.dense_prob[-1], activation='linear',             name=label + "Qs")(Pre_Q)
        return Qs

    def predict(self,state):
        assert lc.P2_current_phase == "Train_Buy"
        lv, sv = state
        if not hasattr(self, "OB_model"):
            assert False, "should build or load model before"
        Qs = self.OB_model.predict({'P_input_lv': lv, 'P_input_sv': sv})
        return Qs


    def choose_action(self, state):
        assert lc.P2_current_phase == "Train_Buy"

        assert not lc.flag_multi_buy
        lv, sv, av = state
        buy_Qs = self.OB_model.predict({'P_input_lv': lv, 'P_input_sv': sv})
        if not hasattr(self, "OS_agent"):
            self.OS_agent = V2OS_4_OB_agent(lc.P2_sell_system_name, lc.P2_sell_model_tc)
            self.i_OS_action=actionOBOS("OS")
        sel_probs, sell_SVs = self.OS_agent.predict(state)

        l_a = []
        l_ap = []
        l_sv = []
        for buy_Q, sell_prob, av_item in zip(buy_Qs, sel_probs, av):
            assert len(buy_Q) == lc.train_action_num
            assert len(sell_prob) == 2
            flag_holding = self.check_holding_fun(av_item)
            if flag_holding:
                #action = np.random.choice([2, 3], p=sell_prob)
                action = self.i_OS_action.I_nets_choose_action(sell_prob)
                l_a.append(action)
                l_ap.append(np.zeros_like(sell_prob))
                l_sv.append(0)   #use l_state_value as Q value
            else:  # not have holding
                ### remove .num_action
                #buy_prob = np.ones(lc.specific_param.LHPP2V4_num_action,dtype=float) * \
                #           lc.specific_param.LHPP2V4_epsilon / lc.specific_param.LHPP2V4_num_action
                buy_prob = np.ones((lc.train_num_action,), dtype=float) * \
                               lc.specific_param.LHPP2V4_epsilon / lc.train_num_action

                action = np.argmax(buy_Q)
                buy_prob[action] += (1.0 - lc.specific_param.LHPP2V4_epsilon)
                #adjust_buy_prob = np.append(buy_prob, [0., 0.])     ### remove .num_action
                #gready_action = np.random.choice(np.arange(len(adjust_buy_prob)),p=adjust_buy_prob) ### remove .num_action
                #l_a.append(gready_action) ### remove .num_action
                #l_ap.append(adjust_buy_prob) ### remove .num_action

                gready_action = np.random.choice(np.arange(len(buy_prob)),p=buy_prob)
                l_a.append(gready_action)
                l_ap.append(buy_prob)
                l_sv.append(0)
        return l_a, l_ap, l_sv

