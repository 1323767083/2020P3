from nets_agent_base import *
from action_comm import actionOBOS

def init_nets_agent_LHPP2V2(ilc, inc,iLNM_LV_SV_joint,iLNM_P, iLNM_V):
    global lc,nc
    lc,nc = ilc, inc
    global LNM_LV_SV_joint,LNM_P,LNM_V, cc
    LNM_LV_SV_joint = iLNM_LV_SV_joint
    LNM_P = iLNM_P
    LNM_V = iLNM_V
    cc=common_component()
class LHPP2V2_Agent:
    def __init__(self):
        keras.backend.set_learning_phase(0)  # add by john for error solved by
        self.DC = {
            "method_SV_state": "{0}_get_SV_state".format(lc.agent_method_sv),  # "RNN_get_SV_state",
            "method_LV_SV_joint_state": "{0}_get_LV_SV_joint_state".format(lc.agent_method_joint_lvsv),
            "method_ap_sv": "get_ap_av_{0}".format(lc.agent_method_apsv)
        }
        self.i_action = actionOBOS(lc.train_action_type)
        i_cav = globals()[lc.CLN_AV_state]()
        self.check_holding_fun = i_cav.check_holding_item
        self.get_OS_av = i_cav.get_OS_av


    def build_predict_model(self, name):
        input_lv = keras.Input(shape=nc.lv_shape, dtype='float32', name="{0}_input_lv".format(name))
        input_sv = keras.Input(shape=nc.sv_shape, dtype='float32', name="{0}_input_sv".format(name))
        input_av = keras.Input(shape=lc.OS_AV_shape, dtype='float32', name="{0}_input_av".format(name))
        i_SV = SV_component()
        i_LV_SV = LV_SV_joint_component()

        sv_state = getattr(i_SV, self.DC["method_SV_state"])(input_sv, name)
        lv_sv_state = getattr(i_LV_SV, self.DC["method_LV_SV_joint_state"])([input_lv, sv_state], name + "for_ap")

        if not lc.flag_sv_joint_state_stop_gradient:
            input_method_ap_sv = [lv_sv_state, input_av]
        else:
            sv_state_stop_gradient = keras.layers.Lambda(lambda x: tf.stop_gradient(x), name="stop_gradiant_SV_state")(sv_state)
            lv_sv_state_stop_gradient = getattr(i_LV_SV, self.DC["method_LV_SV_joint_state"])(
                [input_lv, sv_state_stop_gradient], name + "for_sv")
            input_method_ap_sv = [lv_sv_state, lv_sv_state_stop_gradient, input_av]

        l_agent_output = getattr(self, self.DC["method_ap_sv"])(input_method_ap_sv, name)
        #predict_model = keras.Model(inputs=[input_lv, input_sv, input_av], outputs=l_agent_output, name=name)
        #return predict_model
        self.OS_model = keras.Model(inputs=[input_lv, input_sv, input_av], outputs=l_agent_output, name=name)
        return self.OS_model

    def load_weight(self, weight_fnwp):
        self.OS_model.load_weights(weight_fnwp)

    #HP means status include holding period
    def get_ap_av_HP(self, inputs, name):
        js, input_av = inputs
        label = name + "_OS"

        input_state = keras.layers.Concatenate(axis=-1,                          name=label + "_input")([js, input_av])
        state = cc.construct_denses(nc.dense_l, input_state,        name=label + "_commonD")

        Pre_aSell = cc.construct_denses(nc.dense_prob[:-1], state,  name=label + "_Pre_aSell")
        ap = keras.layers.Dense(nc.dense_prob[-1], activation='softmax',      name=LNM_P)(Pre_aSell)

        if lc.flag_sv_stop_gradient:
            sv_state=keras.layers.Lambda(lambda x: tf.stop_gradient(x),          name=label + "_stop_gradiant_sv")(state)
        else:
            sv_state = keras.layers.Lambda(lambda x: x,                          name=label + "_not_stop_gradiant_sv")(state)
        Pre_sv = cc.construct_denses(nc.dense_advent[:-1], sv_state,name=label + "_Pre_sv")
        sv = keras.layers.Dense(nc.dense_advent[-1], activation='linear',        name=LNM_V)(Pre_sv)
        return ap, sv

    #SP means seperate ap sv 's lv_sv_jiong_state
    def get_ap_av_HP_SP(self, inputs, name):
        ap_js, sv_js, input_av = inputs

        aplabel = name + "_OS_ap"
        svlabel = name + "_OS_sv"
        ap_input_state = keras.layers.Concatenate(axis=-1, name=aplabel + "_input")([ap_js, input_av])
        sv_input_state = keras.layers.Concatenate(axis=-1, name=svlabel + "_input")([sv_js, input_av])


        ap_state = cc.construct_denses(nc.dense_l, ap_input_state,    name=aplabel + "_commonD")


        Pre_aSell = cc.construct_denses(nc.dense_prob[:-1], ap_state, name=aplabel + "_Pre_aSell")
        ap = keras.layers.Dense(nc.dense_prob[-1], activation='softmax',        name=LNM_P)(Pre_aSell)

        sv_state = cc.construct_denses(nc.dense_l, sv_input_state,    name=svlabel + "_commonD")
        Pre_sv = cc.construct_denses(nc.dense_advent[:-1], sv_state,  name=svlabel + "_Pre_sv")
        sv = keras.layers.Dense(nc.dense_advent[-1], activation='linear',          name=LNM_V)(Pre_sv)

        return ap,sv

    def get_ap_av_HP_DAV(self, inputs, name):#DAV means deep down av_input
        label = name + "_OS"


        input_state, input_av = inputs


        #input_state = Concatenate(axis=-1,                          name=label + "_input")([js, input_av])
        state = cc.construct_denses(nc.dense_l, input_state,        name=label + "_commonD")

        itermediate_state = keras.layers.Concatenate(axis=-1, name=label + "_input")([state, input_av])

        Pre_aSell = cc.construct_denses(nc.dense_prob[:-1], itermediate_state,  name=label + "_Pre_aSell")
        ap = keras.layers.Dense(nc.dense_prob[-1], activation='softmax',      name=LNM_P)(Pre_aSell)

        if lc.flag_sv_stop_gradient:
            sv_state = keras.layers.Lambda(lambda x: tf.stop_gradient(x),          name=label + "_stop_gradiant_sv")(itermediate_state)
        else:
            sv_state = keras.layers.Lambda(lambda x: x,                          name=label + "_not_stop_gradiant_sv")(itermediate_state)
        Pre_sv = cc.construct_denses(nc.dense_advent[:-1], sv_state,name=label + "_Pre_sv")
        sv = keras.layers.Dense(nc.dense_advent[-1], activation='linear',        name=LNM_V)(Pre_sv)
        return ap, sv

    def predict(self, state):
        assert lc.P2_current_phase=="Train_Sell"
        lv, sv, av = state
        p, v = self.OS_model.predict({'P_input_lv': lv, 'P_input_sv': sv, 'P_input_av': self.get_OS_av(av)})
        return p,v

    def choose_action(self, state, calledby="Eval"):
        assert lc.P2_current_phase == "Train_Sell"
        assert not lc.flag_multi_buy
        lv, sv, av = state
        actions_probs, SVs = self.predict(state)
        l_a = []
        l_ap = []
        l_sv = []
        for sell_prob, SV, av_item in zip(actions_probs, SVs, av):
            assert len(sell_prob) == 2, sell_prob
            flag_holding=self.check_holding_fun(av_item)
            if flag_holding:
                #action = np.random.choice([2, 3], p=action_probs)
                action =self.i_action.I_nets_choose_action(sell_prob)
                l_a.append(action)
                l_ap.append(sell_prob.ravel())
            else:  # not have holding
                action = 0
                l_a.append(action)
                l_ap.append(sell_prob.ravel())
            l_sv.append(SV[0])
        return l_a, l_ap, l_sv
