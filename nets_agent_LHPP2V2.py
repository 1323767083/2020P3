from nets_agent_base import *
from action_comm import actionOBOS
class LHPP2V2_Agent(net_aget_base):
    def __init__(self,lc):
        net_aget_base.__init__(self,lc)

    def choose_action(self, state, calledby="Eval"):
        assert self.lc.P2_current_phase == "Train_Sell"
        lv, sv, av = state
        actions_probs, SVs = self.predict(state)
        l_a = []
        l_ap = []
        l_sv = []
        for sell_prob, SV, av_item in zip(actions_probs, SVs, av):
            assert len(sell_prob) == 2, sell_prob
            flag_holding=self.i_cav.Is_Holding_Item(av_item)
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

'''
class LHPP2V2_Agent_old:
    def __init__(self, lc):
        self.lc=lc
        self.nc = get_agent_nc(lc)
        self.cc=common_component()
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
        input_av = keras.Input(shape=self.lc.OS_AV_shape, dtype='float32', name="{0}_input_av".format(name))
        i_SV = SV_component(self.nc)
        i_LV_SV = LV_SV_joint_component(self.nc, self.cc)

        sv_state = getattr(i_SV, self.DC["method_SV_state"])(input_sv, name)
        lv_sv_state = getattr(i_LV_SV, self.DC["method_LV_SV_joint_state"])([input_lv, sv_state], name + "for_ap")

        if not self.lc.flag_sv_joint_state_stop_gradient:
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
        state = self.cc.construct_denses(self.nc.dense_l, input_state,        name=label + "_commonD")

        Pre_aSell = self.cc.construct_denses(self.nc.dense_prob[:-1], state,  name=label + "_Pre_aSell")
        ap = keras.layers.Dense(self.nc.dense_prob[-1], activation='softmax',      name=self.nc.LNM_P)(Pre_aSell)

        if self.lc.flag_sv_stop_gradient:
            sv_state=keras.layers.Lambda(lambda x: tf.stop_gradient(x),          name=label + "_stop_gradiant_sv")(state)
        else:
            sv_state = keras.layers.Lambda(lambda x: x,                          name=label + "_not_stop_gradiant_sv")(state)
        Pre_sv = self.cc.construct_denses(self.nc.dense_advent[:-1], sv_state,name=label + "_Pre_sv")
        sv = keras.layers.Dense(self.nc.dense_advent[-1], activation='linear',        name=self.nc.LNM_V)(Pre_sv)
        return ap, sv

    #SP means seperate ap sv 's lv_sv_jiong_state
    def get_ap_av_HP_SP(self, inputs, name):
        ap_js, sv_js, input_av = inputs

        aplabel = name + "_OS_ap"
        svlabel = name + "_OS_sv"
        ap_input_state = keras.layers.Concatenate(axis=-1, name=aplabel + "_input")([ap_js, input_av])
        sv_input_state = keras.layers.Concatenate(axis=-1, name=svlabel + "_input")([sv_js, input_av])


        ap_state = self.cc.construct_denses(self.nc.dense_l, ap_input_state,    name=aplabel + "_commonD")


        Pre_aSell = self.cc.construct_denses(self.nc.dense_prob[:-1], ap_state, name=aplabel + "_Pre_aSell")
        ap = keras.layers.Dense(self.nc.dense_prob[-1], activation='softmax',        name=self.nc.LNM_P)(Pre_aSell)

        sv_state = self.cc.construct_denses(self.nc.dense_l, sv_input_state,    name=svlabel + "_commonD")
        Pre_sv = self.cc.construct_denses(self.nc.dense_advent[:-1], sv_state,  name=svlabel + "_Pre_sv")
        sv = keras.layers.Dense(self.nc.dense_advent[-1], activation='linear',          name=self.nc.LNM_V)(Pre_sv)

        return ap,sv

    def get_ap_av_HP_DAV(self, inputs, name):#DAV means deep down av_input
        label = name + "_OS"


        input_state, input_av = inputs


        #input_state = Concatenate(axis=-1,                          name=label + "_input")([js, input_av])
        state = self.cc.construct_denses(self.nc.dense_l, input_state,        name=label + "_commonD")

        itermediate_state = keras.layers.Concatenate(axis=-1, name=label + "_input")([state, input_av])

        Pre_aSell = self.cc.construct_denses(self.nc.dense_prob[:-1], itermediate_state,  name=label + "_Pre_aSell")
        ap = keras.layers.Dense(self.nc.dense_prob[-1], activation='softmax',      name=self.nc.LNM_P)(Pre_aSell)

        if self.lc.flag_sv_stop_gradient:
            sv_state = keras.layers.Lambda(lambda x: tf.stop_gradient(x),          name=label + "_stop_gradiant_sv")(itermediate_state)
        else:
            sv_state = keras.layers.Lambda(lambda x: x,                          name=label + "_not_stop_gradiant_sv")(itermediate_state)
        Pre_sv = self.cc.construct_denses(self.nc.dense_advent[:-1], sv_state,name=label + "_Pre_sv")
        sv = keras.layers.Dense(self.nc.dense_advent[-1], activation='linear',        name=self.nc.LNM_V)(Pre_sv)
        return ap, sv

    def predict(self, state):
        assert self.lc.P2_current_phase=="Train_Sell"
        lv, sv, av = state
        p, v = self.OS_model.predict({'P_input_lv': lv, 'P_input_sv': sv, 'P_input_av': self.i_cav.get_OS_AV(av)})
        return p,v

    def choose_action(self, state, calledby="Eval"):
        assert self.lc.P2_current_phase == "Train_Sell"
        lv, sv, av = state
        actions_probs, SVs = self.predict(state)
        l_a = []
        l_ap = []
        l_sv = []
        for sell_prob, SV, av_item in zip(actions_probs, SVs, av):
            assert len(sell_prob) == 2, sell_prob
            flag_holding=self.i_cav.Is_Holding_Item(av_item)
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
'''