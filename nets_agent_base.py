import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import os,re
import config as sc
from State import *
from action_comm import actionOBOS
class nets_conf:
    """
    @DynamicAttrs
    """
    LNM_LV_SV_joint = "State_LSV"
    LNM_P = "Act_prob"
    LNM_V = "State_value"

def get_agent_nc(lc):
    nc=nets_conf()
    N_item_list=["lv_shape","sv_shape"]

    CNN_sv_list=["flag_s_level","s_kernel_l","s_filter_l","s_maxpool_l"]
    RNN_sv_list=["ms_param_TimeDistributed"]

    CNN_lv_list=["flag_l_level","l_kernel_l","l_filter_l","l_maxpool_l"]
    RNN_lv_list=["ms_param_CuDNNLSTM"]

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
        nc_item_list += RNN_lv_list
    elif lc.agent_method_joint_lvsv=="CNN":
        nc_item_list += CNN_lv_list
    else:
        assert lc.agent_method_joint_lvsv=="RCN"
        nc_item_list += RNN_lv_list
        nc_item_list += CNN_lv_list


    for item_title in nc_item_list:
        assert item_title in list(lc.net_config.keys())
        setattr(nc, item_title, lc.net_config[item_title])

    nc.lv_shape = tuple(nc.lv_shape)
    nc.sv_shape = tuple(nc.sv_shape)
    return nc

class common_component:
    def construct_conv_branch(self, cov_kernel_list, cov_filter_list, max_pool_list, input_tensor):
        a, d = None, None
        for idx, [kernel, filter, maxpool] in enumerate(zip(cov_kernel_list, cov_filter_list, max_pool_list)):
            a = keras.layers.Conv1D(filters=filter, padding='same', kernel_size=kernel)(input_tensor if idx==0 else d)
            b = keras.layers.MaxPooling1D(pool_size=maxpool, padding='valid')(a)
            c = keras.layers.BatchNormalization()(b)
            d = keras.layers.LeakyReLU()(c)
        return d
    def construct_lstms(self,lstm_units_list, input_tensor, name=None):
        a = None
        if len(lstm_units_list) == 1:
            name_prefix = name + "_LSTM{0}".format(0) if name is not None else None
            a = keras.layers.LSTM(lstm_units_list[0], name=name_prefix)(input_tensor)
        else:
            for idx, lstm_units in enumerate(lstm_units_list):
                name_prefix = name + "_LSTM{0}".format(idx) if name is not None else None
                if idx == len(lstm_units_list) - 1:
                    a = keras.layers.LSTM(lstm_units,name=name_prefix)(a)
                else:
                    a = keras.layers.LSTM(lstm_units, return_sequences=True,name=name_prefix)(input_tensor if idx==0 else a)
        return a
    def construct_denses(self, dense_list, input_tensor, name=None):
        a = None
        for idx, dense_number in enumerate(dense_list):
            name_prefix = name + "_Dense{0}".format(idx) if name is not None else None
            a = keras.layers.Dense(dense_number, activation='relu', name=name_prefix)(input_tensor if idx==0 else a)
        return a
    def Cov_1D_module(self, kernel, filters,max_pool, input_tensor, name=None):
        if name is not None:
            conv_name,max_pool_name, relu_name= name + '_conv',name + '_pool',name + '_relu'
        else:
            conv_name, max_pool_name, relu_name = None, None, None
        a = keras.layers.Conv1D(filters=filters, padding='same', kernel_size=kernel,name=conv_name)(input_tensor)
        if max_pool>1:
            a = keras.layers.MaxPooling1D(pool_size=max_pool, padding='valid', name=max_pool_name)(a)
        d = keras.layers.LeakyReLU(name=relu_name)(a)

        return d

    def Inception_1D_module(self,filters,input, name=None):
        if name is not None:
            cp, pp =name + "_conv",name + "_pool"
            tower11_name,tower12_name,tower21_name,tower22_name,tower31_name,tower32_name=\
                cp+"_t11",cp+ "_t12",cp+"_t21",cp+"_t22",pp+"_t31",cp+"_t32"
            out_name = name + "_out"
        else:
            tower11_name, tower12_name, tower21_name, tower22_name, tower31_name, tower32_name=\
                None,None,None,None,None,None
            out_name = None
        tower_1 = keras.layers.Conv1D(filters=filters, kernel_size=1, padding='same', activation='relu', name=tower11_name)(input)
        tower_1 = keras.layers.Conv1D(filters=filters, kernel_size=3, padding='same', activation='relu', name=tower12_name )(tower_1)
        tower_2 = keras.layers.Conv1D(filters=filters, kernel_size=1, padding='same', activation='relu', name=tower21_name)(input)
        tower_2 = keras.layers.Conv1D(filters=filters, kernel_size=5, padding='same', activation='relu', name=tower22_name)(tower_2)
        tower_3 = keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same', name=tower31_name)(input)
        tower_3 = keras.layers.Conv1D(filters=filters, kernel_size=1, padding='same', activation='relu',name=tower32_name)(tower_3)
        output = keras.layers.Concatenate( axis=2, name=out_name)([tower_1, tower_2, tower_3])  # 2d sampel axis =3
        return output

class SV_component:
    def __init__(self,nc):
        self.nc=nc
    def CNN_get_SV_state(self, input, name):
        input_sv = input
        immediate_sv = input_sv
        for idx, [kernel, filter, maxpool, flag] in enumerate(zip(self.nc.s_kernel_l, self.nc.s_filter_l,self.nc.s_maxpool_l, self.nc.flag_s_level)):
            assert flag == "C", "sv only support flag ==C"
            prefix = name + "_sv{0}".format(idx)
            conv_nm, pool_nm, relu_nm = prefix + '_conv', prefix + '_pool', prefix + '_relu'

            immediate_sv = keras.layers.TimeDistributed(keras.layers.Conv1D(filters=filter, padding='same', kernel_size=kernel, name=conv_nm)
                                           , name="TD_{0}".format(conv_nm))(immediate_sv)
            if maxpool > 1:
                immediate_sv = keras.layers.TimeDistributed(keras.layers.MaxPooling1D(pool_size=maxpool, padding='valid', name=pool_nm)
                                               , name="TD_{0}".format(pool_nm))(immediate_sv)
            immediate_sv = keras.layers.TimeDistributed(keras.layers.LeakyReLU(name=relu_nm), name="TD_{0}".format(relu_nm))(immediate_sv)

        output_sv = keras.layers.Lambda(lambda x: keras.backend.squeeze(x, axis=2), name=name + "_sv_squeezed")(immediate_sv)
        return output_sv

    def RNN_get_SV_state(self, input, name="SV"):
        input_sv = input
        last_index = len(self.nc.ms_param_TimeDistributed) - 1
        assert last_index > 0
        immediate_sv = input_sv
        for idx, param in enumerate(self.nc.ms_param_TimeDistributed):
            if idx != last_index:
                immediate_sv = keras.layers.TimeDistributed(keras.layers.LSTM(param, return_sequences=True, name="{0}_SV_LSTM_{1}".format(name, idx)),
                                    name="TD_{0}".format(idx))(immediate_sv)
            else:
                immediate_sv = keras.layers.TimeDistributed(keras.layers.LSTM(param, name="{0}_SV_LSTM_{1}".format(name, idx)),
                                    name="TD_{0}".format(idx))(immediate_sv)
        return immediate_sv

    def RCN_get_SV_state(self, input, name):
        CNN_immediate_sv=self.CNN_get_SV_state(input, name)
        CuDNNLSTM__immediate_sv= self.RNN_get_SV_state(input, name)
        immediate_sv = keras.layers.Concatenate(axis=2, name="{0}_SV_JOINT".format(name))([CNN_immediate_sv, CuDNNLSTM__immediate_sv])
        return immediate_sv

class LV_SV_joint_component:
    def __init__(self, nc,cc):
        self.nc=nc
        self.cc=cc
    def _RNN_LV_Sved(self,LV_SVed, name):
        immediate_lv=LV_SVed
        last_index = len(self.nc.ms_param_CuDNNLSTM) - 1
        assert last_index > 0
        for idx, param in enumerate(self.nc.ms_param_CuDNNLSTM):
            if idx != last_index:
                immediate_lv = keras.layers.LSTM(param, return_sequences=True, name="{0}_LVSV_LSTM_{1}".format(name, idx))(immediate_lv)
            else:
                immediate_lv = keras.layers.LSTM(param, name="{0}_LVSV_LSTM_{1}".format(name, idx))(immediate_lv)
        return immediate_lv

    def _CNN_LV_SVed(self,LV_SVed, name):
        inception_fn = getattr(self.cc, "Inception_1D_module")
        immediate_lv = LV_SVed
        for idx, [kernel, filter, maxpool, flag] in enumerate(zip(self.nc.l_kernel_l, self.nc.l_filter_l,
                                                                  self.nc.l_maxpool_l, self.nc.flag_l_level)):
            conv_name, pool_name = name + "_LVSV_branch{0}".format(idx), name + "_LVSV_pool{0}".format(idx)
            if flag == "C":
                immediate_lv = self.cc.Cov_1D_module(kernel, filter, maxpool, immediate_lv,name=conv_name)
            else:
                assert flag == "I"
                assert kernel == 0
                immediate_lv_t = inception_fn(filter, immediate_lv, name=conv_name)
                immediate_lv = keras.layers.MaxPooling1D(pool_size=maxpool, padding='valid', name=pool_name)(immediate_lv_t)
        immediate_lv = keras.layers.Reshape((self.nc.l_filter_l[-1],), name=name+"_LVSV_reshaped")(immediate_lv)
        return immediate_lv

    def RNN_get_LV_SV_joint_state(self, inputs, name):
        input_lv, SV_state = inputs
        LV_SVed = keras.layers.Concatenate(axis=2, name=name + "_LV_SVed")([input_lv, SV_state])
        immediate_lv = self._RNN_LV_Sved(LV_SVed, name)
        lv_sv_joint_state= keras.layers.Lambda(lambda  x: x, name=name + self.nc.LNM_LV_SV_joint)(immediate_lv)
        return lv_sv_joint_state


    def CNN_get_LV_SV_joint_state(self, inputs, name):
        input_lv, SV_state = inputs
        LV_SVed = keras.layers.Concatenate(axis=2, name=name + "_LV_SVed")([input_lv, SV_state])
        immediate_lv=self._CNN_LV_SVed(LV_SVed, name)
        #lv_sv_joint_state = Reshape((nc.l_filter_l[-1],), name=LNM_LV_SV_joint)(immediate_lv)
        lv_sv_joint_state = keras.layers.Lambda(lambda x: x, name=name + self.nc.LNM_LV_SV_joint)(immediate_lv)
        return lv_sv_joint_state

    def RCN_get_LV_SV_joint_state(self, inputs, name):
        input_lv, SV_state = inputs
        LV_SVed = keras.layers.Concatenate(axis=2, name=name + "_LV_SVed")([input_lv, SV_state])
        immediate_rnn_lv = self._RNN_LV_Sved(LV_SVed, name)
        immediate_cnn_lv = self._CNN_LV_SVed(LV_SVed, name)
        lv_sv_joint_state = keras.layers.Concatenate(axis=1, name=name + self.nc.LNM_LV_SV_joint)([immediate_rnn_lv, immediate_cnn_lv])
        return lv_sv_joint_state

class V2OS_4_OB_agent:
    def __init__(self,lc,ob_system_name, Ob_model_tc):
        self.lc=lc
        self._load_model(ob_system_name, Ob_model_tc)
        self.i_cav=globals()[self.lc.CLN_AV_Handler](lc)
    def _load_model(self, ob_system_name, Ob_model_tc):
        OB_model_dir=os.path.join(sc.base_dir_RL_system, ob_system_name, "model")
        model_config_fnwp=os.path.join(OB_model_dir, "config.json")
        regex = r'weight_\w+T{0}.h5'.format(Ob_model_tc)
        lfn=[fn for fn in os.listdir(OB_model_dir) if re.findall(regex, fn)]
        assert len(lfn)==1, "{0} model with train count {1} not found".format(ob_system_name,Ob_model_tc)
        weight_fnwp=os.path.join(OB_model_dir, lfn[0])
        load_jason_custom_objects={"softmax": keras.backend.softmax,"tf":tf, "concatenate":keras.backend.concatenate,"lc":self.lc}
        model = keras.models.model_from_json(open(model_config_fnwp, "r").read(),custom_objects=load_jason_custom_objects)
        model.load_weights(weight_fnwp)
        print("successful load model form {0} {1}".format(model_config_fnwp, weight_fnwp))
        self.OS_model=model
        #return model

    def predict(self, state):
        lv, sv, av = state
        #p, v = model.predict({'P_input_lv': lv, 'P_input_sv': sv, 'P_input_av': av})
        p, v = self.OS_model.predict({'P_input_lv': lv, 'P_input_sv': sv, 'P_input_av': self.i_cav.get_OS_AV(av)})
        return p,v

class net_aget_base:
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
        assert self.lc.system_type in["LHPP2V2","LHPP2V3"]
        if self.lc.system_type== "LHPP2V2":
            self.av_shape = self.lc.OS_AV_shape
            self.get_av = self.i_cav.get_OS_AV
            self.layer_label = "OS"
            assert self.lc.P2_current_phase == "Train_Sell"
        else:
            self.av_shape = self.lc.OB_AV_shape
            self.get_av = self.i_cav.get_OB_AV
            self.layer_label = "OB"
            assert self.lc.P2_current_phase == "Train_Buy"

    def build_predict_model(self, name):
        input_lv = keras.Input(shape=self.nc.lv_shape, dtype='float32', name="{0}_input_lv".format(name))
        input_sv = keras.Input(shape=self.nc.sv_shape, dtype='float32', name="{0}_input_sv".format(name))
        input_av = keras.Input(shape=self.av_shape, dtype='float32', name="{0}_input_av".format(name))
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
        self.model = keras.Model(inputs=[input_lv, input_sv, input_av], outputs=l_agent_output, name=name)
        return self.model

    def load_weight(self, weight_fnwp):
        self.model.load_weights(weight_fnwp)

    #HP means status include holding period
    def get_ap_av_HP(self, inputs, name):
        js, input_av = inputs
        label = name + "_"+ self.layer_label

        input_state = keras.layers.Concatenate(axis=-1,                          name=label + "_input")([js, input_av])
        state = self.cc.construct_denses(self.nc.dense_l, input_state,        name=label + "_commonD")
        Pre_a = self.cc.construct_denses(self.nc.dense_prob[:-1], state,  name=label + "_Pre_a")
        ap = keras.layers.Dense(self.nc.dense_prob[-1], activation='softmax',      name=self.nc.LNM_P)(Pre_a)
        if self.lc.flag_sv_stop_gradient:
            sv_state=keras.layers.Lambda(lambda x: tf.stop_gradient(x),          name=label + "_stop_gradiant_sv")(state)
        else:
            sv_state = keras.layers.Lambda(lambda x: x,                          name=label + "_not_stop_gradiant_sv")(state)
        Pre_sv = self.cc.construct_denses(self.nc.dense_advent[:-1], sv_state,name=label + "_Pre_sv")
        sv = keras.layers.Dense(self.nc.dense_advent[-1], activation='linear',        name=self.nc.LNM_V)(Pre_sv)
        return ap, sv

    def predict(self, state):
        lv, sv, av = state
        p, v = self.model.predict({'P_input_lv': lv, 'P_input_sv': sv, 'P_input_av': self.get_av(av)})
        return p,v
