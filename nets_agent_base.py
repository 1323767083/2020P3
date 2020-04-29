import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import os,re
import config as sc
from av_state import Phase_State_V8,Phase_State_V3
def init_nets_agent_base(ilc, inc,iLNM_LV_SV_joint,iLNM_P, iLNM_V):
    global lc,nc
    lc,nc = ilc, inc
    global LNM_LV_SV_joint,LNM_P,LNM_V, cc
    LNM_LV_SV_joint = iLNM_LV_SV_joint
    LNM_P = iLNM_P
    LNM_V = iLNM_V
    cc = common_component()

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
    def CNN_get_SV_state(self, input, name):
        input_sv = input
        immediate_sv = input_sv
        for idx, [kernel, filter, maxpool, flag] in enumerate(zip(nc.s_kernel_l, nc.s_filter_l,nc.s_maxpool_l, nc.flag_s_level)):
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
        last_index = len(nc.ms_param_TimeDistributed) - 1
        assert last_index > 0
        immediate_sv = input_sv
        for idx, param in enumerate(nc.ms_param_TimeDistributed):
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
    def _RNN_LV_Sved(self,LV_SVed, name):
        immediate_lv=LV_SVed
        last_index = len(nc.ms_param_CuDNNLSTM) - 1
        assert last_index > 0
        for idx, param in enumerate(nc.ms_param_CuDNNLSTM):
            if idx != last_index:
                immediate_lv = keras.layers.LSTM(param, return_sequences=True, name="{0}_LVSV_LSTM_{1}".format(name, idx))(immediate_lv)
            else:
                immediate_lv = keras.layers.LSTM(param, name="{0}_LVSV_LSTM_{1}".format(name, idx))(immediate_lv)
        return immediate_lv

    def _CNN_LV_SVed(self,LV_SVed, name):
        inception_fn = getattr(cc, "Inception_1D_module")
        immediate_lv = LV_SVed
        for idx, [kernel, filter, maxpool, flag] in enumerate(zip(nc.l_kernel_l, nc.l_filter_l,
                                                                  nc.l_maxpool_l, nc.flag_l_level)):
            conv_name, pool_name = name + "_LVSV_branch{0}".format(idx), name + "_LVSV_pool{0}".format(idx)
            if flag == "C":
                immediate_lv = cc.Cov_1D_module(kernel, filter, maxpool, immediate_lv,name=conv_name)
            else:
                assert flag == "I"
                assert kernel == 0
                immediate_lv_t = inception_fn(filter, immediate_lv, name=conv_name)
                immediate_lv = keras.layers.MaxPooling1D(pool_size=maxpool, padding='valid', name=pool_name)(immediate_lv_t)
        immediate_lv = keras.layers.Reshape((nc.l_filter_l[-1],), name=name+"_LVSV_reshaped")(immediate_lv)
        return immediate_lv

    def RNN_get_LV_SV_joint_state(self, inputs, name):
        input_lv, SV_state = inputs
        LV_SVed = keras.layers.Concatenate(axis=2, name=name + "_LV_SVed")([input_lv, SV_state])
        immediate_lv = self._RNN_LV_Sved(LV_SVed, name)
        lv_sv_joint_state= keras.layers.Lambda(lambda  x: x, name=name + LNM_LV_SV_joint)(immediate_lv)
        return lv_sv_joint_state


    def CNN_get_LV_SV_joint_state(self, inputs, name):
        input_lv, SV_state = inputs
        LV_SVed = keras.layers.Concatenate(axis=2, name=name + "_LV_SVed")([input_lv, SV_state])
        immediate_lv=self._CNN_LV_SVed(LV_SVed, name)
        #lv_sv_joint_state = Reshape((nc.l_filter_l[-1],), name=LNM_LV_SV_joint)(immediate_lv)
        lv_sv_joint_state = keras.layers.Lambda(lambda x: x, name=name + LNM_LV_SV_joint)(immediate_lv)
        return lv_sv_joint_state

    def RCN_get_LV_SV_joint_state(self, inputs, name):
        input_lv, SV_state = inputs
        LV_SVed = keras.layers.Concatenate(axis=2, name=name + "_LV_SVed")([input_lv, SV_state])
        immediate_rnn_lv = self._RNN_LV_Sved(LV_SVed, name)
        immediate_cnn_lv = self._CNN_LV_SVed(LV_SVed, name)
        lv_sv_joint_state = keras.layers.Concatenate(axis=1, name=name + LNM_LV_SV_joint)([immediate_rnn_lv, immediate_cnn_lv])
        return lv_sv_joint_state

class V2OS_4_OB_agent:
    def __init__(self,ob_system_name, Ob_model_tc):
        self._load_model(ob_system_name, Ob_model_tc)
        if  hasattr(lc.specific_param,"CLN_AV"):
            self.get_OS_AV =globals()[lc.specific_param.CLN_AV]().get_OS_av
        else:
            self.get_OS_AV = LHPP2V2_get_AV


    def _load_model(self, ob_system_name, Ob_model_tc):
        OB_model_dir=os.path.join(sc.base_dir_RL_system, ob_system_name, "model")
        model_config_fnwp=os.path.join(OB_model_dir, "config.json")
        regex = r'weight_\w+T{0}.h5'.format(Ob_model_tc)
        lfn=[fn for fn in os.listdir(OB_model_dir) if re.findall(regex, fn)]
        assert len(lfn)==1, "{0} model with train count {1} not found".format(ob_system_name,Ob_model_tc)
        weight_fnwp=os.path.join(OB_model_dir, lfn[0])
        load_jason_custom_objects={"softmax": keras.backend.softmax,"tf":tf, "concatenate":keras.backend.concatenate,"lc":lc}
        model = keras.models.model_from_json(open(model_config_fnwp, "r").read(),custom_objects=load_jason_custom_objects)
        model.load_weights(weight_fnwp)
        print("successful load model form {0} {1}".format(model_config_fnwp, weight_fnwp))
        self.OS_model=model
        #return model

    def predict(self, state):
        lv, sv, av = state
        #p, v = model.predict({'P_input_lv': lv, 'P_input_sv': sv, 'P_input_av': av})
        p, v = self.OS_model.predict({'P_input_lv': lv, 'P_input_sv': sv, 'P_input_av': self.get_OS_AV(av)})
        return p,v

LHPP2V2_check_holding = lambda av_item: False if av_item[0] == 1 else True
LHPP2V2_get_AV=lambda n_av:np.concatenate([n_av[:,:lc.LHP+1]])
#LHPP2V7_get_AV=lambda n_av:np.concatenate([n_av[:,1:2],n_av[:,lc.LHP+1:]], axis=-1)
Train_Buy_get_AV_2=lambda n_av:np.concatenate([n_av[:,1:2],n_av[:,lc.LHP+1:]], axis=-1)
