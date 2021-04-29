import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers import Lambda
import numpy as np
import os,re
import config as sc
from State import *
from action_comm import actionOBOS
class nets_conf:
    """
    @DynamicAttrs
    """
def get_agent_nc(lc):
    assert "CNN" in lc.agent_method_sv and "CNN" in lc.agent_method_joint_lvsv
    nc=nets_conf()
    N_item_list=["lv_shape","sv_shape"]
    CNN_sv_list=["flag_s_level","s_kernel_l","s_filter_l","s_maxpool_l"]
    CNN_lv_list=["flag_l_level","l_kernel_l","l_filter_l","l_maxpool_l"]
    D_list=["dense_l","dense_prob","dense_advent"]
    nc_item_list=[]
    for list_add in [N_item_list,D_list,CNN_sv_list,CNN_lv_list]:
        nc_item_list += list_add
    for item_title in nc_item_list:
        assert item_title in list(lc.net_config.keys()), item_title
        setattr(nc, item_title, lc.net_config[item_title])
    nc.lv_shape = tuple(nc.lv_shape)
    nc.sv_shape = tuple(nc.sv_shape)
    return nc

class common_component:
    def construct_denses(self, dense_list, input_tensor, name):
        assert name is not None
        a = None
        for idx, dense_number in enumerate(dense_list):
            name_prefix = name + "_Dense{0}".format(idx)
            a = keras.layers.Dense(dense_number, activation='relu', name=name_prefix)(input_tensor if idx==0 else a)
        return a

    def Cov_1D_module(self, kernel, filters,max_pool, input_tensor, name):
        assert name is not None
        conv_name,max_pool_name, relu_name= name + '_conv',name + '_pool',name + '_relu'
        a = keras.layers.Conv1D(filters=filters, padding='same', kernel_size=kernel,name=conv_name)(input_tensor)
        if max_pool>1 and a.shape[1]>1:#todo dirty solution for 1D only
            a = keras.layers.MaxPool1D(pool_size=max_pool, padding='valid', name=max_pool_name)(a) #strides=1,不注明就是和pool_size一样
        d = keras.layers.LeakyReLU(name=relu_name)(a)
        return d

    def Cov_2D_module(self, kernel, filters,max_pool, input_tensor, name):
        assert name is not None
        conv_name,max_pool_name, relu_name= name + '_conv',name + '_pool',name + '_relu'
        a = keras.layers.Conv2D(filters=filters, padding='same', kernel_size=kernel,name=conv_name)(input_tensor)
        if max_pool > 1 and a.shape[1]>1 and a.shape[2]>1:  #todo dirty solution for 2D only
            a = keras.layers.MaxPool2D(pool_size=max_pool, padding='valid', name=max_pool_name)(a) #strides=(1,1),不注明就是和pool_size一样
        d = keras.layers.LeakyReLU(name=relu_name)(a)
        return d

    def Inception_1D_module(self,filters,input, name):
        assert name is not None
        cp, pp =name + "_conv",name + "_pool"
        t11nm,t12nm,t21nm,t22nm,t31nm,t32nm=cp+"_t11",cp+"_t12",cp+"_t21",cp+"_t22",pp+"_t31",cp+"_t32"
        outnm = name + "_out"
        t1 = keras.layers.Conv1D(filters=filters, kernel_size=1,padding='same',activation='relu',name=t11nm)(input)
        t1 = keras.layers.Conv1D(filters=filters, kernel_size=3,padding='same',activation='relu',name=t12nm)(t1)
        t2 = keras.layers.Conv1D(filters=filters, kernel_size=1,padding='same',activation='relu',name=t21nm)(input)
        t2 = keras.layers.Conv1D(filters=filters, kernel_size=5,padding='same',activation='relu',name=t22nm)(t2)
        t3 = keras.layers.MaxPool1D(pool_size=3, strides=1, padding='same', name=t31nm)(input)
        t3 = keras.layers.Conv1D(filters=filters, kernel_size=1, padding='same', activation='relu',name=t32nm)(t3)
        output = keras.layers.Concatenate( axis=2, name=outnm)([t1, t2, t3])  # 2d sampel axis =3
        return output

    def Inception_2D_module(self,filters,input, name):
        assert name is not None
        cp, pp =name + "_conv",name + "_pool"
        t11nm,t12nm,t21nm,t22nm,t31nm,t32nm=cp+"_t11",cp+ "_t12",cp+"_t21",cp+"_t22",pp+"_t31",cp+"_t32"
        outnm = name + "_out"
        t1 = keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same', activation='relu', name=t11nm)(input)
        t1 = keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu', name=t12nm)(t1)
        t2 = keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same', activation='relu', name=t21nm)(input)
        t2 = keras.layers.Conv2D(filters=filters, kernel_size=5, padding='same', activation='relu', name=t22nm)(t2)
        t3 = keras.layers.MaxPool2D(pool_size=3, strides=1, padding='same',name=t31nm)(input)
        t3 = keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same', activation='relu',name=t32nm)(t3)
        output = keras.layers.Concatenate( axis=3, name=outnm)([t1, t2, t3])  # 2d sampel axis =3
        return output

class LVSV_component:
    def __init__(self, nc,cc, lc):
        self.nc=nc
        self.cc=cc
        self.lc=lc
    def CNN_get_SV_state(self, input, name):
        immediate_sv = input
        for idx, [kernel, filter, maxpool, flag] in enumerate(zip(self.nc.s_kernel_l, self.nc.s_filter_l,
                                                                  self.nc.s_maxpool_l, self.nc.flag_s_level)):
            assert flag == "C", "sv only support flag ==C"
            prefix = name + "_sv{0}".format(idx)
            conv_nm, pool_nm, relu_nm = prefix + '_conv', prefix + '_pool', prefix + '_relu'

            immediate_sv = keras.layers.TimeDistributed(keras.layers.Conv1D(filters=filter, padding='same',
                                        kernel_size=kernel, name=conv_nm), name="TD_{0}".format(conv_nm))(immediate_sv)
            if maxpool > 1 and immediate_sv.shape[1]>1:
                immediate_sv = keras.layers.TimeDistributed(keras.layers.MaxPool1D(pool_size=maxpool, padding='valid',
                                        name=pool_nm), name="TD_{0}".format(pool_nm))(immediate_sv) #strides=1,不注明就是和pool_size一样
            immediate_sv = keras.layers.TimeDistributed(keras.layers.LeakyReLU(name=relu_nm), name="TD_{0}".format(relu_nm))(immediate_sv)
        output_sv = keras.layers.Lambda(lambda x: keras.backend.squeeze(x, axis=2), name=name + "_SV")(immediate_sv)
        return output_sv


    def CNN_get_LV_SV_joint_state(self, inputs, name):
        input_lv, SV_state = inputs
        immediate_lv = keras.layers.Concatenate(axis=2, name=name + "_LV_SVed")([input_lv, SV_state])
        for idx, [kernel, filter, maxpool, flag] in enumerate(zip(self.nc.l_kernel_l, self.nc.l_filter_l,
                                                                  self.nc.l_maxpool_l, self.nc.flag_l_level)):
            conv_name, pool_name = name + "_LVSV_branch{0}".format(idx), name + "_LVSV_pool{0}".format(idx)
            if flag == "C":
                immediate_lv = self.cc.Cov_1D_module(kernel, filter, maxpool, immediate_lv,name=conv_name)
            else:
                assert flag == "I"
                assert kernel == 0
                immediate_lv_t = self.cc.Inception_1D_module(filter, immediate_lv, name=conv_name)
                if maxpool > 1 and immediate_lv.shape[1]>1:
                    immediate_lv = keras.layers.MaxPool1D(pool_size=maxpool, padding='valid',name=pool_name)(immediate_lv_t) #strides=1,不注明就是和pool_size一样
        lv_sv_joint_state = keras.layers.Reshape((self.nc.l_filter_l[-1],), name=name + "LV_SV_joint")(immediate_lv)
        return lv_sv_joint_state

    def _get_2D_state_base(self,input_vector, name,kernel_l,filter_l,maxpool_l, flag_level, padding_type, flag_stride_1, flag_residence=False):
        assert padding_type in ['same', 'valid']
        if flag_residence:
            l_flag_Lresidence=[]
            flag_not_zero_maxpool_met=False
            for maxpool in maxpool_l:
                if not flag_not_zero_maxpool_met:
                    if maxpool==0:
                        l_flag_Lresidence.append(True)
                    else:
                        flag_not_zero_maxpool_met=True
                        l_flag_Lresidence.append(False)
                else:
                    l_flag_Lresidence.append(False)
        else:
            l_flag_Lresidence=[False for _ in maxpool_l]
        immediate = input_vector
        for idx, [kernel, filter, maxpool, flag_CI,flag_Lresidence ] in enumerate(zip(kernel_l,filter_l,maxpool_l, flag_level,l_flag_Lresidence)):
            assert flag_CI in ["C", "I"]
            prefix=name+str(idx)
            conv_nm, pool_nm, relu_nm = prefix + '_conv', prefix + '_pool', prefix + '_relu'
            if flag_CI == "C":
                assert  immediate.shape[1]!=1 or immediate.shape[2]!=1, "should not be all 1, {0}".format(immediate.shape)
                strides=(1,1) if flag_stride_1 else (kernel if immediate.shape[1]!=1 else 1,kernel if immediate.shape[2]!=1 else 1)
                a = keras.layers.Conv2D(filters=filter, padding=padding_type, kernel_size=kernel, name=conv_nm,strides=strides)(immediate)
                if maxpool > 1 and a.shape[1] > 1 and a.shape[2] > 1:  # todo dirty solution for 2D only
                    if flag_stride_1:
                        a = keras.layers.MaxPool2D(pool_size=maxpool, padding='valid', name=pool_nm)(a)  # strides=(1,1),不注明就是和pool_size一样
                        immediate = keras.layers.LeakyReLU(name=relu_nm)(a)
                    else:
                        assert False, "while taking stride 3 not allow to maxpool"
                else:
                    if not flag_Lresidence:
                        immediate = keras.layers.LeakyReLU(name=relu_nm)(a)
                    else:
                        immediate=keras.layers.LeakyReLU(name=relu_nm)(a)
                        immediate=keras.layers.Concatenate(axis=3, name=prefix+"Res")([input_vector, immediate])  # 2d sampel axis =3
            else:
                assert kernel == 0
                immediate = self.cc.Inception_2D_module(filter, immediate, name=conv_nm)
                if maxpool > 1 and immediate.shape[1]>1 and immediate.shape[2]>1:  #todo dirty solution for 2D only:
                    immediate = keras.layers.MaxPool2D(pool_size=maxpool, padding='valid',name=pool_nm)(immediate) #strides=(1,1),不注明就是和pool_size一样
        output_sv = keras.layers.Flatten(name=name)(immediate)
        return output_sv

    def CNN2D_get_SV_state(self, input_sv, name):
        return self._get_2D_state_base(input_sv, name+"SV",self.nc.s_kernel_l, self.nc.s_filter_l,
                                  self.nc.s_maxpool_l, self.nc.flag_s_level,'same' , True)

    def CNN2D_get_LV_SV_joint_state(self, inputs, name):
        input_lv, SV_state = inputs
        immediate_lv = keras.layers.Reshape((input_lv.shape[1], input_lv.shape[2], 1))(input_lv)

        lv_state= self._get_2D_state_base(immediate_lv, name+"LV",self.nc.l_kernel_l, self.nc.l_filter_l,
                                                             self.nc.l_maxpool_l, self.nc.flag_l_level,'same', True)
        return keras.layers.Concatenate(axis=1, name=name + "LV_SV_joint")([lv_state, SV_state])



    def CNN2Dvalid_get_SV_state(self, input_sv, name):
        return self._get_2D_state_base(input_sv, name+"SV",self.nc.s_kernel_l, self.nc.s_filter_l,
                                  self.nc.s_maxpool_l, self.nc.flag_s_level,'valid' , True)

    def CNN2Dvalid_get_LV_SV_joint_state(self, inputs, name):
        input_lv, SV_state = inputs
        immediate_lv = keras.layers.Reshape((input_lv.shape[1], input_lv.shape[2], 1))(input_lv)
        lv_state= self._get_2D_state_base(immediate_lv, name+"LV",self.nc.l_kernel_l, self.nc.l_filter_l,
                                                             self.nc.l_maxpool_l, self.nc.flag_l_level,'valid', True)
        return keras.layers.Concatenate(axis=1, name=name + "LV_SV_joint")([lv_state, SV_state])

    ##CNN2DV2 only lv
    def CNN2DV2_get_SV_state(self, input_sv, name):
        return input_sv
    def CNN2DV2_get_LV_SV_joint_state(self, inputs, name):
        input_lv, _ = inputs
        immediate_lv = keras.layers.Reshape((input_lv.shape[1], input_lv.shape[2], 1))(input_lv)

        lv_state= self._get_2D_state_base(immediate_lv, name+"LV",self.nc.l_kernel_l, self.nc.l_filter_l,
                                                             self.nc.l_maxpool_l, self.nc.flag_l_level,'same', True)
        return lv_state

    ##CNN2DV3 only sv
    def CNN2DV3_get_SV_state(self, input_sv, name):
        return self._get_2D_state_base(input_sv, name+"SV",self.nc.s_kernel_l, self.nc.s_filter_l,
                                  self.nc.s_maxpool_l, self.nc.flag_s_level,'same' , True)
    def CNN2DV3_get_LV_SV_joint_state(self, inputs, name):
        _, input_sv = inputs
        return input_sv


    def CNN2DV4_get_SV_state(self, input_sv, name):
        return self._get_2D_state_base(input_sv, name+"SV",self.nc.s_kernel_l, self.nc.s_filter_l,
                                  self.nc.s_maxpool_l, self.nc.flag_s_level,'same',False )

    def CNN2DV4_get_LV_SV_joint_state(self, inputs, name):
        input_lv, SV_state = inputs
        immediate_lv = keras.layers.Reshape((input_lv.shape[1], input_lv.shape[2], 1))(input_lv)
        lv_state = self._get_2D_state_base(immediate_lv, name + "LV", self.nc.l_kernel_l,
                                           self.nc.l_filter_l,  # todo the lv not change to stride yet
                                           self.nc.l_maxpool_l, self.nc.flag_l_level,'same',False)
        return keras.layers.Concatenate(axis=1, name=name + "LV_SV_joint")([lv_state, SV_state])

    def CNN2DV5_get_SV_state(self, input_sv, name):
        return self._get_2D_state_base(input_sv, name+"SV",self.nc.s_kernel_l, self.nc.s_filter_l,
                                  self.nc.s_maxpool_l, self.nc.flag_s_level,'same',True, flag_residence=True)

    def CNN2DV5_get_LV_SV_joint_state(self, inputs, name):
        input_lv, SV_state = inputs
        immediate_lv = keras.layers.Reshape((input_lv.shape[1], input_lv.shape[2], 1))(input_lv)
        lv_state = self._get_2D_state_base(immediate_lv, name + "LV", self.nc.l_kernel_l,
                                           self.nc.l_filter_l,
                                           self.nc.l_maxpool_l, self.nc.flag_l_level,'same',True,flag_residence=True)
        return keras.layers.Concatenate(axis=1, name=name + "LV_SV_joint")([lv_state, SV_state])



    def get_ap_av_HP(self, inputs, name):
        if self.lc.flag_use_av_in_model:
            js, input_av = inputs
            input_state = keras.layers.Concatenate(axis=-1,                      name=name + "_input")([js, input_av])
        else:
            input_state = inputs[0]
        state = self.cc.construct_denses(self.nc.dense_l, input_state,       name=name + "_commonD")
        Pre_a = self.cc.construct_denses(self.nc.dense_prob[:-1], state,     name=name + "_Pre_a")
        ap = keras.layers.Dense(self.nc.dense_prob[-1], activation='softmax',name="Action_prob")(Pre_a)
        Pre_sv = self.cc.construct_denses(self.nc.dense_advent[:-1], state,name=name + "_Pre_sv")
        sv = keras.layers.Dense(self.nc.dense_advent[-1], activation='linear',name="State_value")(Pre_sv)
        #sv = keras.layers.Dense(self.nc.dense_advent[-1], activation='tanh', name="State_value")(Pre_sv)
        return ap, sv

class V2OS_4_OB_agent:
    def __init__(self,lc,ob_system_name, Ob_model_tc):
        self.lc=lc
        if ob_system_name!="Just_sell":
            self.model=self._load_model(ob_system_name, Ob_model_tc)
            self.i_cav=globals()[self.lc.CLN_AV_Handler](lc)
            self.predict=self.model_predict
        else:
            #self.predict=lambda x: [np.array([[1,0] for _ in range(lc.batch_size)]),np.NaN]
            self.predict = lambda x: [np.array([[1, 0] for _ in x[0]]), np.NaN] # predict per explore term is all stock list in that process not batch

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
        return model

    def model_predict(self, state):
        lv, sv, av = state
        if self.lc.flag_use_av_in_model:
            p, v = self.model.predict({'P_input_lv': lv, 'P_input_sv': sv, 'P_input_av': self.i_cav.get_OS_AV(av)})
        else:
            p, v = self.model.predict({'P_input_lv': lv, 'P_input_sv': sv})
        return p,v

class net_agent_base:
    def __init__(self, lc):
        self.lc=lc
        self.nc = get_agent_nc(lc)
        self.cc=common_component()
        keras.backend.set_learning_phase(0)  # add by john for error solved by
        self.DC = {
            "method_SV_state": "{0}_get_SV_state".format(lc.agent_method_sv),
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
            self.choose_action=self.V2_choose_action
            self.choose_action_CC=None
            assert self.lc.P2_current_phase == "Train_Sell"
        else:
            self.av_shape = self.lc.OB_AV_shape
            self.get_av = self.i_cav.get_OB_AV
            self.layer_label = "OB"
            assert self.lc.P2_current_phase == "Train_Buy"
            self.choose_action=self.V3_choose_action
            self.choose_action_CC=self.V3_choose_action_CC

    def build_predict_model(self, name):
        input_lv = keras.Input(shape=self.nc.lv_shape, dtype='float32', name="{0}_input_lv".format(name))
        input_sv = keras.Input(shape=self.nc.sv_shape, dtype='float32', name="{0}_input_sv".format(name))
        if self.lc.flag_use_av_in_model:
            input_av = keras.Input(shape=self.av_shape, dtype='float32', name="{0}_input_av".format(name))
            l_agent_output=self.layers_with_av([input_lv,input_sv,input_av], name)
            self.model = keras.Model(inputs=[input_lv, input_sv, input_av], outputs=l_agent_output, name=name)
        else:
            l_agent_output = self.layers_without_av([input_lv, input_sv], name)
            self.model = keras.Model(inputs=[input_lv, input_sv], outputs=l_agent_output, name=name)
        return self.model

    def layers_with_av(self, inputs, name):
        lv,sv,av=inputs
        i_LVSV = LVSV_component(self.nc, self.cc, self.lc)
        sv_state = getattr(i_LVSV, self.DC["method_SV_state"])(sv, name)
        lv_sv_state = getattr(i_LVSV, self.DC["method_LV_SV_joint_state"])([lv, sv_state], name)
        input_method_ap_sv = [lv_sv_state, av]
        l_agent_output = getattr(i_LVSV, self.DC["method_ap_sv"])(input_method_ap_sv, name + self.layer_label)
        return l_agent_output

    def layers_without_av(self, inputs, name):
        lv,sv=inputs
        i_LVSV = LVSV_component(self.nc, self.cc, self.lc)
        sv_state = getattr(i_LVSV, self.DC["method_SV_state"])(sv, name)
        lv_sv_state = getattr(i_LVSV, self.DC["method_LV_SV_joint_state"])([lv, sv_state], name)
        input_method_ap_sv = [lv_sv_state]
        l_agent_output = getattr(i_LVSV, self.DC["method_ap_sv"])(input_method_ap_sv, name + self.layer_label)
        return l_agent_output


    def load_weight(self, weight_fnwp):
        self.model.load_weights(weight_fnwp)

    def predict(self, state):
        lv, sv, av = state
        if self.lc.flag_use_av_in_model:
            p, v = self.model.predict({'P_input_lv': lv, 'P_input_sv': sv, 'P_input_av': self.get_av(av)})
        else:
            p, v = self.model.predict({'P_input_lv': lv, 'P_input_sv': sv})
        return p,v

    def V2_choose_action(self, state, calledby="Eval"):
        assert self.lc.P2_current_phase == "Train_Sell"
        lv, sv, av = state
        actions_probs, SVs = self.predict(state)
        l_a,l_ap,l_sv = [],[],[]
        for sell_prob, SV, av_item in zip(actions_probs, SVs, av):
            assert len(sell_prob) == 2, sell_prob
            flag_holding=self.i_cav.Is_Holding_Item(av_item)
            if flag_holding:
                action =self.i_action.I_nets_choose_action(sell_prob)
                l_a.append(action)
                l_ap.append(sell_prob.ravel())
            else:  # not have holding
                action = 0
                l_a.append(action)
                l_ap.append(sell_prob.ravel())
            l_sv.append(SV[0])
        return l_a, l_ap, l_sv

    def V3_choose_action(self,state,calledby):
        assert self.lc.P2_current_phase == "Train_Buy"
        _, _, av = state
        buy_probs, buy_SVs = self.predict(state)
        if not hasattr(self, "OS_agent"):
            self.OS_agent = V2OS_4_OB_agent(self.lc,self.lc.P2_sell_system_name, self.lc.P2_sell_model_tc)
            self.i_OS_action=actionOBOS("OS")
        #sel_probs, sell_SVs = self.OS_agent.predict(state)
        sel_probs, _ = self.OS_agent.predict(state)
        l_a,l_ap,l_sv = [],[],[]
        #for buy_prob, sell_prob, buy_sv, sell_sv, av_item in zip(buy_probs,sel_probs,buy_SVs,sell_SVs,av):
        for buy_prob, sell_prob, buy_sv, av_item in zip(buy_probs, sel_probs, buy_SVs, av):
            assert len(buy_prob)==2 and len(sell_prob) == 2
            if self.i_cav.Is_Holding_Item(av_item):
                #action = np.random.choice([2, 3], p=sell_prob)
                action = self.i_OS_action.I_nets_choose_action(sell_prob)
                l_a.append(action)
                l_ap.append(np.zeros_like(sell_prob))  # this is add zero and this record will be removed by TD_buffer before send to server for train
                l_sv.append(np.NaN) #l_sv.append(sell_sv[0])
            else: # not have holding
                if self.lc.flag_train_random_explore:
                    if calledby=="Explore":
                        #if np.random.choice([0, 1], p=[0.8,0.2]): #TODO need to find whether configure in config needed:
                        if np.random.choice([0, 1],
                                p=[1-self.lc.train_random_explore_prob_buy, self.lc.train_random_explore_prob_buy]):
                            action=0
                        else:
                            action = self.i_action.I_nets_choose_action(buy_prob)
                    elif calledby=="Eval":
                        action = self.i_action.I_nets_choose_action(buy_prob)
                    else:
                        assert False, "Only support Explore and Eval as calledby not support {0}".format(calledby)
                else:
                    action = self.i_action.I_nets_choose_action(buy_prob)
                l_a.append(action)
                l_ap.append(buy_prob)
                l_sv.append(buy_sv[0])
        return l_a, l_ap,l_sv

    def V3_choose_action_CC(self,state,calledby):
        assert self.lc.P2_current_phase == "Train_Buy"
        buy_probs, buy_SVs = self.predict(state)
        if not hasattr(self, "OS_agent"):
            self.OS_agent = V2OS_4_OB_agent(self.lc,self.lc.P2_sell_system_name, self.lc.P2_sell_model_tc)
            self.i_OS_action=actionOBOS("OS")
        #sel_probs, sell_SVs = self.OS_agent.predict(state)
        sel_probs, _ = self.OS_agent.predict(state)
        #TODO check whether need to convert to list and if there is effecient way to convert to list
        l_buy_a  = [self.i_action.I_nets_choose_action(buy_prob) for buy_prob in buy_probs ]
        l_sell_a  = [self.i_OS_action.I_nets_choose_action(sell_prob) for sell_prob in sel_probs ]
        return l_buy_a,l_sell_a  # This only used in CC eval ,so AP and sv information in not necessary

    def V3_get_AP_AT(self,state,calledby):
        assert self.lc.P2_current_phase == "Train_Buy"
        buy_probs, buy_SVs = self.predict(state)
        if not hasattr(self, "OS_agent"):
            self.OS_agent = V2OS_4_OB_agent(self.lc,self.lc.P2_sell_system_name, self.lc.P2_sell_model_tc)
            self.i_OS_action=actionOBOS("OS")
        #sel_probs, sell_SVs = self.OS_agent.predict(state)
        sel_probs, _ = self.OS_agent.predict(state)
        return buy_probs,sel_probs  # This only used in CC eval ,so AP and sv information in not necessary

