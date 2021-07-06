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
def get_agent_nc(lc):
    #assert "CNN" in lc.agent_method_sv and "CNN" in lc.agent_method_joint_lvsv
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
    def __init__(self, nc, lc):
        self.nc=nc
        self.lc=lc

    def construct_denses(self, dense_list, input_tensor, name):
        assert name is not None
        a = None
        for idx, dense_number in enumerate(dense_list):
            name_prefix = name + "_Dense{0}".format(idx)
            a = keras.layers.Dense(dense_number, activation='relu', name=name_prefix)(input_tensor if idx==0 else a)
        return a

    def Residule_get_LV_SV_joint_state(self, inputs, name):
        input_lv, input_sv = inputs
        lv_net = Residule_Conv1D_Compoment(self.nc.l_filter_l, self.nc.l_kernel_l,[1 for _ in self.nc.l_filter_l], f"{name}lv")
        sv_net = Repeat_Residule_Conv1D_Compoment(self.nc.s_filter_l, self.nc.s_kernel_l, [1 for _ in self.nc.s_filter_l], f"{name}sv")
        svr_net = Residule_Conv1D_Compoment([64,128], [3,3],[1 for _ in range(2)], f"{name}svr")
        lvr = lv_net.Residule_Conv1D(input_lv)
        svr = sv_net.Repeat_Residule_Conv1D(input_sv)
        svr =keras.layers.Reshape((5,128),name=name + "ReshapeSVr")(svr)
        svrr=svr_net.Residule_Conv1D(svr)
        lsvr = tf.keras.layers.Concatenate(axis=-1)([lvr, svrr])
        return lsvr

    def Inception_get_LV_SV_joint_state(self, inputs, name):
        input_lv, input_sv = inputs
        lvr = Inception_Layer(num_group=2,width=3,nb_filters=32,name=f"{name}lv").Layers(input_lv)
        svr = Repeat_INCEPTION_Layer(num_group=3,width=3,nb_filters=32,name=f"{name}sv").Layer(input_sv)
        svr = keras.layers.Reshape((5, 128), name=name + "ReshapeSVr")(svr)
        svrr=Inception_Layer(num_group=2,width=3,nb_filters=32,name=f"{name}svr").Layers(svr)
        lsvr = tf.keras.layers.Concatenate(axis=-1)([lvr, svrr])
        return lsvr

    def Inception1L_get_LV_SV_joint_state(self, inputs, name):
        input_lv, input_sv = inputs
        lvr = Inception_Layer(num_group=2,width=3,nb_filters=32,name=f"{name}lv").Layers(input_lv)
        svr = Repeat_INCEPTION_Layer(num_group=3,width=3,nb_filters=32,name=f"{name}sv").Layer(input_sv)
        svr = keras.layers.Reshape((640, 1), name=name + "ReshapeSVr")(svr)
        svrr=  Inception_Layer(num_group=2, width=3, nb_filters=32, name=f"{name}svr").Layers(svr)
        lsvr = tf.keras.layers.Concatenate(axis=-1)([lvr, svrr])
        return lsvr

    def InceptionLSTM_get_LV_SV_joint_state(self, inputs, name):
        input_lv, input_sv = inputs
        lvr = Inception_Layer(num_group=2,width=3,nb_filters=32,name=f"{name}lv").Layers(input_lv)
        svr = Repeat_INCEPTION_Layer(num_group=3,width=3,nb_filters=32,name=f"{name}sv").Layer(input_sv)
        svr = keras.layers.Reshape((5, 128), name=name + "ReshapeSVr")(svr)
        svrr= keras.layers.LSTM(512)(svr)
        lsvr = tf.keras.layers.Concatenate(axis=-1)([lvr, svrr])
        return lsvr

    def InceptionLSTMs_get_LV_SV_joint_state(self, inputs, name):
        input_lv, input_sv = inputs
        lvr = Inception_Layer(num_group=2,width=3,nb_filters=32,name=f"{name}lv").Layers(input_lv)
        svr = Repeat_INCEPTION_Layer(num_group=3,width=3,nb_filters=32,name=f"{name}sv").Layer(input_sv)
        svr = keras.layers.Reshape((5, 128), name=name + "ReshapeSVr")(svr)
        lstmdepth=3
        assert lstmdepth>1
        svrr=svr
        for idx in list(range(lstmdepth)):
            if idx==lstmdepth-1:
                svrr = keras.layers.LSTM(512)(svrr)
            else:
                svrr = keras.layers.LSTM(512, return_sequences=True)(svrr)
        lsvr = tf.keras.layers.Concatenate(axis=-1)([lvr, svrr])
        return lsvr

    def LSTMsLSTMs_get_LV_SV_joint_state(self, inputs, name):
        input_lv, input_sv = inputs
        lv_net=LSTMs(4)
        sv_net=TimeDistributedLSTMs(4)
        svr_net=LSTMs(4)
        lv = lv_net.Layers(input_lv)
        sv =  sv_net.Layers(input_sv[:,:5,:,:])
        svr = keras.layers.Reshape((5, 512), name=name + "ReshapeSVr")(sv)
        svrr=svr_net.Layers(svr)
        lsvr = tf.keras.layers.Concatenate(axis=-1)([lv, svrr])
        return lsvr


    def get_ap_av_HP(self, inputs, name):
        input_state = inputs[0]
        state = self.construct_denses(self.nc.dense_l, input_state,       name=name + "_commonD")
        Pre_a = self.construct_denses(self.nc.dense_prob[:-1], state,     name=name + "_Pre_a")
        ap = keras.layers.Dense(self.nc.dense_prob[-1], activation='softmax',name="Action_prob")(Pre_a)
        Pre_sv = self.construct_denses(self.nc.dense_advent[:-1], state,name=name + "_Pre_sv")
        sv = keras.layers.Dense(self.nc.dense_advent[-1], activation='linear',name="State_value")(Pre_sv)
        return ap, sv

class V2OS_4_OB_agent:
    def __init__(self,lc,ob_system_name, Ob_model_tc):
        self.lc=lc
        assert ob_system_name=="Just_sell", "Only support Just_sell"
        self.predict = lambda x: [np.array([[1, 0] for _ in x[0]]), np.NaN]

class LSTMs:
    def __init__(self, depth):
        self.depth=depth
        self.L_lstm =[]
        for idx in list(range(self.depth)):
            if idx==self.depth-1:
                self.L_lstm.append(keras.layers.LSTM(512))
            else:
                self.L_lstm.append(keras.layers.LSTM(512, return_sequences=True))
    def Layers(self, input_tensor):
        x=input_tensor
        for idx in list(range(self.depth)):
            x=self.L_lstm[idx](x)
        return x

class TimeDistributedLSTMs:
    def __init__(self, depth):
        self.depth=depth
        self.L_lstm =[]
        for idx in list(range(self.depth)):
            if idx==self.depth-1:
                self.L_lstm.append(keras.layers.TimeDistributed(keras.layers.LSTM(512)))
            else:
                self.L_lstm.append(keras.layers.TimeDistributed(keras.layers.LSTM(512, return_sequences=True)))
    def Layers(self, input_tensor):
        x=input_tensor
        for idx in list(range(self.depth)):
            x=self.L_lstm[idx](x)
        return x


class Residule_Conv1D_Compoment:
    def __init__(self, lfilter, lkernel, lstride, name):
        self.lconv, self.lbn, self.lactivation, self.lproject = [], [], [], []
        for lidx in list(range(len(lfilter))):
            self.lconv.append(
                keras.layers.Conv1D(filters=lfilter[lidx], kernel_size=lkernel[lidx], strides=lstride[lidx],
                                    padding="same", name=f"{name}cov{lidx}"))
            self.lbn.append(keras.layers.BatchNormalization(name=f"{name}BN{lidx}"))
            self.lactivation.append(keras.layers.LeakyReLU(name=f"{name}Activation{lidx}"))
            self.lproject.append(
                keras.layers.Conv1D(filters=lfilter[lidx], kernel_size=lkernel[lidx], strides=lstride[lidx],
                                    padding="same", name=f"{name}project{lidx}"))
        self.Pooling = keras.layers.GlobalMaxPooling1D(name=f"{name}Pooling")
    def Residule_Conv1D(self, inputs):
        layerInput=inputs
        for lidx in list(range(len(self.lconv))):
            imediate = self.lconv[lidx](layerInput)
            layerProject = self.lproject[lidx](layerInput)
            imediate = self.lbn[lidx](imediate)
            imediate = self.lactivation[lidx](imediate)
            layerInput = imediate + layerProject
        LayerOutput = self.Pooling(layerInput)
        return LayerOutput

class Repeat_Residule_Conv1D_Compoment:
    def __init__(self, sv_filters,sv_kernels, sv_strides,name):
        super().__init__()
        self.sv_sub_net=Residule_Conv1D_Compoment(sv_filters,sv_kernels, sv_strides,name)
        self.concate = tf.keras.layers.Concatenate( name=f"{name}Concate")
    def Repeat_Residule_Conv1D(self, inputs, *args, **kwargs):
        lr = []
        for idx in list(range(5)):
            lr.append(self.sv_sub_net.Residule_Conv1D(inputs[:, idx, :, :]))
        return self.concate(lr)

class INCEPTION_Compoment:
    def __init__(self, width,nb_filters,name): #nb_filters=32
        Interactivation = 'linear'
        Interstride = 1
        bottleneck_size = 32
        kernel_size = 40
        kernel_size_s = [kernel_size // (2 **i ) for i in range(width)]
        self.Lbottle= keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                              padding='same', activation=Interactivation,
                                              use_bias=False, name=f"{name}Bottlenet")
        self.Lconvs = []
        for i, ikernel_size in enumerate(kernel_size_s):
            self.Lconvs.append(keras.layers.Conv1D(filters=nb_filters, kernel_size=ikernel_size,
                                                 strides=Interstride, padding='same',
                                                 activation=Interactivation,
                                                 use_bias=False, name=f"{name}Cov{i}"))
        self.Lmax = keras.layers.MaxPool1D(pool_size=3, strides=Interstride, padding='same',
                                            name=f"{name}Maxpool")
        self.Lconvlast = keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                     padding='same', activation=Interactivation,
                                     use_bias=False, name=f"{name}ConvLast")
        self.Lconcate = keras.layers.Concatenate(axis=2, name=f"{name}Concate")
        self.LBN = keras.layers.BatchNormalization(name=f"{name}BatchNorm")
        self.LActivate = keras.layers.Activation(activation='relu')

    def Module(self, input_tensor):
        x=self.Lbottle(input_tensor)
        Lconvrs = []
        for Lconv in self.Lconvs:
            Lconvrs.append(Lconv(x))
        max_pool_1 = self.Lmax(input_tensor)
        convr_last = self.Lconvlast(max_pool_1)
        Lconvrs.append(convr_last)
        x = self.Lconcate(Lconvrs)
        x = self.LBN(x)
        x = self.LActivate(x)
        return x

class Shortcut_Compoment:
    def __init__(self, second_input_last_dimention,name):
        self.LShortcut=keras.layers.Conv1D(filters=int(second_input_last_dimention), kernel_size=1,
                                             padding='same',
                                             use_bias=False, name=f"{name}Shortcut")
        self.LBN=keras.layers.BatchNormalization(name=f"{name}BatchNorm")
        self.LAdd= keras.layers.Add()
        self.LActivation=keras.layers.Activation('relu')
    def Module(self, input_tensors):
        first_input, second_input=input_tensors
        x=self.LShortcut(first_input)
        x=self.LBN(x)
        x=self.LAdd([x,second_input ])
        x=self.LActivation(x)
        return x

class Inception_Layer:
    def __init__(self, num_group,width,nb_filters,name ):
        self.LINCEPTION_Compoment=[]
        self.LShortcut_Compoment=[]
        self.depth=num_group*3
        for d in list(range(self.depth)):
            self.LINCEPTION_Compoment.append(INCEPTION_Compoment(width,nb_filters,f"{name}LI_{d}").Module)
            if d % 3 == 2:
                self.LShortcut_Compoment.append(Shortcut_Compoment((width+1)*nb_filters,f"{name}LS_{d}").Module)
        self.Lgap= keras.layers.GlobalAveragePooling1D()

    def Layers(self, input_tensor):
        x=input_tensor
        for d,INCEPTION_Compoment in enumerate(self.LINCEPTION_Compoment):
            x=INCEPTION_Compoment(x)
            if d % 3 == 2:
                x=self.LShortcut_Compoment[d//3]([input_tensor,x])
        x=self.Lgap(x)
        return x

class Repeat_INCEPTION_Layer:
    def __init__(self,num_group,width,nb_filters, name): #num_group=3,width=3,nb_filters=32, name=name
        self.iInception_Layer=Inception_Layer(num_group,width,nb_filters, name).Layers
        self.concate = tf.keras.layers.Concatenate( name=f"{name}Concate")

    def Layer(self, inputs, *args, **kwargs):
        lr = []
        for idx in list(range(5)):
            lr.append(self.iInception_Layer(inputs[:, idx, :, :]))
        return self.concate(lr)

class net_agent_base:
    def __init__(self, lc):
        self.lc=lc
        self.nc = get_agent_nc(lc)
        self.cc=common_component(self.nc, self.lc)
        keras.backend.set_learning_phase(0)  # add by john for error solved by
        self.i_action = actionOBOS(lc.train_action_type)
        self.i_cav = globals()[lc.CLN_AV_Handler](lc)
        if self.lc.system_type == "LHPP2V3":
            self.layer_label = "OB"
            assert self.lc.P2_current_phase == "Train_Buy"
            self.choose_action=self.V3_choose_action
        else:
            assert False

    def build_predict_model(self, name):
        input_lv = keras.Input(shape=self.nc.lv_shape, dtype='float32', name="{0}_input_lv".format(name))
        input_sv = keras.Input(shape=self.nc.sv_shape, dtype='float32', name="{0}_input_sv".format(name))
        l_agent_output = self.layers_without_av([input_lv, input_sv], name)
        self.model = keras.Model(inputs=[input_lv, input_sv], outputs=l_agent_output, name=name)
        return self.model

    def layers_without_av(self, inputs, name):
        lv,sv=inputs
        lv_sv_state = getattr(self.cc, f"{self.lc.agent_method_joint_lvsv}_get_LV_SV_joint_state")([lv, sv], name)
        l_agent_output = getattr(self.cc, f"get_ap_av_{self.lc.agent_method_apsv}")([lv_sv_state], name + self.layer_label)
        return l_agent_output

    def load_weight(self, weight_fnwp):
        self.model.load_weights(weight_fnwp)

    def predict(self, state):
        lv, sv, av = state
        p, v = self.model.predict({'P_input_lv': lv, 'P_input_sv': sv})
        return p,v
    def V3_choose_action(self,state,calledby):
        assert self.lc.P2_current_phase == "Train_Buy"
        _, _, av = state
        buy_probs, buy_SVs = self.predict(state)
        if not hasattr(self, "OS_agent"):
            self.OS_agent = V2OS_4_OB_agent(self.lc,self.lc.P2_sell_system_name, self.lc.P2_sell_model_tc)
            self.i_OS_action=actionOBOS("OS")
        sel_probs, _ = self.OS_agent.predict(state)
        l_a,l_ap,l_sv = [],[],[]
        for buy_prob, sell_prob, buy_sv, av_item in zip(buy_probs, sel_probs, buy_SVs, av):
            assert len(buy_prob)==2 and len(sell_prob) == 2
            if self.i_cav.Is_Phase_Success_finished(av_item,self.i_cav.P_NB ):
                action = 2
                l_a.append(action)
                #l_ap.append(np.zeros_like(sell_prob))  # this is add zero and this record will be removed by TD_buffer before send to server for train
                l_ap.append(np.array([np.NaN, np.NaN]))
                l_sv.append(np.NaN) #l_sv.append(sell_sv[0])
            elif self.i_cav.Is_Phase_Error_Finished(av_item,self.i_cav.P_NB ):
                action = 3
                l_a.append(action)
                #l_ap.append(np.zeros_like(sell_prob))  # this is add zero and this record will be removed by TD_buffer before send to server for train
                l_ap.append(np.array([np.NaN, np.NaN]))
                l_sv.append(np.NaN) #l_sv.append(sell_sv[0])
            else:  # in NB phase
                if calledby=="Explore":
                    if np.random.choice([0, 1], p=[1-self.lc.train_random_explore_prob_buy,self.lc.train_random_explore_prob_buy]):
                        action=0
                    else:
                        action = self.i_action.I_nets_choose_action(buy_prob)
                elif calledby=="Eval":
                    action = self.i_action.I_nets_choose_action(buy_prob)
                else:
                    assert False, "Only support Explore and Eval as calledby not support {0}".format(calledby)
                l_a.append(action)
                l_ap.append(buy_prob)
                l_sv.append(buy_sv[0])
        return l_a, l_ap,l_sv

    def V3_get_AP_AT(self,state,calledby):
        assert self.lc.P2_current_phase == "Train_Buy"
        buy_probs, buy_SVs = self.predict(state)
        if not hasattr(self, "OS_agent"):
            self.OS_agent = V2OS_4_OB_agent(self.lc,self.lc.P2_sell_system_name, self.lc.P2_sell_model_tc)
            self.i_OS_action=actionOBOS("OS")
        sel_probs, _ = self.OS_agent.predict(state)
        return buy_probs,sel_probs  # This only used in CC eval ,so AP and sv information in not necessary

