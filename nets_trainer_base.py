from keras.layers import Reshape, Input, Lambda,Flatten,Subtract, dot,TimeDistributed
from keras.layers.advanced_activations import *
from keras.models import Model, model_from_json, load_model
from keras.optimizers import Adam, SGD, Adagrad, Adadelta
from keras.layers import Concatenate, concatenate
from keras.activations import softmax
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import bitwise_ops
from nets_agent import *
from recorder import *

def init_nets_trainer_base(lc_in,nc_in):
    global lc, nc
    lc=lc_in
    nc=nc_in

#########################################################################################################
# Base trainer and basic training methods
#########################################################################################################
class base_trainer:
    def __init__(self):
        self.gammaN = lc.Brain_gamma ** lc.TDn
        #self.i_policy_agent = General_agent(class_name_policy_agent)
        self.i_policy_agent = globals()[lc.system_type]()
        self.build_predict_model=self.i_policy_agent.build_predict_model
        if lc.flag_record_state:
            self.rv = globals()[lc.CLN_record_variable](lc)

        self.comile_metrics = []
        self.load_jason_custom_objects = {"softmax": softmax, "tf": tf, "concatenate": concatenate, "lc": lc}
        self.load_model_custom_objects = {"tf": tf, "concatenate": concatenate, "lc": lc}

    def select_optimizer(self, name, learning_rate):
        assert name in ["Adam", "SGD", "Adagrad", "Adadelta"]
        if name == "Adam":
            Optimizer = Adam(lr=learning_rate)
        elif name == "SGD":
            # Optimizer = SGD(lr=0.01, nesterov=True)
            Optimizer = SGD(lr=learning_rate, nesterov=True)
        elif name == "Adagrad":
            # Optimizer = Adagrad(lr=0.01, epsilon=None, decay=0.0)
            Optimizer = Adagrad(lr=learning_rate)
        else:  # name == "AdaDelta":
            # Optimizer= Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
            Optimizer = Adadelta(lr=learning_rate)
        return Optimizer

    def load_train_model(self, fnwps):
        model_AIO_fnwp, config_fnwp, weight_fnwp = fnwps
        with open(config_fnwp, 'r') as json_file:
            loaded_model_json = json_file.read()
        Pmodel = model_from_json(loaded_model_json, custom_objects=self.load_jason_custom_objects)
        self.load_model_custom_objects["P"]=Pmodel
        L_Tmodel = load_model(model_AIO_fnwp, compile=True, custom_objects=self.load_model_custom_objects)
        syncronize_predict_model = L_Tmodel.get_layer("P")
        return L_Tmodel, syncronize_predict_model

    def _vstack_states(self,i_train_buffer):
        flag_got, s_lv, s_sv, s_av, a, r, s__lv, s__sv, s__av, done_flag, l_support_view = i_train_buffer.train_get(lc.batch_size)
        if not flag_got:
            return False, [],[]
        n_s_lv = np.vstack(s_lv)
        n_s_sv = np.vstack(s_sv)
        n_s_av = np.vstack(s_av)
        n_a = np.vstack(a)
        n_r = np.vstack(r)
        n_s__lv = np.vstack(s__lv)
        n_s__sv = np.vstack(s__sv)
        n_s__av = np.vstack(s__av)
        return True, [n_s_lv,n_s_sv,n_s_av,n_a,n_r,n_s__lv,n_s__sv,n_s__av],\
               [s_lv, s_sv, s_av, a, r, s__lv, s__sv, s__av, done_flag, l_support_view]



