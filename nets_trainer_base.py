import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
import numpy as np
#from nets_agent_LHPP2V3 import *
#from nets_agent_LHPP2V2 import *
from nets_agent_base import *
from recorder import *
class nets_conf:
    """
    @DynamicAttrs
    """
def get_trainer_nc(lc):
    N_item_list = ["lv_shape", "sv_shape"]
    nc_item_list =[]
    nc_item_list +=N_item_list
    nc=nets_conf()
    for item_title in nc_item_list:
        assert item_title in list(lc.net_config.keys())
        setattr(nc, item_title, lc.net_config[item_title])
    nc.lv_shape = tuple(nc.lv_shape)
    nc.sv_shape = tuple(nc.sv_shape)
    return nc

class PPO_trainer:
    def __init__(self,lc):
        self.gammaN = lc.Brain_gamma ** lc.TDn
        self.i_policy_agent = net_agent_base(lc)
        self.build_predict_model=self.i_policy_agent.build_predict_model
        if lc.flag_record_state:
            self.rv = globals()[lc.CLN_record_variable](lc)
        self.lc=lc
        self.nc=get_trainer_nc(lc)

        self.comile_metrics = [self.M_policy_loss, self.M_value_loss, self.M_entropy_loss, self.M_state_value, self.M_advent,
                               self.M_advent_low, self.M_advent_high]
        self.load_jason_custom_objects = {"softmax": keras.backend.softmax, "tf": tf, "concatenate": keras.backend.concatenate, "lc": lc}
        self.load_model_custom_objects = {"join_loss": self.join_loss, "tf": tf, "concatenate": keras.backend.concatenate,
                                          "M_policy_loss": self.M_policy_loss, "M_value_loss": self.M_value_loss,
                                          "M_entropy_loss": self.M_entropy_loss, "M_state_value": self.M_state_value,
                                          "M_advent": self.M_advent, "M_advent_low": self.M_advent_low,
                                          "M_advent_high": self.M_advent_high, "lc": lc}

        self.i_cav = globals()[lc.CLN_AV_Handler](lc)
        assert self.lc.system_type in["LHPP2V2","LHPP2V3"]
        if self.lc.system_type== "LHPP2V2":
            self.av_shape = self.lc.OS_AV_shape
            self.get_av = self.i_cav.get_OS_AV
            assert self.lc.P2_current_phase == "Train_Sell"
        else:
            self.av_shape = self.lc.OB_AV_shape
            self.get_av = self.i_cav.get_OB_AV
            assert self.lc.P2_current_phase == "Train_Buy"

    def select_optimizer(self, name, learning_rate):
        assert name in ["Adam", "SGD", "Adagrad", "Adadelta"]
        if name == "Adam":
            Optimizer = keras.optimizers.Adam(lr=learning_rate)
        elif name == "SGD":
            # Optimizer = SGD(lr=0.01, nesterov=True)
            Optimizer = keras.optimizers.SGD(lr=learning_rate, nesterov=True)
        elif name == "Adagrad":
            # Optimizer = Adagrad(lr=0.01, epsilon=None, decay=0.0)
            Optimizer = keras.optimizers.Adagrad(lr=learning_rate)
        else:  # name == "AdaDelta":
            # Optimizer= Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
            Optimizer = keras.optimizers.Adadelta(lr=learning_rate)
        return Optimizer

    def load_train_model(self, fnwps):
        model_AIO_fnwp, config_fnwp, weight_fnwp = fnwps
        with open(config_fnwp, 'r') as json_file:
            loaded_model_json = json_file.read()
        Pmodel = keras.models.model_from_json(loaded_model_json, custom_objects=self.load_jason_custom_objects)
        self.load_model_custom_objects["P"]=Pmodel
        L_Tmodel = keras.models.load_model(model_AIO_fnwp, compile=True, custom_objects=self.load_model_custom_objects)
        syncronize_predict_model = L_Tmodel.get_layer("P")
        return L_Tmodel, syncronize_predict_model

    def _vstack_states(self,i_train_buffer):
        flag_got, s_lv, s_sv, s_av, a, r, s__lv, s__sv, s__av, done_flag, l_support_view = i_train_buffer.train_get(self.lc.batch_size)
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

    def get_reward(self, n_r, v, n_s__av, l_support_view):
        l_adjR=[]
        for item_r, item_v, item_av,item_support_view in zip(n_r, v, n_s__av,l_support_view):
            if self.i_cav.check_final_record_AV(item_av):
                l_adjR.append(item_r)
            else:
                l_adjR.append(item_r + self.lc.Brain_gamma**item_support_view[0,0]["SdisS_"] * item_v)
        return np.array(l_adjR)

    def extract_y(self, y):
        prob =          y[:, : self.lc.train_num_action]
        v=              y[:, self.lc.train_num_action     :  self.lc.train_num_action+1]
        input_a =       y[:, self.lc.train_num_action+1   :  2*self.lc.train_num_action+1]
        advent =        y[:, 2*self.lc.train_num_action+1 :  2*self.lc.train_num_action+2]
        oldAP =   y[:, 2*self.lc.train_num_action+2:   2*self.lc.train_num_action+2+1]
        return prob, v, input_a, advent,oldAP

    def join_loss_policy_part_old(self,y_true,y_pred):
        prob, v, input_a, advent,oldAP= self.extract_y(y_pred)
        prob_ratio = tf.reduce_sum(prob * input_a, axis=-1, keepdims=True) / (oldAP+1e-10)
        loss_policy = self.lc.LOSS_POLICY * keras.backend.minimum(prob_ratio * tf.stop_gradient(advent),
                        tf.clip_by_value(prob_ratio,clip_value_min=1 - self.lc.LOSS_clip, clip_value_max=1 + self.lc.LOSS_clip) * tf.stop_gradient(advent))
        return tf.reduce_mean(-loss_policy)
    #compare with old m new is to avoid policy losss is huge, which is very seldom, but if happen totally distroy the learning
    #normally loss is at 0.02 scale
    def join_loss_policy_part(self,y_true,y_pred):
        prob, v, input_a, advent,oldAP= self.extract_y(y_pred)
        prob_ratio = tf.reduce_sum(prob * input_a, axis=-1, keepdims=True) / (oldAP+1e-10)

        loss_policy_origin = self.lc.LOSS_POLICY * keras.backend.minimum(prob_ratio * tf.stop_gradient(advent),
                        tf.clip_by_value(prob_ratio,clip_value_min=1 - self.lc.LOSS_clip, clip_value_max=1 + self.lc.LOSS_clip) * tf.stop_gradient(advent))

        loss_policy =tf.clip_by_value(loss_policy_origin,clip_value_min=-1, clip_value_max=1)
        return tf.reduce_mean(-loss_policy)

    def join_loss_entropy_part(self, y_true, y_pred):
        prob, v, input_a, advent,oldAP  = self.extract_y(y_pred)
        entropy = self.lc.LOSS_ENTROPY * tf.reduce_sum(prob * keras.backend.log(prob + 1e-10), axis=1, keepdims=True)
        return tf.reduce_mean(-entropy)

    def join_loss_sv_part(self, y_true, y_pred):
        prob, v, input_a, advent,oldAP = self.extract_y(y_pred)
        if self.lc.LOSS_sqr_threadhold==0:  # 0 MEANS NOT TAKE SQR THREADHOLD
            loss_value = self.lc.LOSS_V * tf.square(advent)
        else:
            loss_value = self.lc.LOSS_V * keras.backend.minimum (tf.square(advent),self.lc.LOSS_sqr_threadhold)
        return tf.reduce_mean(loss_value)

    def join_loss(self,y_true, y_pred):
        loss_p = self.join_loss_policy_part(y_true, y_pred)
        loss_e = self.join_loss_entropy_part(y_true, y_pred)
        loss_v = self.join_loss_sv_part(y_true, y_pred)
        return loss_p + loss_v + loss_e

    def M_policy_loss(self,y_true, y_pred):
        return self.join_loss_policy_part(y_true, y_pred)

    def M_entropy_loss(self,y_true, y_pred):
        return self.join_loss_entropy_part(y_true, y_pred)

    def M_value_loss(self,y_true, y_pred):
        return self.join_loss_sv_part(y_true, y_pred)

    def M_state_value(self,y_true, y_pred):
        _, v, _, _,  _= self.extract_y(y_pred)
        return tf.reduce_mean(v)

    def M_advent(self,y_true, y_pred):
        _, _, _, advent,  _= self.extract_y(y_pred)
        return tf.reduce_mean(advent)

    def M_advent_low(self,y_true, y_pred):
        _, _, _, advent,  _= self.extract_y(y_pred)
        return tfp.stats.percentile(advent, 10., interpolation='lower')

    def M_advent_high(self,y_true, y_pred):
        _, _, _, advent, _= self.extract_y(y_pred)
        return tfp.stats.percentile(advent, 90., interpolation='higher')

    def build_train_model(self, name="T"):
        Pmodel = self.build_predict_model("P")
        input_lv = keras.Input(shape=self.nc.lv_shape, dtype='float32', name='input_l_view')
        input_sv = keras.Input(shape=self.nc.sv_shape, dtype='float32', name='input_s_view')
        input_av = keras.Input(shape=self.av_shape, dtype='float32', name='input_account')
        input_a = keras.Input(shape=(self.lc.train_num_action,), dtype='float32', name='input_action')
        input_oldAP = keras.Input(shape=(1,), dtype='float32', name='input_oldAP')
        input_r = keras.Input(shape=(1,), dtype='float32', name='input_reward')
        p, v = Pmodel([input_lv, input_sv, input_av])
        advent = keras.layers.Lambda(lambda x: x[0] - x[1], name="advantage")([input_r, v])
        Optimizer = self.select_optimizer(self.lc.Brain_optimizer, self.lc.Brain_leanring_rate)
        con_out = keras.layers.Concatenate(axis=1, name="train_output")([p, v, input_a, advent,input_oldAP])
        Tmodel = keras.Model(inputs=[input_lv, input_sv, input_av, input_a,input_r,input_oldAP], outputs=[con_out], name=name)
        Tmodel.compile(optimizer=Optimizer, loss=self.join_loss, metrics=self.comile_metrics)
        return Tmodel, Pmodel

    def optimize_com(self, i_train_buffer, Pmodel, Tmodel):
        flag_data_available, stack_states, raw_states=self._vstack_states(i_train_buffer)
        if not flag_data_available:
            return 0, None
        s_lv, s_sv, s_av, a, r, s__lv, s__sv, s__av, done_flag, l_support_view = raw_states
        n_s_lv, n_s_sv, n_s_av, n_a, n_r, n_s__lv, n_s__sv, n_s__av=stack_states
        fake_y = np.ones((self.lc.batch_size, 1))
        n_old_ap = np.array([item[0, 0]["old_ap"] for item in l_support_view])
        assert not any(n_old_ap==-1), " -1 add in a3c_worker should be removed at TD_buffer" #todo dobule check this original in v2 not in v3
        num_record_to_train = len(n_s_lv)
        assert num_record_to_train == self.lc.batch_size, "num_record_to_train={0} != lc.batch_size={1} n_s_lv={2}".format(num_record_to_train ,self.lc.batch_size,n_s_lv)
        _, v = Pmodel.predict({'P_input_lv': n_s__lv, 'P_input_sv': n_s__sv, 'P_input_av': self.get_av(n_s__av)})
        rg = self.get_reward(n_r, v, n_s__av,l_support_view)
        loss_this_round = Tmodel.train_on_batch({'input_l_view': n_s_lv, 'input_s_view': n_s_sv,'input_account': self.get_av(n_s_av),
                                                 'input_action': n_a, 'input_reward': rg,
                                                 "input_oldAP":n_old_ap }, fake_y)
        if self.lc.flag_record_state:
            self.rv.check_need_record([Tmodel.metrics_names,loss_this_round])
            self.rv.recorder_trainer([s_lv, s_sv, s_av, a, r, s__lv, s__sv, s__av, done_flag, l_support_view])
        return num_record_to_train,loss_this_round
