import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from nets_agent_base import *
#from recorder import *
import pickle,shutil
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
        self.lc=lc
        self.nc=get_trainer_nc(lc)

        self.comile_metrics = [self.join_loss_policy_part, self.join_loss_sv_part, self.join_loss_entropy_part, self.M_state_value, self.M_advent]
                               #self.M_advent_low, self.M_advent_high]
        self.load_jason_custom_objects = {"softmax": keras.backend.softmax, "tf": tf, "concatenate": keras.backend.concatenate, "lc": lc}
        self.load_model_custom_objects = {"join_loss": self.join_loss, "tf": tf, "concatenate": keras.backend.concatenate,
                                          "M_policy_loss": self.join_loss_policy_part, "M_value_loss": self.join_loss_sv_part,
                                          "M_entropy_loss": self.join_loss_entropy_part, "M_state_value": self.M_state_value,
                                          "M_advent": self.M_advent,"lc": lc}

        self.i_cav = globals()[lc.CLN_AV_Handler](lc)
        if self.lc.system_type == "LHPP2V3":
            assert self.lc.P2_current_phase == "Train_Buy"
        else:
            assert False

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
        else:  # name == "Adadelta":
            # Optimizer= Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
            Optimizer = keras.optimizers.Adadelta(lr=learning_rate)
        return Optimizer

    def load_train_model(self, fnwps):
        model_AIO_fnwp, _, _ = fnwps
        Tmodel = keras.models.load_model(model_AIO_fnwp, compile=True, custom_objects=self.load_model_custom_objects)
        p = Tmodel.get_layer("Action_prob").output
        v = Tmodel.get_layer("State_value").output
        Pmodel = keras.Model(inputs=Tmodel.inputs[:2], outputs=[p, v], name="P")
        return Tmodel, Pmodel

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


    def extract_y(self, y):
        prob =          y[:, : self.lc.train_num_action]
        v=              y[:, self.lc.train_num_action     :  self.lc.train_num_action+1]
        input_a =       y[:, self.lc.train_num_action+1   :  2*self.lc.train_num_action+1]
        advent =        y[:, 2*self.lc.train_num_action+1 :  2*self.lc.train_num_action+2]
        oldAP =   y[:, 2*self.lc.train_num_action+2:   2*self.lc.train_num_action+2+1]
        return prob, v, input_a, advent,oldAP

    def join_loss_policy_part(self,y_true,y_pred):
        prob, v, input_a, advent,oldAP= self.extract_y(y_pred)
        prob_ratio = tf.reduce_sum(prob * input_a, axis=-1, keepdims=True) / (oldAP+1e-10)

        loss_policy_origin = self.lc.LOSS_POLICY * keras.backend.minimum(prob_ratio * tf.stop_gradient(advent),
                        tf.clip_by_value(prob_ratio,clip_value_min=1 - self.lc.LOSS_clip, clip_value_max=1 + self.lc.LOSS_clip) * tf.stop_gradient(advent))

        #loss_policy =tf.clip_by_value(loss_policy_origin,clip_value_min=-10, clip_value_max=10)
        assert loss_policy_origin.shape[1] == 1, loss_policy_origin.shape
        return tf.reduce_mean(-loss_policy_origin,axis=0)


    def join_loss_entropy_part(self, y_true, y_pred):
        prob, v, input_a, advent,oldAP  = self.extract_y(y_pred)
        entropy = self.lc.LOSS_ENTROPY * tf.reduce_sum(prob * keras.backend.log(prob + 1e-10), axis=1, keepdims=True)
        assert entropy.shape[1] ==1, entropy.shape
        return tf.reduce_mean(-entropy,axis=0)

    def join_loss_sv_part(self, y_true, y_pred):
        prob, v, input_a, advent,oldAP = self.extract_y(y_pred)
        loss_value = self.lc.LOSS_V * tf.square(advent)
        assert loss_value.shape[1] == 1, loss_value.shape
        return tf.reduce_mean(loss_value,axis=0)

    def join_loss(self,y_true, y_pred):
        loss_p = self.join_loss_policy_part(y_true, y_pred)
        loss_e = self.join_loss_entropy_part(y_true, y_pred)
        loss_v = self.join_loss_sv_part(y_true, y_pred)
        loss=loss_p + loss_v + loss_e
        assert loss.shape[0] == 1, loss
        if self.lc.train_total_los_clip!=0:
            loss=tf.clip_by_value(loss, clip_value_min=-self.lc.train_total_los_clip, clip_value_max=self.lc.train_total_los_clip)
        return loss

    def M_state_value(self,y_true, y_pred):
        _, v, _, _,  _= self.extract_y(y_pred)
        return tf.reduce_mean(v,axis=0)

    def M_advent(self,y_true, y_pred):
        _, _, _, advent,  _= self.extract_y(y_pred)
        return tf.reduce_mean(advent,axis=0)

    def build_train_model(self, name="T"):
        input_lv = keras.Input(shape=self.nc.lv_shape, dtype='float32', name='input_l_view')
        input_sv = keras.Input(shape=self.nc.sv_shape, dtype='float32', name='input_s_view')
        input_a = keras.Input(shape=(self.lc.train_num_action,), dtype='float32', name='input_action')
        input_oldAP = keras.Input(shape=(1,), dtype='float32', name='input_oldAP')
        input_r = keras.Input(shape=(1,), dtype='float32', name='input_reward')
        p, v = self.i_policy_agent.layers_without_av([input_lv, input_sv], "P")
        advent = keras.layers.Lambda(lambda x: x[0] - x[1], name="advantage")([input_r, v])
        Optimizer = self.select_optimizer(self.lc.Brain_optimizer, self.lc.Brain_leanring_rate)
        con_out = keras.layers.Concatenate(axis=1, name="train_output")([p, v, input_a, advent,input_oldAP])
        Tmodel = keras.Model(inputs=[input_lv, input_sv, input_a, input_r, input_oldAP],
                               outputs=[con_out], name=name)
        Pmodel =keras.Model(inputs=[input_lv, input_sv], outputs=[p, v], name="P")
        Tmodel.compile(optimizer=Optimizer, loss=self.join_loss, metrics=self.comile_metrics)
        return Tmodel, Pmodel

    def get_reward(self, n_r, v, n_s__av, l_support_view):
        l_adjR=[]
        for item_r, item_v, item_av,item_support_view in zip(n_r, v, n_s__av,l_support_view):
            assert len(item_r)==1
            if self.i_cav.check_final_record_AV(item_av):
                l_adjR.append(item_r)
            else: #todo more specific assert to check whether following case meet the LNB setting LNB==1 following should not happen
                if self.lc.system_type =="LHPP2V3" and self.lc.LHP==1 and self.lc.LNB==1:
                    assert False, "In this situation only one record for train per experiement this situation not happen"
                l_adjR.append(item_r + self.lc.Brain_gamma**item_support_view[0,0]["SdisS_"] * item_v)
        return np.array(l_adjR)


    def optimize_com(self, i_train_buffer, Pmodel, Tmodel):
        flag_data_available, stack_states, raw_states=self._vstack_states(i_train_buffer)
        if not flag_data_available:
            return 0, None,None

        #s_lv, s_sv, s_av, a, r, s__lv, s__sv, s__av, done_flag, l_support_view = raw_states
        _, _, _, _, _, _, _, _, _, l_support_view = raw_states
        n_s_lv, n_s_sv, n_s_av, n_a, n_r, n_s__lv, n_s__sv, n_s__av=stack_states
        fake_y = np.ones((self.lc.batch_size, 1))
        n_old_ap = np.array([item[0, 0]["old_ap"] for item in l_support_view])
        assert not any(n_old_ap==-1), " -1 add in a3c_worker should be removed at TD_buffer" #todo dobule check this original in v2 not in v3
        if self.lc.flag_debug_optimize_get_reward:
            df_describe = pd.DataFrame(n_old_ap)
            print (f"here2 {df_describe.describe()}")
        num_record_to_train = len(n_s_lv)
        assert num_record_to_train == self.lc.batch_size, "num_record_to_train={0} != lc.batch_size={1} n_s_lv={2}".format(num_record_to_train ,self.lc.batch_size,n_s_lv)
        _, v = Pmodel.predict({'input_l_view': n_s__lv, 'input_s_view': n_s__sv})
        rg = self.get_reward(n_r, v, n_s__av, l_support_view)
        loss_this_round = Tmodel.train_on_batch({'input_l_view': n_s_lv, 'input_s_view': n_s_sv,
                                                 'input_action': n_a, 'input_reward': rg,
                                                 "input_oldAP":n_old_ap }, fake_y)

        buy_r=rg[n_a[:, 0] == 1]
        NAction_r=rg[n_a[:, 1] == 1]
        #n_r # v # rg
        Custom_Dic={
            "R_PRvsNR_Count":(n_r>0).sum()-(n_r < 0).sum(),
            "r_mean":n_r.mean(),
            "buy_vs_NAction_Count":n_a[:,0].sum()-n_a[:, 1].sum(),
            "buy_PRvsNR_Count": len(buy_r[buy_r>0])-len(buy_r[buy_r < 0]),
            "buy_PRvsNR_Mean": (buy_r[buy_r > 0].mean() if len(buy_r[buy_r>0])>0 else 0)
                               - abs(buy_r[buy_r < 0].mean()  if len(buy_r[buy_r<0])>0 else 0),
            "NAction_Count": len(NAction_r),
        }

        if self.lc.flag_debug_save_train_input:
            if hasattr(self, "ii"):
                self.ii+=1
            else:
                self.ii=0
                self.debug_temp_dir=os.path.join("/mnt/data_disk2/debug",self.lc.RL_system_name)
                if os.path.exists(self.debug_temp_dir):
                    shutil.rmtree(self.debug_temp_dir)
                    os.mkdir(self.debug_temp_dir)
                else:
                    os.mkdir(self.debug_temp_dir)
            dnwp=os.path.join(self.debug_temp_dir, f"D{self.ii // 250}")
            if not os.path.exists(dnwp):
                os.mkdir(dnwp)
            fnwp=os.path.join(dnwp, f"{self.ii}.pickle")
            pickle.dump([raw_states,loss_this_round,Custom_Dic],open(fnwp,"wb"))

        return num_record_to_train,loss_this_round,Custom_Dic


