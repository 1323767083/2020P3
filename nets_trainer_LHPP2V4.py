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
from nets_trainer_base import *

def init_nets_trainer_LHPP2V4(lc_in,nc_in):
    global lc, nc
    lc=lc_in
    nc=nc_in

#########################################################################################################
# Base trainer and basic training methods
#########################################################################################################
class Q_base_trainer(base_trainer):
    def __init__(self):
        base_trainer.__init__(self)
        assert lc.system_type == "LHPP2V4"
        self.gammaN = lc.Brain_gamma ** lc.TDn
        self.comile_metrics=[]
        self.load_jason_custom_objects={"softmax": softmax,"tf":tf, "concatenate":concatenate,"lc":lc}
        self.load_model_custom_objects={"join_loss": self.join_loss, "tf":tf,"concatenate":concatenate,"lc":lc}

    def extract_y(self, y):
        SQstate      =          y[:, : 1]   #SQstate is selected SQstate=Qstate*input_a
        input_mask=            y[:, 1:  2]
        return SQstate, input_mask


    def join_loss(self,y_true,y_pred):
        #Qstate,input_mask=y_pred
        SQstate,  input_mask= self.extract_y(y_pred)
        loss_value = tf.reduce_mean(tf.square(y_true-SQstate)*input_mask, axis=1, keepdims=True)
        return loss_value



class LHPP2V4_Q_trainer1(Q_base_trainer):
    def __init__(self):
        Q_base_trainer.__init__(self)

    def build_train_model(self, name="T"):
        Pmodel = self.build_predict_model("P")
        input_lv = Input(shape=nc.lv_shape, dtype='float32', name='input_l_view')
        input_sv = Input(shape=nc.sv_shape, dtype='float32', name='input_s_view')
        input_a = Input(shape=(lc.train_action_num,), dtype='float32', name='input_action')
        input_mask = Input(shape=(1,), dtype='float32', name='input_mask')

        Qstate = Pmodel([input_lv, input_sv])
        Optimizer = self.select_optimizer(lc.Brain_optimizer, lc.Brain_leanring_rate)


        SQstate=Lambda(lambda x: tf.reduce_sum(x[0]*x[1], axis=-1,keep_dims=True), name="Selected_Qstate")([Qstate,input_a])
        con_out = Concatenate(axis=1, name="train_output")([SQstate, input_mask])

        Tmodel = Model(inputs=[input_lv, input_sv, input_a, input_mask], outputs=[con_out], name=name)
        Tmodel.compile(optimizer=Optimizer, loss=self.join_loss)
        return Tmodel, Pmodel

    def optimize_com(self, i_train_buffer, Pmodel, Tmodel):
        flag_data_available, stack_states, raw_states=self._vstack_states(i_train_buffer)
        if not flag_data_available:
            return 0, None
        s_lv, s_sv, s_av, a, r, s__lv, s__sv, s__av, done_flag, l_support_view = raw_states
        n_s_lv, n_s_sv, n_s_av, n_a, n_r, n_s__lv, n_s__sv, n_s__av=stack_states

        num_record_to_train = len(n_s_lv)
        assert num_record_to_train == lc.batch_size


        Qstate_ = Pmodel.predict({'P_input_lv': n_s__lv, 'P_input_sv': n_s__sv})

        l_mask=[]
        l_target=[]
        for idx,support_view_dic in enumerate(l_support_view):
            item_r=n_r[idx,:]
            item_Qstate_=Qstate_[idx,:]
            #item_av=n_s_av[idx,:]

            #print item_a
            #assert False
            if support_view_dic[0, 0]["action_return_message"] == "Success" and support_view_dic[0, 0][
                "action_taken"] == "Buy":
                l_target.append(item_r[0])
                #assert item_av[0] == 1, item_av
            else:
                with np.errstate(invalid='raise'):
                    try:
                        #l_target.append(item_r[0] + self.gammaN * item_Qstate_.max(axis=-1))
                        l_target.append(item_r[0] + lc.Brain_gamma **support_view_dic[0, 0]["SdisS_"]* item_Qstate_.max(axis=-1))
                    except Exception as e:
                        print("there", n_s__lv, n_s__sv, n_s__av,Qstate_)
                        print(item_Qstate_)
                        print(e)
                        assert False
                #assert item_av[0] == 1, item_av

            if support_view_dic[0, 0]["action_return_message"] in ["Tinpai","Exceed_limit"] and support_view_dic[0, 0]["action_taken"] == "Buy":
                l_mask.append(0)
                assert False, "while use TD_memory_LHPP2V3 these \"Tinpai\",\"Exceed_limit\" should be removed already"
            else:
                l_mask.append(1)
        n_mask=np.expand_dims(np.array(l_mask),-1)
        n_target = np.expand_dims(np.array(l_target), -1)

        loss_this_round = Tmodel.train_on_batch({'input_l_view': n_s_lv, 'input_s_view': n_s_sv,
                                                 'input_action': n_a,"input_mask":n_mask}, n_target)

        if lc.flag_record_state:
            self.rv.check_need_record([Tmodel.metrics_names,loss_this_round])
            self.rv.recorder_trainer([s_lv, s_sv, s_av, a, r, s__lv, s__sv, s__av, done_flag, l_support_view])
        return num_record_to_train,loss_this_round

