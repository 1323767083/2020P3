from nets_trainer_base import *

def init_nets_trainer_LHPP2V2(lc_in,nc_in):
    global lc, nc
    lc=lc_in
    nc=nc_in
    init_nets_trainer_base(lc_in, nc_in)

class LHPP2V2_trainer_base(base_trainer):
    def __init__(self):
        base_trainer.__init__(self)

    #fake
    def extract_y(self, y):
        prob, v, input_a, advent, oldAP=0,0,0,0,0
        return prob, v, input_a, advent, oldAP

    # fake
    def join_loss_policy_part(self,y_true,y_pred):
        assert False
        loss_policy=0
        return -loss_policy

    def join_loss_entropy_part(self, y_true, y_pred):
        prob, v, input_a, advent,oldAP  = self.extract_y(y_pred)
        entropy = lc.LOSS_ENTROPY * tf.reduce_sum(prob * keras.backend.log(prob + 1e-10), axis=1, keepdims=True)
        return tf.reduce_mean(-entropy)

    def join_loss_sv_part(self, y_true, y_pred):
        prob, v, input_a, advent,oldAP = self.extract_y(y_pred)
        if lc.LOSS_sqr_threadhold==0:  # 0 MEANS NOT TAKE SQR THREADHOLD
            loss_value = lc.LOSS_V * tf.square(advent)
        else:
            loss_value = lc.LOSS_V * keras.backend.minimum (tf.square(advent),lc.LOSS_sqr_threadhold)
        return tf.reduce_mean(loss_value)

    def join_loss(self,y_true, y_pred):
        loss_p = self.join_loss_policy_part(y_true, y_pred)
        loss_e = self.join_loss_entropy_part(y_true, y_pred)
        loss_v = self.join_loss_sv_part(y_true, y_pred)
        return loss_p + loss_v + loss_e

    def M_policy_loss(self, y_true, y_pred):
        loss_p = self.join_loss_policy_part(y_true, y_pred)
        return tf.reduce_mean(loss_p)

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
        #return tf.contrib.distributions.percentile(advent, 10., interpolation='lower')
        return tfp.stats.percentile(advent, 10., interpolation='lower')

    def M_advent_high(self,y_true, y_pred):
        _, _, _, advent, _= self.extract_y(y_pred)
        #return tf.contrib.distributions.percentile(advent, 90., interpolation='higher')
        return tfp.stats.percentile(advent, 90., interpolation='higher')

    #accumulate_reward method
    def OS_accumulate_reward(self,n_r,v,l_support_view):
        return np.array([item_r + lc.Brain_gamma**item_support_view[0,0]["SdisS_"] * item_v   for item_r, item_v, item_support_view in zip(n_r, v, l_support_view)])

    def OS_ForceSell_accumulate_reward(self,n_r,v,l_support_view):
        l_adjR=[]
        for item_r, item_v, item_support_view in zip(n_r, v, l_support_view):
            if item_support_view[0, 0]["flag_force_sell"]:
                if item_support_view[0, 0]["action_taken"]=="Sell" and item_support_view[0, 0]["action_return_message"]=="Success":
                    l_adjR.append(item_r)
                else:
                    l_adjR.append(0)
                    assert False, "These records should be already removed from TD_memory_LHPP2V2"
            else:
                if item_support_view[0, 0]["action_taken"]=="Sell" and item_support_view[0, 0]["action_return_message"]=="Success":
                    l_adjR.append(item_r)
                else:
                    l_adjR.append(item_r + lc.Brain_gamma**item_support_view[0,0]["SdisS_"] * item_v)
        return np.array(l_adjR)

    def OS_s_0_reward(self,n_r,v,l_support_view):
        l_adjR=[]
        for item_r, item_v, item_support_view in zip(n_r, v, l_support_view):
            if item_support_view[0, 0]["flag_force_sell"]:
                if item_support_view[0, 0]["action_taken"]=="Sell" and item_support_view[0, 0]["action_return_message"]=="Success":
                    l_adjR.append(item_r)
                else:
                    l_adjR.append(0)
                    assert False, "These records should be already removed from TD_memory_LHPP2V2"
            else:
                if item_support_view[0, 0]["action_taken"]=="Sell" and item_support_view[0, 0]["action_return_message"]=="Success":
                    l_adjR.append(item_r)
                else:
                    l_adjR.append(item_r)
        return np.array(l_adjR)


class LHPP2V2_PG_trainer(LHPP2V2_trainer_base):
    def __init__(self):
        LHPP2V2_trainer_base.__init__(self)
        self.ac_reward_fun=getattr(self,lc.specific_param.accumulate_reward_method)
        self.comile_metrics=[self.M_policy_loss, self.M_value_loss,self.M_entropy_loss,self.M_state_value,self.M_advent,
                             self.M_advent_low,self.M_advent_high]

        self.load_jason_custom_objects={"softmax": keras.backend.softmax,"tf":tf, "concatenate":keras.backend.concatenate,"lc":lc}
        self.load_model_custom_objects={"join_loss": self.join_loss, "tf":tf,"concatenate":keras.backend.concatenate,
                                        "M_policy_loss":self.M_policy_loss,"M_value_loss":self.M_value_loss,
                                        "M_entropy":self.M_entropy_loss,"M_state_value":self.M_state_value,
                                        "M_advent":self.M_advent,"M_advent_low":self.M_advent_low,
                                        "M_advent_high":self.M_advent_high,"lc":lc}

    def build_train_model(self, name="T"):

        Pmodel = self.build_predict_model("P")
        input_lv = keras.Input(shape=nc.lv_shape, dtype='float32', name='input_l_view')
        input_sv = keras.Input(shape=nc.sv_shape, dtype='float32', name='input_s_view')

        input_av = keras.Input(shape=lc.specific_param.OS_AV_shape, dtype='float32', name='input_account')
        input_a = keras.Input(shape=(lc.train_num_action,), dtype='float32', name='input_action')

        input_r = keras.Input(shape=(1,), dtype='float32', name='input_reward')
        p, v = Pmodel([input_lv, input_sv, input_av])
        advent = keras.layers.Lambda(lambda x: x[0] - x[1], name="advantage")([input_r, v])
        Optimizer = self.select_optimizer(lc.Brain_optimizer, lc.Brain_leanring_rate)
        con_out = keras.layers.Concatenate(axis=1, name="train_output")([p, v, input_a, advent])
        Tmodel = keras.Model(inputs=[input_lv, input_sv, input_av, input_a, input_r], outputs=[con_out], name=name)

        Tmodel.compile(optimizer=Optimizer, loss=self.join_loss, metrics=self.comile_metrics)
        return Tmodel, Pmodel

    def optimize_com(self, i_train_buffer, Pmodel, Tmodel):
        flag_data_available, stack_states, raw_states=self._vstack_states(i_train_buffer)
        if not flag_data_available:
            return 0, None
        s_lv, s_sv, s_av, a, r, s__lv, s__sv, s__av, done_flag, l_support_view = raw_states
        n_s_lv, n_s_sv, n_s_av, n_a, n_r, n_s__lv, n_s__sv, n_s__av=stack_states

        fake_y = np.ones((lc.batch_size, 1))

        num_record_to_train = len(n_s_lv)
        assert num_record_to_train == lc.batch_size
        _, v = Pmodel.predict({'P_input_lv': n_s__lv, 'P_input_sv': n_s__sv, 'P_input_av': n_s__av})

        rg=self.ac_reward_fun(n_r,v,l_support_view)
        loss_this_round = Tmodel.train_on_batch({'input_l_view': n_s_lv, 'input_s_view': n_s_sv,'input_account': n_s_av,
                                                 'input_action': n_a, 'input_reward': rg }, fake_y)
        if lc.flag_record_state:
            self.rv.check_need_record([Tmodel.metrics_names,loss_this_round])
            self.rv.recorder_trainer([s_lv, s_sv, s_av, a, r, s__lv, s__sv, s__av, done_flag, l_support_view])
        return num_record_to_train,loss_this_round

    def extract_y(self, y):
        prob =          y[:, : lc.train_num_action]
        v=              y[:, lc.train_num_action     :  lc.train_num_action+1]
        input_a =       y[:, lc.train_num_action+1   :  2*lc.train_num_action+1]
        advent =        y[:, 2*lc.train_num_action+1 :  2*lc.train_num_action+2]
        return prob, v, input_a, advent,0  # oldAP=0 should not a scaler 0 but a vectore 0

    def join_loss_policy_part(self, y_true, y_pred):
        prob, _, input_a, advent, _, = self.extract_y(y_pred)
        log_prob = keras.backend.log(tf.reduce_sum(prob * input_a, axis=1, keepdims=True) + 1e-10)
        loss_policy = lc.LOSS_POLICY * log_prob * tf.stop_gradient(advent)
        return -loss_policy

class LHPP2V2_PPO_trainer(LHPP2V2_trainer_base):
    def __init__(self):
        LHPP2V2_trainer_base.__init__(self)
        assert lc.P2_current_phase == "Train_Sell"
        self.join_loss_policy_part=self.join_loss_policy_part_new
        self.ac_reward_fun=getattr(self,lc.specific_param.accumulate_reward_method)
        self.comile_metrics = [self.M_policy_loss, self.M_value_loss, self.M_entropy_loss, self.M_state_value, self.M_advent,
                               self.M_advent_low, self.M_advent_high]
        self.load_jason_custom_objects = {"softmax": keras.backend.softmax, "tf": tf, "concatenate": keras.backend.concatenate, "lc": lc}
        self.load_model_custom_objects = {"join_loss": self.join_loss, "tf": tf, "concatenate": keras.backend.concatenate,
                                          "M_policy_loss": self.M_policy_loss, "M_value_loss": self.M_value_loss,
                                          "M_entropy": self.M_entropy_loss, "M_state_value": self.M_state_value,
                                          "M_advent": self.M_advent, "M_advent_low": self.M_advent_low,
                                          "M_advent_high": self.M_advent_high, "lc": lc}

    def build_train_model(self, name="T"):

        Pmodel = self.build_predict_model("P")
        input_lv = keras.Input(shape=nc.lv_shape, dtype='float32', name='input_l_view')
        input_sv = keras.Input(shape=nc.sv_shape, dtype='float32', name='input_s_view')
        input_av = keras.Input(shape=lc.specific_param.OS_AV_shape, dtype='float32', name='input_account')
        input_a = keras.Input(shape=(lc.train_num_action,), dtype='float32', name='input_action')
        input_oldAP = keras.Input(shape=(1,), dtype='float32', name='input_oldAP')
        input_r = keras.Input(shape=(1,), dtype='float32', name='input_reward')
        p, v = Pmodel([input_lv, input_sv, input_av])
        advent = keras.layers.Lambda(lambda x: x[0] - x[1], name="advantage")([input_r, v])
        Optimizer = self.select_optimizer(lc.Brain_optimizer, lc.Brain_leanring_rate)

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

        fake_y = np.ones((lc.batch_size, 1))
        n_old_ap = np.array([item[0, 0]["old_ap"] for item in l_support_view])
        assert not any(n_old_ap==-1), " -1 add in a3c_worker should be removed at TD_buffer"
        num_record_to_train = len(n_s_lv)
        assert num_record_to_train == lc.batch_size
        _, v = Pmodel.predict({'P_input_lv': n_s__lv, 'P_input_sv': n_s__sv, 'P_input_av': n_s__av})
        rg = self.ac_reward_fun(n_r, v, l_support_view)
        loss_this_round = Tmodel.train_on_batch({'input_l_view': n_s_lv, 'input_s_view': n_s_sv,'input_account': n_s_av,
                                                 'input_action': n_a, 'input_reward': rg,
                                                 "input_oldAP":n_old_ap }, fake_y)
        if lc.flag_record_state:
            self.rv.check_need_record([Tmodel.metrics_names,loss_this_round])
            self.rv.recorder_trainer([s_lv, s_sv, s_av, a, r, s__lv, s__sv, s__av, done_flag, l_support_view])
        return num_record_to_train,loss_this_round

    def extract_y(self, y):
        prob =          y[:, : lc.train_num_action]
        v=              y[:, lc.train_num_action     :  lc.train_num_action+1]
        input_a =       y[:, lc.train_num_action+1   :  2*lc.train_num_action+1]
        advent =        y[:, 2*lc.train_num_action+1 :  2*lc.train_num_action+2]
        oldAP =   y[:, 2*lc.train_num_action+2:   2*lc.train_num_action+2+1]
        return prob, v, input_a, advent,oldAP

    def join_loss_policy_part_old(self,y_true,y_pred):
        prob, v, input_a, advent,oldAP= self.extract_y(y_pred)
        prob_ratio = tf.reduce_sum(prob * input_a, axis=-1, keepdims=True) / (oldAP+1e-10)
        #loss_policy = lc.LOSS_POLICY * K.minimum(prob_ratio * advent,
        #                tf.clip_by_value(prob_ratio,clip_value_min=1 - lc.LOSS_clip, clip_value_max=1 + lc.LOSS_clip) * advent)

        loss_policy = lc.LOSS_POLICY * keras.backend.minimum(prob_ratio * tf.stop_gradient(advent),
                        tf.clip_by_value(prob_ratio,clip_value_min=1 - lc.LOSS_clip, clip_value_max=1 + lc.LOSS_clip) * tf.stop_gradient(advent))

        return tf.reduce_mean(-loss_policy)


    def join_loss_policy_part_new(self,y_true,y_pred):
        prob, v, input_a, advent,oldAP= self.extract_y(y_pred)
        prob_ratio = tf.reduce_sum(prob * input_a, axis=-1, keepdims=True) / (oldAP+1e-10)
        #loss_policy = lc.LOSS_POLICY * K.minimum(prob_ratio * advent,
        #                tf.clip_by_value(prob_ratio,clip_value_min=1 - lc.LOSS_clip, clip_value_max=1 + lc.LOSS_clip) * advent)

        loss_policy_origin = lc.LOSS_POLICY * keras.backend.minimum(prob_ratio * tf.stop_gradient(advent),
                        tf.clip_by_value(prob_ratio,clip_value_min=1 - lc.LOSS_clip, clip_value_max=1 + lc.LOSS_clip) * tf.stop_gradient(advent))

        loss_policy =tf.clip_by_value(loss_policy_origin,clip_value_min=-1, clip_value_max=1)
        return tf.reduce_mean(-loss_policy)


'''
 return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - loss_clipping,
                                                           max_value=1 + loss_clipping) * advantage) + entropy_loss * (
prob * K.log(prob + 1e-10)))
'''