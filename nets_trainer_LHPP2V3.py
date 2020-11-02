from nets_trainer_base import *

def init_nets_trainer_LHPP2V3(lc_in,nc_in):
    global lc, nc
    lc=lc_in
    nc=nc_in
    init_nets_trainer_base(lc_in, nc_in)


class LHPP2V3_PPO_trainer(base_trainer):
    def __init__(self):
        base_trainer.__init__(self)
        self.comile_metrics=[self.M_policy_loss, self.M_value_loss,self.M_entropy_loss,self.M_state_value,self.M_advent,
                             self.M_advent_low,self.M_advent_high]

        self.load_jason_custom_objects={"softmax": keras.backend.softmax,"tf":tf, "concatenate":keras.backend.concatenate,"lc":lc}
        self.load_model_custom_objects={"join_loss": self.join_loss, "tf":tf,"concatenate":keras.backend.concatenate,
                                        "M_policy_loss":self.M_policy_loss,"M_value_loss":self.M_value_loss,
                                        "M_entropy":self.M_entropy_loss,"M_state_value":self.M_state_value,
                                        "M_advent":self.M_advent,"M_advent_low":self.M_advent_low,
                                        "M_advent_high":self.M_advent_high,"lc":lc}

        self.ac_reward_fun = getattr(self, lc.Optimize_accumulate_reward_method)
        i_cav=globals()[lc.CLN_AV_state]()
        self.get_OB_AV = i_cav.get_OB_av


    def build_train_model(self, name="T"):
        Pmodel = self.build_predict_model("P")
        lv = keras.Input(shape=nc.lv_shape, dtype='float32', name='input_l_view')
        sv = keras.Input(shape=nc.sv_shape, dtype='float32', name='input_s_view')
        av = keras.Input(shape=lc.OB_AV_shape, dtype='float32', name='input_av_view')
        input_a = keras.Input(shape=(lc.train_num_action,), dtype='float32', name='input_action')
        input_r = keras.Input(shape=(1,), dtype='float32', name='input_reward')
        input_oldAP = keras.Input(shape=(1,), dtype='float32', name='input_oldAP')

        p, v = Pmodel([lv, sv,av])
        advent = keras.layers.Lambda(lambda x: x[0] - x[1], name="advantage")([input_r, v])
        Optimizer = self.select_optimizer(lc.Brain_optimizer, lc.Brain_leanring_rate)
        con_out = keras.layers.Concatenate(axis=1, name="train_output")([p, v, input_a, advent,input_oldAP])
        Tmodel = keras.Model(inputs=[lv, sv, av,input_a, input_r, input_oldAP],outputs=[con_out], name=name)
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
        #assert not any(n_old_ap==-1.0)," -1 add in a3c_worker should be removed at TD_buffer" # sanity check -1 which added in A3C_worker have been removed in TD_buffer
        #float can not use ==


        num_record_to_train = len(n_s_lv)
        assert num_record_to_train == lc.batch_size
        _, train_sv = Pmodel.predict({'P_input_lv': n_s__lv, 'P_input_sv': n_s__sv, "P_input_av": self.get_OB_AV(n_s__av)})


        #rg=self.get_accumulate_r([n_r, n_a,train_sv, n_s_av,l_support_view])
        rg=self.ac_reward_fun([n_r, n_a,train_sv, n_s_av,l_support_view])
        loss_this_round = Tmodel.train_on_batch({'input_l_view': n_s_lv, 'input_s_view': n_s_sv,"input_av_view":self.get_OB_AV(n_s_av),
                                                 'input_action': n_a, 'input_reward': rg,
                                                 "input_oldAP":n_old_ap }, fake_y)
        if lc.flag_record_state:
            self.rv.check_need_record([Tmodel.metrics_names,loss_this_round])
            self.rv.recorder_trainer([s_lv, s_sv, s_av, a, r, s__lv, s__sv, s__av, done_flag, l_support_view])
        return num_record_to_train,loss_this_round


    def get_accumulate_r_old(self,inputs):
        n_r, n_a,train_sv, n_s_av,l_support_view = inputs
        l_adjr=[]
        for item_a,item_r, item_train_sv, support_view_dic, av_item in zip(n_a,n_r, train_sv, l_support_view,n_s_av):
            if item_a[0]==1:  # buy
                l_adjr.append(item_r[0])
            elif item_a[1]==1: # no_action
                l_adjr.append(item_r[0] + lc.Brain_gamma ** support_view_dic[0, 0]["SdisS_"] * item_train_sv[0])
            else:
                assert False, "action {0} support_view {1}".format(item_a, support_view_dic)
        rg=np.expand_dims(np.array(l_adjr),-1)
        return rg

    def optimize_get_accumulate_r(self,inputs):
        n_r, n_a,train_sv, n_s_av,l_support_view = inputs
        l_adjr=[]
        for item_a,item_r, item_train_sv, support_view_dic, av_item in zip(n_a,n_r, train_sv, l_support_view,n_s_av):
            if item_a[0]==1:  # buy
                l_adjr.append(item_r[0])
            elif item_a[1]==1: # no_action
                l_adjr.append(item_r[0] + lc.Brain_gamma ** support_view_dic[0, 0]["SdisS_"] * item_train_sv[0])
            else:
                assert False, "action {0} support_view {1}".format(item_a, support_view_dic)
        rg=np.expand_dims(np.array(l_adjr),-1)
        return rg


    def extract_y(self, y):
        prob =          y[:, : lc.train_num_action]
        v=              y[:, lc.train_num_action     :  lc.train_num_action+1]
        input_a =       y[:, lc.train_num_action+1   :  2*lc.train_num_action+1]
        advent =        y[:, 2*lc.train_num_action+1 :  2*lc.train_num_action+2]
        input_oldAP =   y[:, 2*lc.train_num_action+2:   2*lc.train_num_action+3]
        return prob, v, input_a, advent,input_oldAP


    def join_loss_policy_part(self,y_true,y_pred):
        prob, v, input_a, advent,oldAP= self.extract_y(y_pred)
        prob_ratio = tf.reduce_sum(prob * input_a, axis=-1, keepdims=True) / (oldAP+1e-10)
        loss_policy_origin = lc.LOSS_POLICY * keras.backend.minimum (prob_ratio * tf.stop_gradient(advent),
                        tf.clip_by_value(prob_ratio,clip_value_min=1 - lc.LOSS_clip, clip_value_max=1 + lc.LOSS_clip) * tf.stop_gradient(advent))

        loss_policy =tf.clip_by_value(loss_policy_origin,clip_value_min=-1, clip_value_max=1)
        return tf.reduce_mean(-loss_policy)

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


    #join loss  matrixs
    def join_loss(self,y_true,y_pred):
        loss_p=self.join_loss_policy_part(y_true, y_pred)
        loss_e=self.join_loss_entropy_part(y_true, y_pred)
        loss_v=self.join_loss_sv_part(y_true, y_pred)
        return loss_p + loss_e + loss_v

    def M_policy_loss(self,y_true, y_pred):
        return self.join_loss_policy_part(y_true, y_pred)

    def M_value_loss(self,y_true, y_pred):
        return self.join_loss_sv_part(y_true, y_pred)

    def M_entropy_loss(self,y_true, y_pred):
        return self.join_loss_entropy_part(y_true, y_pred)

    def M_state_value(self,y_true, y_pred):
        _, v, _, _, _= self.extract_y(y_pred)
        return tf.reduce_mean(v)

    def M_advent(self,y_true, y_pred):
        _, _, _, advent, _= self.extract_y(y_pred)
        return tf.reduce_mean(advent)

    def M_advent_low(self,y_true, y_pred):
        _, _, _, advent, _= self.extract_y(y_pred)
        advent10=tfp.stats.percentile(advent, 10., interpolation='lower')

        return advent10

    def M_advent_high(self,y_true, y_pred):
        _, _, _, advent, _= self.extract_y(y_pred)
        advent90=tfp.stats.percentile(advent, 90., interpolation='higher')
        return advent90

