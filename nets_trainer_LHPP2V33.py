from nets_trainer_base import *

def init_nets_trainer_LHPP2V33(lc_in,nc_in):
    global lc, nc
    lc=lc_in
    nc=nc_in
    init_nets_trainer_base(lc_in, nc_in)

class LHPP2V33_PPO_trainer1(base_trainer):
    def __init__(self ):
        base_trainer.__init__(self)
        self.comile_metrics=[self.M_policy_loss, self.M_value_loss,self.M_entropy,self.M_advent,self.MC_advent,
                             self.M_V,self.M_V_,self.M_r,self.M_adjr]

        self.load_jason_custom_objects={"softmax": keras.backend.softmax,"tf":tf, "concatenate":keras.backend.concatenate,"lc":lc}
        self.load_model_custom_objects={"join_loss": self.join_loss, "tf":tf,"concatenate":keras.backend.concatenate,
                                        "M_policy_loss":self.M_policy_loss,"M_value_loss":self.M_value_loss,
                                        "M_entropy":self.M_entropy,
                                        "M_advent":self.M_advent,"MC_advent":self.MC_advent,
                                        "M_V":self.M_V,"M_V_":self.M_V_,
                                        "M_r":self.M_r,"M_adjr":self.M_adjr,
                                        "lc":lc}

    def build_train_model(self, name="T"):
        Pmodel = self.build_predict_model("P")
        input_lv = keras.Input(shape=nc.lv_shape, dtype='float32', name='input_l_view')
        input_sv = keras.Input(shape=nc.sv_shape, dtype='float32', name='input_s_view')
        input_a = keras.Input(shape=(lc.train_num_action,), dtype='float32', name='input_action')
        input_r = keras.Input(shape=(1,), dtype='float32', name='input_reward')
        input_V_= keras.Input(shape=(1,), dtype='float32', name='input_V_')
        input_flag_buy_gamma=keras.Input(shape=(1,), dtype='float32', name='input_flag_buy_gamma')
        input_oldAP = keras.Input(shape=(1,), dtype='float32', name='input_oldAP')

        P, V = Pmodel([input_lv, input_sv])

        advent = keras.layers.Lambda(lambda x:x[0]-(x[1]+x[2]*x[3]))([V,input_r,input_flag_buy_gamma,input_V_])
        #V-(input_r+input_flag_buy_gamma*input_V_)

        Optimizer = self.select_optimizer(lc.Brain_optimizer, lc.Brain_leanring_rate)
        con_out = keras.layers.Concatenate(axis=1, name="train_output")([P, V, input_a, advent,input_oldAP,input_r,input_V_,input_flag_buy_gamma])
        Tmodel = keras.Model(inputs=[input_lv, input_sv, input_a, input_r, input_V_, input_flag_buy_gamma, input_oldAP],outputs=[con_out], name=name)
        Tmodel.compile(optimizer=Optimizer, loss=self.join_loss, metrics=self.comile_metrics)
        return Tmodel, Pmodel


    def MC_advent(self,y_true,y_pred):
        prob, v, a, advent,oldAP,r,v_,flag_buy_gamma = self.extract_y(y_pred)
        return tf.reduce_mean(v -  (r+ flag_buy_gamma*v_))

    def M_advent(self,y_true, y_pred):
        prob, v, a, advent,oldAP,r,v_,flag_buy_gamma= self.extract_y(y_pred)
        return tf.reduce_mean(advent)

    def M_V_(self,y_true,y_pred):
        prob, v, a, advent,oldAP,r,v_,flag_buy_gamma= self.extract_y(y_pred)
        return tf.reduce_mean(v_)


    def M_r(self,y_true,y_pred):
        prob, v, a, advent,oldAP,r,v_,flag_buy_gamma= self.extract_y(y_pred)
        return tf.reduce_mean(r)

    def M_adjr(self,y_true,y_pred):
        prob, v, a, advent,oldAP,r,v_,flag_buy_gamma= self.extract_y(y_pred)
        return tf.reduce_mean(r+ flag_buy_gamma*v_)


    def M_V(self,y_true,y_pred):
        prob, v, a, advent,oldAP,r,v_,flag_buy_gamma= self.extract_y(y_pred)
        return tf.reduce_mean(v)

    def join_loss_entropy_part(self, y_true, y_pred):
        prob, v, a, advent,oldAP,r,v_,flag_buy_gamma = self.extract_y(y_pred)
        entropy = lc.LOSS_ENTROPY * tf.reduce_sum(prob * keras.backend.log(prob + 1e-10), axis=1, keepdims=True)
        return -entropy


    def join_loss_policy_part(self,y_true,y_pred):
        prob, _, a, advent, oldAP,r,v_,flag_buy_gamma= self.extract_y(y_pred)
        prob_ratio = tf.reduce_sum(prob * a, axis=-1, keepdims=True) / (oldAP+1e-10)
        loss_policy = lc.LOSS_POLICY * K.minimum(prob_ratio * advent,
                        tf.clip_by_value(prob_ratio,clip_value_min=1 - lc.LOSS_clip, clip_value_max=1 + lc.LOSS_clip) * advent)
        return -loss_policy

    def join_loss_sv_part(self, y_true, y_pred):
        prob, v, a, advent,oldAP,r,v_,flag_buy_gamma= self.extract_y(y_pred)
        loss_value = lc.LOSS_V * tf.square(advent)
        return loss_value


    #join loss  matrixs
    def join_loss(self,y_true,y_pred):
        loss_p=self.join_loss_policy_part(y_true, y_pred)
        loss_e=self.join_loss_entropy_part(y_true, y_pred)
        loss_v=self.join_loss_sv_part(y_true, y_pred)
        return loss_p  + loss_e + loss_v

    def M_policy_loss(self,y_true, y_pred):
        loss_p=self.join_loss_policy_part(y_true, y_pred)
        return tf.reduce_mean(loss_p)


    def M_value_loss(self,y_true, y_pred):
        loss_v = self.join_loss_sv_part(y_true, y_pred)
        return tf.reduce_mean(loss_v)

    def M_entropy(self,y_true, y_pred):
        loss_e = self.join_loss_entropy_part(y_true, y_pred)
        return tf.reduce_mean(loss_e)


    def optimize_com(self, i_train_buffer, Pmodel, Tmodel):
        flag_data_available, stack_states, raw_states=self._vstack_states(i_train_buffer)
        if not flag_data_available:
            return 0, None
        s_lv, s_sv, s_av, a, r, s__lv, s__sv, s__av, done_flag, l_support_view = raw_states
        n_s_lv, n_s_sv, n_s_av, n_a, n_r, n_s__lv, n_s__sv, n_s__av=stack_states

        fake_y = np.ones((lc.batch_size, 1))
        n_old_ap = np.array([item[0, 0]["old_ap"] for item in l_support_view])

        num_record_to_train = len(n_s_lv)
        assert num_record_to_train == lc.batch_size
        _, V_ = Pmodel.predict({'P_input_lv': n_s__lv, 'P_input_sv': n_s__sv})

        l_flag_buy_gamma=[]
        for support_view_dic in l_support_view:
            if support_view_dic[0, 0]["action_return_message"] =="Success" and support_view_dic[0, 0]["action_taken"] == "Buy":
                l_flag_buy_gamma.append(0)
            else:
                l_flag_buy_gamma.append(lc.Brain_gamma**support_view_dic[0, 0]["SdisS_"])

        np_flag_buy_gamma=np.expand_dims(np.array(l_flag_buy_gamma),-1)

        loss_this_round = Tmodel.train_on_batch({'input_l_view': n_s_lv, 'input_s_view': n_s_sv,
                                                 'input_action': n_a, 'input_reward': n_r,"input_V_":V_,"input_flag_buy_gamma":np_flag_buy_gamma,
                                                 "input_oldAP":n_old_ap}, fake_y)

        if lc.flag_record_state:
            self.rv.check_need_record([Tmodel.metrics_names,loss_this_round])
            self.rv.recorder_trainer([s_lv, s_sv, s_av, a, r, s__lv, s__sv, s__av, done_flag, l_support_view])
        return num_record_to_train,loss_this_round

    def extract_y(self, y):
        prob =                  y[:, : lc.train_num_action]
        v=                      y[:, lc.train_num_action        :  lc.train_num_action +  1]
        a =                     y[:, lc.train_num_action + 1    :  2*lc.train_num_action + 1 ]
        advent =                y[:, 2 * lc.train_num_action + 1: 2 * lc.train_num_action + 2]
        oldAP =                 y[:, 2 * lc.train_num_action + 2: 2 * lc.train_num_action + 3]
        r     =                 y[:, 2 * lc.train_num_action + 3: 2 * lc.train_num_action + 4]
        v_    =                 y[:, 2 * lc.train_num_action + 4: 2 * lc.train_num_action + 5]
        flag_buy_gamma  =       y[:, 2 * lc.train_num_action + 5: 2 * lc.train_num_action + 6]
        return prob, v, a, advent,oldAP,r,v_,flag_buy_gamma
