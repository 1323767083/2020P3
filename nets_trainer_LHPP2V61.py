from nets_trainer_base import *

def init_nets_trainer_LHPP2V61(lc_in,nc_in):
    global lc, nc
    lc=lc_in
    nc=nc_in
    init_nets_trainer_base(lc_in, nc_in)

#LHPP2V5_PPO_trainer is same as LHPP2V3_PPO_trainer except  get_accumulate_r
class LHPP2V61_PPO_trainer(base_trainer):
    def __init__(self):
        base_trainer.__init__(self)
        self.comile_metrics=[self.M_policy_loss, self.M_value_loss,self.M_entropy_loss,self.M_state_value_TNT,
                             self.M_state_value_BNB,self.M_advent_TNT,self.M_advent_BNB]

        self.load_jason_custom_objects={"softmax": keras.backend.softmax,"tf":tf, "concatenate":keras.backend.concatenate,"lc":lc}
        self.load_model_custom_objects={"join_loss": self.join_loss, "tf":tf,"concatenate":keras.backend.concatenate,
                                        "M_policy_loss":self.M_policy_loss,"M_value_loss":self.M_value_loss,
                                        "M_entropy":self.M_entropy_loss,"M_state_value_TNT":self.M_state_value_TNT,
                                        "M_state_value_BNB": self.M_state_value_BNB,
                                        "M_advent_TNT":self.M_advent_TNT,"M_advent_BNB":self.M_advent_BNB,
                                        "lc":lc}


    def build_train_model(self, name="T"):
        Pmodel = self.build_predict_model("P")
        lv = keras.Input(shape=nc.lv_shape, dtype='float32', name='l_view')
        sv = keras.Input(shape=nc.sv_shape, dtype='float32', name='s_view')
        a_TNT = keras.Input(shape=(2,), dtype='float32', name='a_TNT')
        oldAP_TNT = keras.Input(shape=(1,), dtype='float32', name='oldAP_TNT')
        a_BNB = keras.Input(shape=(2,), dtype='float32', name='a_BNB')
        oldAP_BNB = keras.Input(shape=(1,), dtype='float32', name='oldAP_BNB')
        mask_TNT = keras.Input(shape=(1,), dtype='float32', name='mask_TNT')
        adj_r_TNT = keras.Input(shape=(1,), dtype='float32', name='reward_TNT')
        adj_r_BNB = keras.Input(shape=(1,), dtype='float32', name='reward_BNB')

        #[pTNT,pBNB], v = Pmodel([lv, sv])
        prob4, v2 = Pmodel([lv, sv])
        pTNT=keras.layers.Lambda(lambda x: x[:,:2], name=name+"prob_TNT")(prob4)
        pBNB = keras.layers.Lambda(lambda x: x[:, 2:], name=name+"prob_BNB")(prob4)
        vTNT=keras.layers.Lambda(lambda x: x[:,:1], name=name+"v_TNT")(v2)
        vBNB = keras.layers.Lambda(lambda x: x[:, 1:], name=name+"v_BNB")(v2)
        #advent = keras.layers.Lambda(lambda x: x[0] - x[1], name="advantage")([input_r, v])
        adventTNT = keras.layers.Lambda(lambda x: x[0] - x[1], name=name+"advant_TNT")([adj_r_TNT, vTNT])
        adventBNB = keras.layers.Lambda(lambda x: x[0] - x[1], name=name+"advant_BNB")([adj_r_BNB, vBNB])

        Optimizer = self.select_optimizer(lc.Brain_optimizer, lc.Brain_leanring_rate)
        con_out = keras.layers.Concatenate(axis=1, name="train_output")([pTNT,pBNB, a_TNT, a_BNB, vTNT,vBNB, adventTNT,adventBNB ,oldAP_TNT,oldAP_BNB,mask_TNT])
        Tmodel = keras.Model(inputs=[lv, sv, a_TNT, a_BNB, adj_r_TNT,adj_r_BNB, oldAP_TNT,oldAP_BNB,mask_TNT],outputs=[con_out], name=name)
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
        _, sv__ = Pmodel.predict({'P_input_lv': n_s__lv, 'P_input_sv': n_s__sv})
        sv__TNT = sv__[:, :1]
        sv__BNB = sv__[:, 1:]

        #l_adjr=[]
        l_mask=[]
        l_a_TNT = []
        l_oldAP_TNT = []
        l_adjr_TNT = []
        l_a_BNB = []
        l_oldAP_BNB = []
        l_adjr_BNB = []

        for item_r, item_a,item_sv__TNT,item_sv__BNB , support_view_dic in zip(n_r, n_a,sv__TNT,sv__BNB, l_support_view):
            if  item_a[0]==1: #buy
                #l_adjr_TNT.append(item_r[0])
                l_adjr_TNT.append(item_r[0])
                l_adjr_BNB.append(item_r[0])
                l_mask.append(1)
                l_a_TNT.append([1,0])
                l_oldAP_TNT.append(support_view_dic[0, 0]["old_ap"][0])
                l_a_BNB.append([1,0])
                l_oldAP_BNB.append(support_view_dic[0, 0]["old_ap"][1])

            elif item_a[1]==1: # no action
                l_adjr_TNT.append(item_r[0]+ lc.Brain_gamma**support_view_dic[0, 0]["SdisS_"] * item_sv__BNB[0])
                l_adjr_BNB.append(item_r[0])


                assert item_r[0]==0.0
                l_mask.append(1)
                l_a_TNT.append([1,0])
                l_oldAP_TNT.append(support_view_dic[0, 0]["old_ap"][0])
                l_a_BNB.append([0,1])
                l_oldAP_BNB.append(support_view_dic[0, 0]["old_ap"][1])

            elif item_a[2]==1: # no_trans
                l_adjr_TNT.append(item_r[0] + lc.Brain_gamma**support_view_dic[0, 0]["SdisS_"] * item_sv__TNT[0])
                l_adjr_BNB.append(-1)  # This will be marsked
                l_mask.append(0)
                l_a_TNT.append([0,1])
                l_oldAP_TNT.append(support_view_dic[0, 0]["old_ap"][0])
                l_a_BNB.append([0,0])  # will be marsked
                l_oldAP_BNB.append(support_view_dic[0, 0]["old_ap"][1])
            else:
                assert False, "{0} get_accumulate_r un expect get combination action_return_message and action_taken in this way {1} "\
                    .format(self.__class__.__name__,support_view_dic)
        #rg=np.expand_dims(np.array(l_adjr),-1)
        n_mask_TNT = np.expand_dims(np.array(l_mask), -1)

        n_a_TNT = np.array(l_a_TNT)
        n_oldAP_TNT = np.expand_dims(np.array(l_oldAP_TNT), -1)
        n_a_BNB = np.array(l_a_BNB)
        n_oldAP_BNB = np.expand_dims(np.array(l_oldAP_BNB), -1)
        n_adjr_TNT = np.expand_dims(np.array(l_adjr_TNT), -1)
        n_adjr_BNB = np.expand_dims(np.array(l_adjr_BNB), -1)

        loss_this_round = Tmodel.train_on_batch({'l_view':n_s_lv,
                                                 's_view':n_s_sv,
                                                 'a_TNT':n_a_TNT,
                                                 'oldAP_TNT':n_oldAP_TNT,
                                                 'a_BNB':n_a_BNB,
                                                 'oldAP_BNB':n_oldAP_BNB,
                                                 'mask_TNT':n_mask_TNT,
                                                 'reward_TNT':n_adjr_TNT,
                                                 'reward_BNB':n_adjr_BNB},
                                                 fake_y)
        if lc.flag_record_state:
            self.rv.check_need_record([Tmodel.metrics_names,loss_this_round])
            self.rv.recorder_trainer([s_lv, s_sv, s_av, a, r, s__lv, s__sv, s__av, done_flag, l_support_view])
        return num_record_to_train,loss_this_round

    '''
    def prepare_inputs(self,inputs):
        n_r, n_a, sv, l_support_view_dic = inputs
        l_adjr=[]
        l_mask=[]
        l_a_TNT = []
        l_oldAP_TNT = []
        l_a_BNB = []
        l_oldAP_BNB = []

        for item_r, item_a,item_train_sv, support_view_dic in zip(n_r, n_a,sv, l_support_view_dic):
            if  item_a[0]==1: #buy
                l_adjr.append(item_r[0])
                l_mask.append(1)
                l_a_TNT.append([1,0])
                l_oldAP_TNT.append(support_view_dic[0, 0]["old_ap"][0])
                l_a_BNB.append([1,0])
                l_oldAP_BNB.append(support_view_dic[0, 0]["old_ap"][1])

            elif item_a[1]==1: # no action
                l_adjr.append(0.0)
                l_mask.append(1)
                l_a_TNT.append([1,0])
                l_oldAP_TNT.append(support_view_dic[0, 0]["old_ap"][0])
                l_a_BNB.append([0,1])
                l_oldAP_BNB.append(support_view_dic[0, 0]["old_ap"][1])

            elif item_a[2]==1: # no_trans
                l_adjr.append(item_r[0] + lc.Brain_gamma**support_view_dic[0, 0]["SdisS_"] * item_train_sv[0])  # this is because the bigger r is the smaller prob of No_trans
                l_mask.append(0)
                l_a_TNT.append([0,1])
                l_oldAP_TNT.append(support_view_dic[0, 0]["old_ap"][0])
                l_a_BNB.append([0,0])  # will be marsked
                l_oldAP_BNB.append(support_view_dic[0, 0]["old_ap"][1])
            else:
                assert False, "{0} get_accumulate_r un expect get combination action_return_message and action_taken in this way {1} "\
                    .format(self.__class__.__name__,support_view_dic)
        rg=np.expand_dims(np.array(l_adjr),-1)
        rm = np.expand_dims(np.array(l_mask), -1)

        n_a_TNT = np.array(l_a_TNT)
        n_oldAP_TNT = np.expand_dims(np.array(l_oldAP_TNT), -1)
        n_a_BNB = np.array(l_a_BNB)
        n_oldAP_BNB = np.expand_dims(np.array(l_oldAP_BNB), -1)

        return rg,rm,n_a_TNT,n_oldAP_TNT,n_a_BNB,n_oldAP_BNB
    '''

    def extract_y(self, y):
        anum = 2 #lc.train_num_action
        pTNT =      y[:,     :             anum]
        pBNB =      y[:, anum:               2 * anum]
        a_TNT=      y[:, 2 * anum:           3 * anum]
        a_BNB=      y[:, 3 * anum:           4 * anum]
        vTNT=       y[:, 4 * anum:           4 * anum + 1]
        vBNB=       y[:, 4 * anum + 1:       4 * anum + 2]
        adventTNT = y[:, 4 * anum + 2:       4 * anum + 3]
        adventBNB = y[:, 4 * anum + 3:       4 * anum + 4]
        oldAP_TNT = y[:, 4 * anum + 4:       4 * anum + 5]
        oldAP_BNB = y[:, 4 * anum + 5:       4 * anum + 6]
        mask_TNT =  y[:, 4 * anum + 6:       4 * anum + 7]
        #return pTNT,pBNB, a_TNT, a_BNB, advent,v,advent,oldAP_TNT,oldAP_BNB,mask_TNT

        return pTNT, pBNB, a_TNT, a_BNB, vTNT, vBNB, adventTNT, adventBNB, oldAP_TNT, oldAP_BNB, mask_TNT
    def join_loss_policy_part(self,y_true,y_pred):
        pTNT,pBNB, a_TNT, a_BNB, vTNT,vBNB, adventTNT,adventBNB ,oldAP_TNT,oldAP_BNB,mask_TNT= self.extract_y(y_pred)
        prob_ratioTNT=tf.reduce_sum(pTNT * a_TNT, axis=-1, keepdims=True) / (oldAP_TNT+1e-10)
        loss_policy_TNT_origin = lc.LOSS_POLICY * keras.backend.minimum(prob_ratioTNT * tf.stop_gradient(adventTNT),
                        tf.clip_by_value(prob_ratioTNT,clip_value_min=1 - lc.LOSS_clip, clip_value_max=1 + lc.LOSS_clip) * tf.stop_gradient(adventTNT))

        loss_policy_TNT = tf.clip_by_value(loss_policy_TNT_origin, clip_value_min=-1, clip_value_max=1)


        prob_ratioBNB=tf.reduce_sum(pBNB * a_BNB, axis=-1, keepdims=True) / (oldAP_BNB+1e-10)
        loss_policy_BNB_origin = lc.LOSS_POLICY * keras.backend.minimum(prob_ratioBNB * tf.stop_gradient(adventBNB),
                        tf.clip_by_value(prob_ratioBNB,clip_value_min=1 - lc.LOSS_clip, clip_value_max=1 + lc.LOSS_clip) * tf.stop_gradient(adventBNB))
        loss_policy_BNB_origin=loss_policy_BNB_origin*mask_TNT
        loss_policy_BNB = tf.clip_by_value(loss_policy_BNB_origin, clip_value_min=-1, clip_value_max=1)

        return tf.reduce_mean((-loss_policy_TNT-loss_policy_BNB)/2)

    def join_loss_entropy_part(self, y_true, y_pred):
        pTNT,pBNB, a_TNT, a_BNB, vTNT,vBNB, adventTNT,adventBNB ,oldAP_TNT,oldAP_BNB,mask_TNT  = self.extract_y(y_pred)
        entropy_TNT = lc.LOSS_ENTROPY * tf.reduce_sum(pTNT * keras.backend.log(pTNT + 1e-10), axis=1, keepdims=True)
        entropy_BNB = lc.LOSS_ENTROPY * tf.reduce_sum(pBNB * keras.backend.log(pBNB + 1e-10), axis=1, keepdims=True)
        return tf.reduce_mean((-entropy_TNT-entropy_BNB)/2)

    def join_loss_sv_part(self, y_true, y_pred):
        pTNT,pBNB, a_TNT, a_BNB, vTNT,vBNB, adventTNT,adventBNB ,oldAP_TNT,oldAP_BNB,mask_TNT = self.extract_y(y_pred)
        if lc.LOSS_sqr_threadhold==0:  # 0 MEANS NOT TAKE SQR THREADHOLD
            loss_value_TNT = lc.LOSS_V * tf.square(adventTNT)
            loss_value_BNB = lc.LOSS_V * tf.square(adventBNB)
        else:
            loss_value_TNT = lc.LOSS_V * keras.backend.minimum(tf.square(adventTNT), lc.LOSS_sqr_threadhold)
            loss_value_BNB = lc.LOSS_V * keras.backend.minimum(tf.square(adventBNB), lc.LOSS_sqr_threadhold)
        return tf.reduce_mean((loss_value_TNT+loss_value_BNB)/2)


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

    def M_state_value_TNT(self,y_true, y_pred):
        pTNT,pBNB, a_TNT, a_BNB, vTNT,vBNB, adventTNT,adventBNB ,oldAP_TNT,oldAP_BNB,mask_TNT = self.extract_y(y_pred)
        return tf.reduce_mean(vTNT)

    def M_state_value_BNB(self,y_true, y_pred):
        pTNT,pBNB, a_TNT, a_BNB, vTNT,vBNB, adventTNT,adventBNB ,oldAP_TNT,oldAP_BNB,mask_TNT = self.extract_y(y_pred)
        return tf.reduce_mean(vBNB)

    def M_advent_TNT(self,y_true, y_pred):
        pTNT,pBNB, a_TNT, a_BNB, vTNT,vBNB, adventTNT,adventBNB ,oldAP_TNT,oldAP_BNB,mask_TNT = self.extract_y(y_pred)
        return tf.reduce_mean(adventTNT)

    def M_advent_BNB(self,y_true, y_pred):
        pTNT,pBNB, a_TNT, a_BNB, vTNT,vBNB, adventTNT,adventBNB ,oldAP_TNT,oldAP_BNB,mask_TNT = self.extract_y(y_pred)
        return tf.reduce_mean(adventBNB)

