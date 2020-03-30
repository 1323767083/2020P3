#from tensorflow import keras
import tensorflow as tf
from tensorflow import keras
from File_comm import check_model_save_finish_write
from Buffer_comm import brain_buffer, brain_buffer_reuse
from nets_trainer import *
from action_comm import actionOBOS
def init_gc(lgc):
    global Ctrainer,Cagent, nc,lc
    lc=lgc
    init_agent_config(lc)
    init_trainer_config(lc)
    Ctrainer = globals()[lc.CLN_trainer]

def init_virtual_GPU(memory_limit):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
    except RuntimeError as e:
        assert False, e
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    return logical_gpus[0]

class Brain:
    def __init__(self, GPU_per_program):
        assert GPU_per_program != 0.0," Only Support GPU"
        #keras.backend.set_learning_phase(1)  # add by john for error solved by
        '''
        if GPU_per_program != 0.0:
            self.start_GPU_env(GPU_per_program)
        else:
            self.start_CPU_env()
        '''
    '''
    def start_GPU_env(self, GPU_per_program):
        tf.reset_default_graph()
        self.default_graph = tf.get_default_graph()
        config = tf.ConfigProto(
            log_device_placement=False
        )
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = GPU_per_program
        self.session = tf.Session(config=config, graph=self.default_graph)
        K.set_session(self.session)
        K.set_learning_phase(1)  # add by john for error solved by
    '''

    '''
    def start_CPU_env(self):
        tf.reset_default_graph()
        self.default_graph = tf.get_default_graph()
        config = tf.ConfigProto(
            device_count={'CPU': 1, 'GPU': 0},
            allow_soft_placement=True,
            log_device_placement=False
        )
        config.gpu_options.visible_device_list = ""
        self.session = tf.Session(config=config, graph=self.default_graph)
        K.set_session(self.session)
        K.set_learning_phase(1)  # add by john for error solved by
    '''

class Train_Brain(Brain):
    def __init__(self, GPU_per_program, load_fnwps,train_count_init):
        Brain.__init__(self, GPU_per_program)
        keras.backend.set_learning_phase(1)  # add by john for error solved by
        self.mc=Ctrainer()
        #self.tb = train_buffer(lc.Buffer_nb_Features)
        self.tb = globals()[lc.CLN_brain_buffer](lc.Buffer_nb_Features)
        self.train_push_many = self.tb.train_push_many
        self.get_buffer_size = self.tb.get_buffer_size
        if len(load_fnwps) == 0:
            self.Tmodel, self.Pmodel = self.build_model()
        else:
            assert len(load_fnwps) == 3
            self.Tmodel, self.Pmodel = self.load_model(load_fnwps)
        self.i_wait = check_model_save_finish_write(time_out=600)

        #self.tensorboard = keras.callbacks.TensorBoard(
        #    log_dir=lc.tensorboard_dir,
        #    histogram_freq=0,
        #    batch_size=lc.batch_size,
        #    write_graph=True,
        #    write_grads=True
        #)
        self.tensorboard = keras.callbacks.TensorBoard(
                log_dir=lc.tensorboard_dir,
                histogram_freq=0,
                write_graph=True
        )
        self.tensorboard.set_model(self.Tmodel)
        self.tensorboard_batch_id = train_count_init

    def build_model(self):
        return self.mc.build_train_model(name="T")
        #with self.default_graph.as_default():
        #    with self.session.as_default():
        #        return self.mc.build_train_model(name="T")

    def save_model(self, fnwps):
        model_AIO_fnwp, config_fnwp, weight_fnwp = fnwps

        #with self.default_graph.as_default():
        #    with self.session.as_default():
        self.Tmodel.save(model_AIO_fnwp, overwrite=True, include_optimizer=True,save_format="h5")
        if not os.path.exists(config_fnwp):
            model_json = self.Pmodel.to_json()
            with open(config_fnwp, "w") as json_file:
                json_file.write(model_json)

        model_dn, fn = os.path.split(weight_fnwp)
        temp_fnwp = os.path.join(model_dn, "temp.h5")
        with open(temp_fnwp, 'a'):
            os.utime(temp_fnwp, None)
        self.i_wait.start_monitor(temp_fnwp)
        self.Pmodel.save_weights(temp_fnwp,save_format="h5")
        wait_result = self.i_wait.wait_till_finish()
        self.i_wait.stop_monitor(temp_fnwp)
        assert wait_result, "Fail to save {0} in 10 minuts(600 second)".format(weight_fnwp)
        os.rename(temp_fnwp, weight_fnwp)

    def load_model(self, fnwps):
        #with self.default_graph.as_default():
        #    with self.session.as_default():
        return self.mc.load_train_model(fnwps)

    def named_loss(self, model, loss):
        result = {}
        if len(model.metrics_names) == 1:
            result[model.metrics_names[0]] = loss
        else:
            assert len(model.metrics_names) > 1
            for l in zip(model.metrics_names, loss):
                result[l[0]] = l[1]
        return result

    def optimize(self):
        #with self.default_graph.as_default():
        #    with self.session.as_default():
        num_record_to_train, loss_this_round = self.mc.optimize_com(self.tb, self.Pmodel, self.Tmodel)
        if num_record_to_train != 0:
            self.tensorboard.on_epoch_end(self.tensorboard_batch_id,
                                          self.named_loss(self.Tmodel, loss_this_round))
            self.tensorboard_batch_id += 1
            if lc.flag_record_state:
                self.mc.rv.recorder_brainer([self.Tmodel.metrics_names, loss_this_round])
        return num_record_to_train, loss_this_round


class Explore_Brain(Brain):
    def __init__(self, GPU_per_program,method_name_of_choose_action):
        Brain.__init__(self, GPU_per_program )
        keras.backend.set_learning_phase(0)  # add by john for error solved by
        self.mc = globals()[lc.system_type]()
        self.Pmodel = self.build_model()
        self.i_action= actionOBOS(lc.train_action_type)
        #self.choose_action = getattr(self,method_name_of_choose_action)(self.LHPP2V2_check_holding)
        self.choose_action = getattr(self, method_name_of_choose_action)

        self.predict= self.predict_OS if lc.P2_current_phase == "Train_Sell" else self.predict_OB

    def build_model(self):
        #with self.default_graph.as_default():
        #     with self.session.as_default():
        Pmodel = self.mc.build_predict_model("P")
        return Pmodel

    def load_weight(self, weight_fnwp):
        #with self.default_graph.as_default():
        #    with self.session.as_default():
        self.Pmodel.load_weights(weight_fnwp)


    def predict_OS(self, state):
        assert lc.P2_current_phase=="Train_Sell"
        lv, sv, av = state
        #with self.default_graph.as_default():
        #    with self.session.as_default():
        p, v = self.Pmodel.predict({'P_input_lv': lv, 'P_input_sv': sv, 'P_input_av': av})
        return p,v

    def predict_OB(self,state):
        assert lc.P2_current_phase == "Train_Buy"
        lv, sv = state
        #with self.default_graph.as_default():
        #    with self.session.as_default():
        p, v = self.Pmodel.predict({'P_input_lv': lv, 'P_input_sv': sv})
        return p, v

    '''
    def predict(self, state):
        if lc.P2_current_phase=="Train_Sell":
            lv, sv, av = state
            with self.default_graph.as_default():
                with self.session.as_default():
                    p, v = self.Pmodel.predict({'P_input_lv': lv, 'P_input_sv': sv, 'P_input_av': av})
                    return p,v
        else:
            assert lc.P2_current_phase=="Train_Buy"
            lv, sv = state
            with self.default_graph.as_default():
                with self.session.as_default():
                    p, v = self.Pmodel.predict({'P_input_lv': lv, 'P_input_sv': sv})
                    return p,v
    '''

    def V2_OS_load_model(self, ob_system_name, Ob_model_tc):
        OB_model_dir=os.path.join(sc.base_dir_RL_system, ob_system_name, "model")
        model_config_fnwp=os.path.join(OB_model_dir, "config.json")
        regex = r'weight_\w+T{0}.h5py'.format(Ob_model_tc)
        lfn=[fn for fn in os.listdir(OB_model_dir) if re.findall(regex, fn)]
        assert len(lfn)==1, "{0} model with train count {1} not found".format(ob_system_name,Ob_model_tc)
        weight_fnwp=os.path.join(OB_model_dir, lfn[0])
        #with self.default_graph.as_default():
        #    with self.session.as_default():
        load_jason_custom_objects={"softmax": keras.backend.softmax,"tf":tf, "concatenate":keras.backend.concatenate,"lc":lc}
        model = keras.models.model_from_json(open(model_config_fnwp, "r").read(),custom_objects=load_jason_custom_objects)
        model.load_weights(weight_fnwp)
        print("successful load model form {0} {1}".format(model_config_fnwp, weight_fnwp))
        return model

    def V2_OS_predict(self, state, model):
        lv, sv, av = state
        #with self.default_graph.as_default():
        #    with self.session.as_default():
        p, v = model.predict({'P_input_lv': lv, 'P_input_sv': sv, 'P_input_av': av})
        return p,v

    LHPP2V2_check_holding = lambda self,av_item: False if av_item[0] == 1 else True
    def choose_action_LHPP2V2(self, state):
        assert lc.P2_current_phase == "Train_Sell"
        assert not lc.flag_multi_buy
        lv, sv, av = state
        actions_probs, SVs = self.predict(state)
        l_a = []
        l_ap = []
        l_sv = []
        for sell_prob, SV, av_item in zip(actions_probs, SVs, av):
            assert len(sell_prob) == 2, sell_prob
            flag_holding=self.LHPP2V2_check_holding(av_item)
            if flag_holding:
                #action = np.random.choice([2, 3], p=action_probs)
                action =self.i_action.I_nets_choose_action(sell_prob)
                l_a.append(action)
                l_ap.append(sell_prob.ravel())
            else:  # not have holding
                action = 0
                l_a.append(action)
                l_ap.append(sell_prob.ravel())
            l_sv.append(SV[0])
        return l_a, l_ap, l_sv


    def choose_action_LHPP2V3(self,state):
        assert lc.P2_current_phase == "Train_Buy"
        assert not lc.flag_multi_buy
        lv, sv, av = state
        buy_probs, buy_SVs = self.predict([lv, sv])
        if not hasattr(self, "OS_agent"):
            self.OS_agent = self.V2_OS_load_model(lc.P2_sell_system_name, lc.P2_sell_model_tc)
            self.i_OS_action=actionOBOS("OS")
        sel_probs, sell_SVs = self.V2_OS_predict(state,self.OS_agent)
        l_a = []
        l_ap = []
        l_sv = []
        for buy_prob, sell_prob, buy_sv, sell_sv, av_item in zip(buy_probs,sel_probs,buy_SVs,sell_SVs,av):
            assert len(buy_prob)==2
            assert len(sell_prob) == 2
            flag_holding=self.LHPP2V2_check_holding(av_item)
            if flag_holding:
                #action = np.random.choice([2, 3], p=sell_prob)
                action = self.i_OS_action.I_nets_choose_action(sell_prob)
                l_a.append(action)
                l_ap.append(np.zeros_like(sell_prob))  # this is add zero and this record will be removed by TD_buffer before send to server for train
                l_sv.append(sell_sv[0])
            else: # not have holding
                #action = np.random.choice([0, 1], p=buy_prob)
                action = self.i_action.I_nets_choose_action(buy_prob)
                l_a.append(action)
                l_ap.append(buy_prob)
                l_sv.append(buy_sv[0])
        return l_a, l_ap,l_sv


    def choose_action_LHPP2V4(self, state):
        assert lc.P2_current_phase == "Train_Buy"

        assert not lc.flag_multi_buy
        lv, sv, av = state
        #with self.default_graph.as_default():
        #    with self.session.as_default():
        buy_Qs = self.Pmodel.predict({'P_input_lv': lv, 'P_input_sv': sv})
        if not hasattr(self, "OS_agent"):
            self.OS_agent = self.V2_OS_load_model(lc.P2_sell_system_name, lc.P2_sell_model_tc)
            self.i_OS_action = actionOBOS("OS")
        sel_probs, sell_SVs = self.V2_OS_predict(state, self.OS_agent)

        l_a = []
        l_ap = []
        l_sv = []
        for buy_Q, sell_prob, av_item in zip(buy_Qs, sel_probs, av):
            assert len(buy_Q) == lc.train_action_num
            assert len(sell_prob) == 2
            flag_holding = self.LHPP2V2_check_holding(av_item)
            if flag_holding:
                #action = np.random.choice([2, 3], p=sell_prob)
                action = self.i_OS_action.I_nets_choose_action(sell_prob)
                l_a.append(action)
                l_ap.append(np.zeros_like(sell_prob))
                l_sv.append(0)   #use l_state_value as Q value
            else:  # not have holding
                ### remove .num_action
                #buy_prob = np.ones(lc.specific_param.LHPP2V4_num_action,dtype=float) * \
                #           lc.specific_param.LHPP2V4_epsilon / lc.specific_param.LHPP2V4_num_action
                buy_prob = np.ones((lc.train_num_action,), dtype=float) * \
                               lc.specific_param.LHPP2V4_epsilon / lc.train_num_action

                action = np.argmax(buy_Q)
                buy_prob[action] += (1.0 - lc.specific_param.LHPP2V4_epsilon)
                #adjust_buy_prob = np.append(buy_prob, [0., 0.])     ### remove .num_action
                #gready_action = np.random.choice(np.arange(len(adjust_buy_prob)),p=adjust_buy_prob) ### remove .num_action
                #l_a.append(gready_action) ### remove .num_action
                #l_ap.append(adjust_buy_prob) ### remove .num_action

                gready_action = np.random.choice(np.arange(len(buy_prob)),p=buy_prob)
                l_a.append(gready_action)
                l_ap.append(buy_prob)
                l_sv.append(0)
        return l_a, l_ap, l_sv
    #copy from choose_action_LHPP2V3  mainly handle sell_prob lenght is 2 and buy prob lenth is 3
    def choose_action_LHPP2V5(self,state):
        assert lc.P2_current_phase == "Train_Buy"
        assert not lc.flag_multi_buy
        lv, sv, av = state
        buy_probs, buy_SVs = self.predict([lv, sv])
        if not hasattr(self, "OS_agent"):
            self.OS_agent = self.V2_OS_load_model(lc.P2_sell_system_name, lc.P2_sell_model_tc)
            self.i_OS_action=actionOBOS("OS")
        sel_probs, sell_SVs = self.V2_OS_predict(state,self.OS_agent)
        l_a = []
        l_ap = []
        l_sv = []
        for buy_prob, sell_prob, buy_sv, sell_sv, av_item in zip(buy_probs,sel_probs,buy_SVs,sell_SVs,av):
            assert len(buy_prob)==3
            assert len(sell_prob) == 2
            flag_holding=self.LHPP2V2_check_holding(av_item)
            if flag_holding:
                #action = np.random.choice([2, 3], p=sell_prob)
                action = self.i_OS_action.I_nets_choose_action(sell_prob)
                l_a.append(action)
                l_ap.append(np.zeros(len(sell_prob)+1))  # this is add zero and this record will be removed by TD_buffer before send to server for train
                l_sv.append(sell_sv[0])
            else: # not have holding
                #action = np.random.choice([0, 1], p=buy_prob)
                action = self.i_action.I_nets_choose_action(buy_prob)
                l_a.append(action)
                l_ap.append(buy_prob)
                l_sv.append(buy_sv[0])
        return l_a, l_ap,l_sv


    '''
    def choose_action_LHPP2V2(self, fun_check_holding):
        assert lc.P2_current_phase == "Train_Sell"
        def base_choose_action_LHPP2V2(state):
            assert not lc.flag_multi_buy
            lv, sv, av = state
            actions_probs, SVs = self.predict(state)
            l_a = []
            l_ap = []
            l_sv = []
            for sell_prob, SV, av_item in zip(actions_probs, SVs, av):
                assert len(sell_prob) == 2, sell_prob
                flag_holding=fun_check_holding(av_item)
                if flag_holding:
                    #action = np.random.choice([2, 3], p=action_probs)
                    action =self.i_action.I_nets_choose_action(sell_prob)
                    l_a.append(action)
                    l_ap.append(sell_prob.ravel())
                else:  # not have holding
                    action = 0
                    l_a.append(action)
                    l_ap.append(sell_prob.ravel())
                l_sv.append(SV[0])
            return l_a, l_ap, l_sv
        return base_choose_action_LHPP2V2

    def choose_action_LHPP2V3(self, fun_check_holding):
        assert lc.P2_current_phase == "Train_Buy"
        def base_choose_action_LHPP2V3(state):
            assert not lc.flag_multi_buy
            lv, sv, av = state
            buy_probs, buy_SVs = self.predict([lv, sv])
            if not hasattr(self, "OS_agent"):
                self.OS_agent = self.V2_OS_load_model(lc.P2_sell_system_name, lc.P2_sell_model_tc)
                self.i_OS_action=actionOBOS("OS")
            sel_probs, sell_SVs = self.V2_OS_predict(state,self.OS_agent)
            l_a = []
            l_ap = []
            l_sv = []
            for buy_prob, sell_prob, buy_sv, sell_sv, av_item in zip(buy_probs,sel_probs,buy_SVs,sell_SVs,av):
                assert len(buy_prob)==2
                assert len(sell_prob) == 2
                flag_holding=fun_check_holding(av_item)
                if flag_holding:
                    #action = np.random.choice([2, 3], p=sell_prob)
                    action = self.i_OS_action.I_nets_choose_action(sell_prob)
                    l_a.append(action)
                    l_ap.append(np.zeros_like(sell_prob))  # this is add zero and this record will be removed by TD_buffer before send to server for train
                    l_sv.append(sell_sv[0])
                else: # not have holding
                    #action = np.random.choice([0, 1], p=buy_prob)
                    action = self.i_action.I_nets_choose_action(buy_prob)
                    l_a.append(action)
                    l_ap.append(buy_prob)
                    l_sv.append(buy_sv[0])
            return l_a, l_ap,l_sv
        return base_choose_action_LHPP2V3

    def choose_action_LHPP2V4(self, fun_check_holding):
        assert lc.P2_current_phase == "Train_Buy"
        def base_choose_action_LHPP2V4(state):
            assert not lc.flag_multi_buy
            lv, sv, av = state
            with self.default_graph.as_default():
                with self.session.as_default():
                    #buy_Qs = self.Pmodel.predict({'P_input_lv': lv, 'P_input_sv': sv, 'P_input_av': av[:,:lc.specific_param.LHPP2V4_av_len]})
                    buy_Qs = self.Pmodel.predict({'P_input_lv': lv, 'P_input_sv': sv})
            if not hasattr(self, "OS_agent"):
                self.OS_agent = self.V2_OS_load_model(lc.P2_sell_system_name, lc.P2_sell_model_tc)
                self.i_OS_action = actionOBOS("OS")
            sel_probs, sell_SVs = self.V2_OS_predict(state, self.OS_agent)

            l_a = []
            l_ap = []
            l_sv = []
            for buy_Q, sell_prob, av_item in zip(buy_Qs, sel_probs, av):
                assert len(buy_Q) == lc.train_action_num
                assert len(sell_prob) == 2
                flag_holding = fun_check_holding(av_item)
                if flag_holding:
                    #action = np.random.choice([2, 3], p=sell_prob)
                    action = self.i_OS_action.I_nets_choose_action(sell_prob)
                    l_a.append(action)
                    l_ap.append(np.zeros_like(sell_prob))
                    l_sv.append(0)   #use l_state_value as Q value
                else:  # not have holding
                    ### remove .num_action
                    #buy_prob = np.ones(lc.specific_param.LHPP2V4_num_action,dtype=float) * \
                    #           lc.specific_param.LHPP2V4_epsilon / lc.specific_param.LHPP2V4_num_action
                    buy_prob = np.ones((lc.train_num_action,), dtype=float) * \
                                   lc.specific_param.LHPP2V4_epsilon / lc.train_num_action

                    action = np.argmax(buy_Q)
                    buy_prob[action] += (1.0 - lc.specific_param.LHPP2V4_epsilon)
                    #adjust_buy_prob = np.append(buy_prob, [0., 0.])     ### remove .num_action
                    #gready_action = np.random.choice(np.arange(len(adjust_buy_prob)),p=adjust_buy_prob) ### remove .num_action
                    #l_a.append(gready_action) ### remove .num_action
                    #l_ap.append(adjust_buy_prob) ### remove .num_action

                    gready_action = np.random.choice(np.arange(len(buy_prob)),p=buy_prob)
                    l_a.append(gready_action)
                    l_ap.append(buy_prob)
                    l_sv.append(0)
            return l_a, l_ap, l_sv
        return base_choose_action_LHPP2V4
    '''
'''
class Eval_Brain(Explore_Brain):
    def build_model(self):
        #with self.default_graph.as_default():
        #     with self.session.as_default():
        Pmodel = self.mc.build_predict_model("E")
        return Pmodel

    def predict_OS(self, state):
        assert lc.P2_current_phase=="Train_Sell"
        lv, sv, av = state
        #with self.default_graph.as_default():
        #    with self.session.as_default():
        p, v = self.Pmodel.predict({'E_input_lv': lv, 'E_input_sv': sv, 'E_input_av': av})
        return p,v

    def predict_OB(self,state):
        assert lc.P2_current_phase == "Train_Buy"
        lv, sv = state
        #with self.default_graph.as_default():
        #    with self.session.as_default():
        p, v = self.Pmodel.predict({'E_input_lv': lv, 'E_input_sv': sv})
        return p, v

    def V2_OS_predict(self, state, model):
        lv, sv, av = state
        #with self.default_graph.as_default():
        #    with self.session.as_default():
        p, v = model.predict({'E_input_lv': lv, 'E_input_sv': sv, 'E_input_av': av})
        return p,v
'''

class visual_Explore_Brain(Explore_Brain):
    def __init__(self, GPU_per_program, layer_name):
        Explore_Brain.__init__(self, GPU_per_program)
        #self.debug_fun=K.function(self.Pmodel.input + [K.learning_phase()],
        #                     self.Pmodel.output + [self.Pmodel.get_layer("P_state").output])
        m=self.Pmodel
        self.get_state_fun = K.function(m.input + [K.learning_phase()], m.output + [m.get_layer(layer_name).output])

    def debug_choose_action(self, inputs):
        #lv, sv, av = input
        #with self.default_graph.as_default():
        #    with self.session.as_default():
        action_probs, v, state = self.get_state_fun(inputs)
        #print action_probs.shape, action_probs
        action = np.random.choice(np.arange(action_probs[0].shape[0]), p=action_probs[0].ravel())
        return action, action_probs[0],  v[0], state[0]

