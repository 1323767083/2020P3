from Agent_Comm import *
from State import *
from Eval_WR import WR_handler
class EvalMain(Process):
    def __init__(self,lc,E_Stop_Agent_Eval, L_E_Start1Round,L_Eval2GPU,LL_GPU2Eval,Share_eval_loop_count):
        Process.__init__(self)
        self.lc,self.E_Stop_Agent_Eval,self.L_E_Start1Round,self.L_Eval2GPU,self.LL_GPU2Eval,self.Share_eval_loop_count\
            =lc,E_Stop_Agent_Eval,  L_E_Start1Round,L_Eval2GPU,LL_GPU2Eval,Share_eval_loop_count
        self.process_name=self.lc.eval_process_seed
        self.logger= lcom.setup_logger(self.lc,self.process_name,flag_file_log=True, flag_screen_show=True)
        self.inp = pcom.name_pipe_cmd(self.lc,self.process_name)
        self.current_eval_count = self.lc.start_eval_count // self.lc.num_train_to_save_model
    def run(self):
        setproctitle.setproctitle("{0}_{1}".format(self.lc.RL_system_name,self.process_name))
        self.logger.info("{0} start".format(self.process_name))
        self.logger.info("All Eval Subs start")
        import tensorflow as tf
        from nets import Explore_Brain, init_virtual_GPU
        assert self.lc.percent_gpu_core_for_eva!= 0.0, "Only Support GPU"
        virtual_GPU = init_virtual_GPU(self.lc.percent_gpu_core_for_eva)

        with tf.device(virtual_GPU):
            self.i_eb = locals()[self.lc.CLN_brain_explore](self.lc)
            self.current_phase = 0
            self.flag_validate_current_eval_count = False
            while not self.E_Stop_Agent_Eval.is_set():
                if self.current_phase ==0 :  #wait weight file ready
                    if not self.flag_validate_current_eval_count:
                        self.current_eval_count += self.lc.eval_model_stride
                        eval_loop_count = self.current_eval_count * self.lc.num_train_to_save_model
                        self.flag_validate_current_eval_count = True
                        self.logger.info("Eval GPU  wait for Weights on {0}".format(eval_loop_count))
                    model_weight_fnwp = self.EvaMain_Init_Round(eval_loop_count)
                    if len(model_weight_fnwp) != 0:
                        self.i_eb.load_weight(model_weight_fnwp)
                        self.flag_validate_current_eval_count = False
                        self.current_phase=1
                        tf.random.set_seed(3)
                        random.seed(3)
                        np.random.seed(3)
                        #with self.Share_eval_loop_count.get_lock(): #TODO share value
                        self.Share_eval_loop_count.value = eval_loop_count
                        for E_Start1Round in self.L_E_Start1Round:
                            E_Start1Round.set()
                        self.logger.info("Eval GPU  Weights Updated to {0}".format(model_weight_fnwp))
                    else:
                        time.sleep(1)
                elif self.current_phase ==1: # wait all subs finish this round
                    if len(self.L_Eval2GPU) != 0:
                        process_idx, stacted_state = self.L_Eval2GPU.pop()
                        result = self.i_eb.choose_action(stacted_state, "Eval")
                        self.LL_GPU2Eval[process_idx].append(result)

                    if all([not E_Start1Round.is_set() for E_Start1Round in self.L_E_Start1Round])and len(self.L_Eval2GPU) == 0:
                        self.logger.info("Eval GPU finish eval count {0}".format(eval_loop_count))
                        self.current_phase = 0
                else:
                    assert False, "only support current phase 0. wait for wait ready 1. wait for round finish bu get {0}".format(self.current_phase)
                self.name_pipe_cmd()

    def EvaMain_Init_Round(self,eval_loop_count):
        found_model_surfix = find_model_surfix(self.lc.brain_model_dir,eval_loop_count, self.lc.flag_train_store_AIO_model)

        if found_model_surfix is None:
            return ""
        weight_fn = "{0}{1}.h5".format(self.lc.actor_weight_fn_seed, found_model_surfix)
        weight_fnwp=os.path.join(self.lc.brain_model_dir,weight_fn)
        return weight_fnwp

    def name_pipe_cmd(self):
        cmd_list = self.inp.check_input_immediate_return()
        if cmd_list is not None:
            if cmd_list[0][:-1] == "status":
                print ("{0} has {1} record in L_Eval2GPU".format(self.process_name, len(self.L_Eval2GPU)))
                print ("{0} current phase is {1} amd flag_validate_current_eval_count is {2}".\
                    format(self.process_name,self.current_phase,self.flag_validate_current_eval_count))
                print ("{0} eval subs E_Start1Round are {1}".format(self.process_name,[E_Start1Round.is_set() for E_Start1Round in self.L_E_Start1Round]))
            else:
                print("Unknown command: {0} receive from name pipe: {1}".format(cmd_list, self.inp.np_fnwp))

class EvalSub(Process):
    def __init__(self, lc,process_group_idx,process_idx,L_Eval2GPU, L_GPU2Eval,E_stop, E_Start1Round,Share_eval_loop_count):
        Process.__init__(self)
        self.lc, self.process_group_idx,self.process_idx, self.L_Eval2GPU, self.L_GPU2Eval, self.E_stop, self.E_Start1Round, self.Share_eval_loop_count=\
            lc,process_group_idx,process_idx,L_Eval2GPU, L_GPU2Eval,E_stop, E_Start1Round,Share_eval_loop_count
        _, self.SL_StartI, self.SL_EndI = self.lc.l_eval_SL_param[self.process_group_idx]
        self.iSL = DBI_Base.StockList(self.lc.SLName)
        self.stock_list=self.iSL.Get_Eval_SubProcess_SL(lc,process_group_idx,process_idx)
        self.process_name = "{0}_{1}".format(self.lc.eval_process_seed, self.process_idx)
        self.process_group_name="{0}_{1}".format(self.lc.eval_process_seed, self.process_group_idx)
        self.process_working_dir = os.path.join(lc.system_working_dir, self.process_group_name)
        if not os.path.exists(self.process_working_dir): os.mkdir(self.process_working_dir)


        self.logger= lcom.setup_logger(self.lc,self.process_name,flag_file_log=self.lc.l_flag_eval_log_file[process_group_idx],
                                       flag_screen_show=self.lc.l_flag_eval_log_screen[process_group_idx])

        self.inp=pcom.name_pipe_cmd(self.lc,self.process_name)

        self.data = Client_Datas_Eval(self.lc, self.process_working_dir, self.lc.data_name, self.stock_list, self.SL_StartI,
                                 self.SL_EndI, self.logger, self.lc.l_CLN_env_get_data_eval[self.process_group_idx],
                                 called_by="Eval")
        assert self.lc.l_CLN_env_get_data_eval[self.process_group_idx] == "DBTP_Eval_WR_Reader"
        self.Flag_WR_log = True
        if self.Flag_WR_log:
            self.i_WRH= WR_handler(self.lc, self.process_name, self.process_group_name, self.logger)
        self.i_ac = actionOBOS(self.lc.train_action_type)



    def run(self):
        setproctitle.setproctitle("{0}_{1}".format(self.lc.RL_system_name, self.process_name))
        self.logger.info("start at eval loop count {0}".format(self.Share_eval_loop_count.value))
        self.CurrentPhase=0 # 0 wait for self.E_Start1Round set  # 1 do eval
        SL_dir=os.path.join(self.lc.system_working_dir,"SL_Evaled")
        if not os.path.exists(SL_dir): os.mkdir(SL_dir)
        pd.DataFrame(self.stock_list,columns=["Stock"]).to_csv(os.path.join(SL_dir,"SL_Evaled_P{0}.csv".format(self.process_idx)),index=False)
        while not self.E_stop.is_set():
            if self.CurrentPhase==0:  #wait for round start
                if self.E_Start1Round.is_set():
                    self.data.eval_reset_data()
                    self.CurrentPhase=1
                    random.seed(3)
                    np.random.seed(3)
                    self.Flag_Wait_GPU_Response =False
                else:
                    time.sleep(1)
            elif self.CurrentPhase==1:  # in evaluation round
                if not self.Flag_Wait_GPU_Response:
                    self.run_env_one_step()
                    stacted_state = self.data.stack_l_state(self.data.l_s)
                    self.L_Eval2GPU.append([self.process_idx, stacted_state])
                    self.Flag_Wait_GPU_Response=True
                else:
                    buf_len = len(self.L_GPU2Eval)
                    if buf_len == 1:
                        result=self.L_GPU2Eval.pop()
                        self.data.l_a, self.data.l_ap, self.data.l_sv = result
                        self.Flag_Wait_GPU_Response= False
                        if not any(self.data.l_idx_valid_flag):
                            if self.Flag_WR_log:
                                self.i_WRH.save(self.Share_eval_loop_count.value)
                            self.E_Start1Round.clear()
                            self.CurrentPhase = 0
                    else:
                        assert buf_len==0, "self.L_GPU2Eval only can have len 0 or 1 , but now {0} ".format(buf_len)
                self.name_pipe_cmd()
            else:
                assert False, "Current phase only can be 0.wait for round start 1. eval round ongoing but is {0}".format(self.CurrentPhase)

        self.logger.info("stopped")

    def run_env_one_step(self):
        l_WR,l_PA=[],[] # only used in Flag_WR_log is True
        CurrentDateI=0
        for idx, i_env in enumerate(self.data.l_i_env):
            if not self.data.l_idx_valid_flag[idx]:
                if self.Flag_WR_log:
                    state, support_view_dic=i_env.WR_get_data()
            else:
                if self.data.l_done_flag[idx]:
                    s, support_view_dic = i_env.reset()
                    self.data.l_s[idx] = s
                    if support_view_dic["flag_all_period_explored"]:
                        self.data.l_idx_valid_flag[idx] = False
                    else:
                        if self.data.l_i_episode_init_flag[idx]:
                            self.data.l_i_episode_init_flag[idx] = False
                    if self.data.l_i_env[idx].i_av.Is_Force_Next_Reset(self.data.l_s[idx][2][0]):
                        self.data.l_done_flag[idx] = True
                    else:
                        self.data.l_done_flag[idx] = False

                else:
                    #s = self.data.l_s[idx]
                    a = self.data.l_a[idx]
                    ap =self.data.l_ap[idx]
                    s_, r, done, support_view_dic, actual_action = i_env.step(a)
                    self.data.l_s[idx] = s_
                    if self.data.l_i_env[idx].i_av.Is_Force_Next_Reset(self.data.l_s[idx][2][0]):
                        self.data.l_done_flag[idx] = True
                    else:
                        self.data.l_done_flag[idx] = done
            if self.Flag_WR_log:
                if CurrentDateI==0:
                    CurrentDateI=i_env.i_get_data.get_the_dateI()
                else:
                    assert CurrentDateI==i_env.i_get_data.get_the_dateI(),"Current DateI {0} and get_teh_DateI should same".format(CurrentDateI,i_env.i_get_data.get_the_dateI())
                l_WR.append(i_env.check_right_or_wrong())
                l_PA.append(i_env.check_profit())
        if self.Flag_WR_log:
            #["BW", "BZ", "BR", "NW", "NZ", "NR", "NA"]
            #self.i_WRH.log_WRs.append([l_WR.count(0),l_WR.count(1),l_WR.count(2),l_WR.count(10),l_WR.count(11),l_WR.count(12),l_WR.count(-1)])
            self.i_WRH.add_log([CurrentDateI,l_WR,l_PA])
    def name_pipe_cmd(self):
        cmd_list = self.inp.check_input_immediate_return()
        if cmd_list is not None:
            if cmd_list[0][:-1] == "status":
                print("{0} CurrentPhase={1} Flag_Wait_GPU_Response={2} E_Start1Round.is_set={3} ".
                    format(self.process_name,self.CurrentPhase, self.Flag_Wait_GPU_Response, self.E_Start1Round.is_set()))
                print("|||Eval:{0} |||l_idx_valid_flag: {1}|||".format(self.process_idx,self.data.l_idx_valid_flag))
                print("are length {0}  ".format([len(self.data.l_log_a_r_e[idx]) for idx in range(len(self.data.l_idx_valid_flag))]))
            else:
                print("Unknown command: {0} receive from name pipe: {1}".format(cmd_list, self.inp.np_fnwp))

