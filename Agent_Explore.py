from Agent_Comm import *

class AgentMain(Process):
    def __init__(self,lc,LL_output, D_share, E_Stop_Agent_Explore, E_Update_Weight, L_E_Weight_Updated_Agent ,L_Agent2GPU,LL_GPU2Agent):
        Process.__init__(self)
        self.lc, self.LL_output, self.D_share, self.E_Stop_Agent_Explore, self.E_Update_Weight, \
        self.L_E_Weight_Updated_Agent ,self.L_Agent2GPU,self.LL_GPU2Agent=\
            lc,LL_output, D_share, E_Stop_Agent_Explore, E_Update_Weight, \
            L_E_Weight_Updated_Agent ,L_Agent2GPU,LL_GPU2Agent
        self.process_name=self.lc.client_process_name_seed
        self.logger= lcom.setup_logger(self.lc,self.process_name,flag_file_log=True, flag_screen_show=True)
        self.inp = pcom.name_pipe_cmd(self.lc,self.process_name)

    def run(self):
        setproctitle.setproctitle("{0}_{1}".format(self.lc.RL_system_name,self.process_name))
        self.logger.info("{0} start".format(self.process_name))
        import tensorflow as tf
        from nets import Explore_Brain,init_virtual_GPU
        tf.random.set_seed(2)
        random.seed(2)
        np.random.seed(2)
        assert self.lc.percent_gpu_core_for_work!= 0.0, "Only Support GPU"
        virtual_GPU = init_virtual_GPU(self.lc.percent_gpu_core_for_work)
        with tf.device(virtual_GPU):
            self.i_wb = locals()[self.lc.CLN_brain_explore](self.lc)
            self.logger.info("Wait for init Agent GPU Weight Updat")

            while not self.E_Update_Weight.is_set():
                time.sleep(1)
            self.weight_fnwp = self.D_share["weight_fnwp"]
            self.i_wb.load_weight(self.weight_fnwp)
            for idx in list(range(self.lc.num_workers)):
                self.L_E_Weight_Updated_Agent[idx].set()
            self.E_Update_Weight.clear()
            self.logger.info("Agent GPU Weight Updated to {0}".format(self.weight_fnwp))

            while not self.E_Stop_Agent_Explore.is_set():
                if len(self.L_Agent2GPU)!=0:
                    process_idx,stacted_state=self.L_Agent2GPU.pop()
                    result = self.i_wb.choose_action(stacted_state, "Explore")
                    self.LL_GPU2Agent[process_idx].append(result)

                if self.E_Update_Weight.is_set():
                    self.weight_fnwp = self.D_share["weight_fnwp"]
                    self.i_wb.load_weight(self.weight_fnwp)
                    for idx in list(range(self.lc.num_workers)):
                        self.L_E_Weight_Updated_Agent[idx].set()
                    self.E_Update_Weight.clear()
                    self.logger.info("Agent GPU Weight Updated to {0}".format(self.weight_fnwp))
                self.name_pipe_cmd()


    def name_pipe_cmd(self):
        cmd_list = self.inp.check_input_immediate_return()
        if cmd_list is not None:
            if cmd_list[0][:-1] == "status":
                print ("{0} has {1} record in L_Agent2GPU".format(self.process_name, len(self.L_Agent2GPU)))
                print ("{0} used as in weight".format(self.weight_fnwp))
            else:
                print("Unknown command: {0} receive from name pipe: {1}".format(cmd_list, self.inp.np_fnwp))

class Agent_Sub(Process):
    def __init__(self,lc, process_idx, L_output, L_Agent2GPU, L_GPU2Agent,E_stop, E_Weight_updated):
        Process.__init__(self)
        self.lc,self.process_idx,self.L_output, self.L_Agent2GPU, self.L_GPU2Agent,self.E_stop,self.E_Weight_updated = \
            lc,process_idx, L_output, L_Agent2GPU, L_GPU2Agent,E_stop,E_Weight_updated

        self.iSL = DBI_Base.StockList(self.lc.SLName)
        SL_idx, self.SL_StartI, self.SL_EndI =  self.lc.train_SL_param
        flag, total_stock_list = self.iSL.get_sub_sl("Train", SL_idx)
        assert flag, "Get Stock list {0} tag=\"Train\" index={1}".format(self.lc.SLName, self.process_idx)
        mod=len(total_stock_list)//self.lc.num_workers
        left=len(total_stock_list)%self.lc.num_workers
        self.stock_list = total_stock_list[process_idx * mod:(process_idx + 1) * mod]
        if process_idx<left:
            self.stock_list.append(total_stock_list[-(process_idx+1)])

        self.process_name = "{0}_{1}".format(self.lc.client_process_name_seed,self.process_idx)
        self.process_working_dir = os.path.join(lc.system_working_dir, self.process_name)
        if not os.path.exists(self.process_working_dir): os.mkdir(self.process_working_dir)
        self.inp = pcom.name_pipe_cmd(self.lc,self.process_name)

        self.logger= lcom.setup_logger(self.lc,self.process_name,flag_file_log=self.lc.l_flag_worker_log_file[self.process_idx],
                                       flag_screen_show=self.lc.l_flag_worker_log_screen[self.process_idx])

        self.data = client_datas(self.lc, self.process_working_dir, self.lc.data_name, self.stock_list, self.SL_StartI,
                    self.SL_EndI, self.logger, self.lc.CLN_env_get_data_train, called_by="Explore")
        self.i_train_buffer_to_server = globals()[self.lc.CLN_buffer_to_train](self.lc,len(self.stock_list))
        self.i_bs = buffer_series()

        '''modify move_get actula to action common '''
        self.i_ac = actionOBOS(self.lc.train_action_type)

        #self.max_record_sent_per_update_weight=int((self.lc.num_train_to_save_model*self.lc.batch_size/self.lc.num_workers)+20*self.lc.batch_size)
        self.max_record_sent_per_update_weight=int((self.lc.num_train_to_save_model*self.lc.batch_size/
                                    (self.lc.num_workers*self.lc.brain_buffer_reuse_times))+20*self.lc.batch_size)

        if self.lc.flag_record_buffer_to_server:
            dirwp = os.path.join(sc.base_dir_RL_system, self.lc.RL_system_name, "record_send_buffer")
            self.i_record_send_to_server=record_send_to_server(dirwp ,self.lc.flag_record_buffer_to_server)

    def run(self):
        setproctitle.setproctitle("{0}_{1}".format(self.lc.RL_system_name, self.process_name))
        random.seed(2)
        np.random.seed(2)
        self.logger.info("{0} start".format(self.process_name))
        Ds={}
        Ds["worker_loop_count"]=-1
        Ds["accumulate_record_sent_per_print"] = 0
        Ds["accumulate_record_sent_per_update"] = 0
        Ds["flag_sent_enough_item"] = False
        Ds["flag_response_received"]=True
        while not self.E_stop.is_set():
            if Ds["accumulate_record_sent_per_update"]<self.max_record_sent_per_update_weight:
                if Ds["flag_response_received"]:
                    Ds["worker_loop_count"] += 1
                    self.run_env_one_step(Ds["worker_loop_count"])
                    stacted_state = self.data.stack_l_state(self.data.l_s)
                    self.L_Agent2GPU.append([self.process_idx, stacted_state])
                    Ds["flag_response_received"]=False
                else:
                    buf_len = len(self.L_GPU2Agent)
                    if  buf_len== 1:
                        Ds["flag_response_received"],result=True, self.L_GPU2Agent.pop()
                        self.data.l_a, self.data.l_ap, self.data.l_sv = result
                        num_record_sent = self.worker_send_buffer_brain()
                        Ds["accumulate_record_sent_per_print"] += num_record_sent
                        Ds["accumulate_record_sent_per_update"] += num_record_sent
                    else:
                        assert buf_len== 0, "L_GPU2Agent only can have length 0 or 1 , not get {0} {1}".format(buf_len,self.L_GPU2Agent)

                '''
                Ds["worker_loop_count"] +=1
                self.run_env_one_step(Ds["worker_loop_count"])
                stacted_state = self.data.stack_l_state(self.data.l_s)
                self.L_Agent2GPU.append([self.process_idx, stacted_state])
                flag_received,result=False,[]
                while not flag_received and not self.E_stop.is_set():
                    buf_len = len(self.L_GPU2Agent)
                    assert buf_len in [0, 1]
                    if buf_len == 1:
                        flag_received,result=True, self.L_GPU2Agent.pop()
                    else:
                        time.sleep(0.1)
                else:
                    assert len(result)!=0
                    self.data.l_a, self.data.l_ap, self.data.l_sv=result
                num_record_sent=self.worker_send_buffer_brain()
                Ds["accumulate_record_sent_per_print"] +=num_record_sent
                Ds["accumulate_record_sent_per_update"] +=num_record_sent
                '''
            else:
                if not Ds["flag_sent_enough_item"]:
                    Ds["flag_sent_enough_item"]=True
                    self.logger.info("loop_count {0} sent enough record {1} start wait".
                                     format(Ds["worker_loop_count"],Ds["accumulate_record_sent_per_update"]))
            if self.E_Weight_updated.is_set():
                self.logger.info("loop_count {0} weight updated and worker start work".format(
                    Ds["worker_loop_count"]))
                Ds["accumulate_record_sent_per_update"]=0
                Ds["flag_sent_enough_item"] = False
                self.E_Weight_updated.clear()

            if Ds["worker_loop_count"] %500==0 and Ds["worker_loop_count"]!=0:
                self.logger.info("loop_count {0} accumulate record sent during this period {1} send buffer series {2}".
                                 format(Ds["worker_loop_count"],Ds["accumulate_record_sent_per_print"],self.i_bs.get_current() ))
                Ds["accumulate_record_sent_per_print"] =0

            self.name_pipe_cmd(Ds)

            if Ds["flag_sent_enough_item"]:  # this is add to relase the worker while not need to create more train records
                time.sleep(1)
        self.logger.info("stopped")


    def run_env_one_step(self,worker_loop_count):
        for idx, i_env in enumerate(self.data.l_i_env):
            if self.data.l_done_flag[idx]:
                s, _ = i_env.reset() #s, support_view_dic = i_env.reset()
                self.data.l_s[idx] = s
                self.data.l_done_flag[idx] = False
            else:
                s = self.data.l_s[idx]
                a = self.data.l_a[idx]
                #ap =self.data.l_ap[idx]
                s_, r, done, support_view_dic, actual_action = i_env.step(a)
                self.data.l_done_flag[idx] = done
                a_onehot04,support_view_dic["old_ap"]=self.i_ac.I_A3C_worker_explorer(actual_action, self.data.l_ap[idx])
                self.clean_support_view_from_worker_to_server(support_view_dic)
                self.i_train_buffer_to_server.add_one_record(idx, s, a_onehot04, r, s_, done, support_view_dic)
                self.data.l_s[idx] = s_

    def clean_support_view_from_worker_to_server(self,support_view_dic):
        support_view_dic.pop("Flag_LastDay")
        support_view_dic.pop("Nprice")
        support_view_dic.pop("HFQRatio")
        support_view_dic.pop("Flag_Tradable")
        support_view_dic.pop("flag_all_period_explored")

        #support_view_dic.pop("Stock")  # add by Env need use by revorder state
        #support_view_dic.pop("DateI")  # add by Env need use by revorder state

        #not change following due to record need modified accordingly
        #support_view_dic.pop("action_return_message") #env
        #support_view_dic.pop("action_taken") #env
        #remove "flag_force_sell" #support_view_dic.pop("flag_force_sell")    #av_state
        #support_view_dic.pop("old_ap")   #Explore_process

        # TD_intergrated will be add after add this fun
        #support_view_dic.pop(""SdisS_"")
        # support_view_dic.pop("_support_view_dic")   #TD_intergrated

        assert len(support_view_dic.keys())==5,support_view_dic.keys()

    def worker_send_buffer_brain(self):
        if self.i_train_buffer_to_server.get_len_train_buffer_to_server() > self.lc.num_train_record_to_brain:
            train_buffer_to_server = self.i_train_buffer_to_server.get_train_buffer_to_server()
            ## add sent explore brain train_count to each record to set
            length_sent = len(train_buffer_to_server)
            train_buffer_to_send = list(train_buffer_to_server)
            if self.lc.flag_record_buffer_to_server:
                self.i_record_send_to_server.saver(train_buffer_to_send)
            self.L_output.append([self.process_idx, self.i_bs.set_get_next(), train_buffer_to_send])
            self.i_train_buffer_to_server.empty_train_buffer_to_server()
            return length_sent
        return 0

    def name_pipe_cmd(self, Ds):
        cmd_list = self.inp.check_input_immediate_return()
        if cmd_list is not None:
            if cmd_list[0][:-1] == "status":
                print("|||worker:{0} loop_count:{1} |||l_idx_valid_flag: {2}|||" \
                    .format(self.process_idx, Ds["worker_loop_count"],self.data.l_idx_valid_flag))
                print(Ds)
                print("are length {0}  ".format([len(self.data.l_log_a_r_e[idx]) for idx in range(len(self.data.l_idx_valid_flag))]))
                print("ssdi length {0}  ".format([len(self.data.l_log_stock_episode[idx]) for idx in range(len(self.data.l_idx_valid_flag))]))
            elif cmd_list[0][:-1] == "send_more":
                self.logger.info("{0}".format(Ds))
                Ds["accumulate_record_sent_per_update"] =self.max_record_sent_per_update_weight-1000
                Ds["flag_sent_enough_item"]=False

            else:
                print("Unknown command: {0} receive from name pipe: {1}".format(cmd_list, self.inp.np_fnwp))
