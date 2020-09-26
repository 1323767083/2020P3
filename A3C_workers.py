import os,re, random,time,setproctitle
import pandas as pd
import numpy as np
from multiprocessing import Process,Event
import pipe_comm as pcom
import logger_comm  as lcom
import config as sc
from recorder import record_send_to_server

from vresult_data_reward import ana_reward_data_A3C_worker_interface
from Buffer_comm import buffer_series,buffer_to_train

from env import Simulator_intergrated
'''modify move_get actula to action common '''
from action_comm import actionOBOS
from miscellaneous import find_model_surfix
def init_A3C_worker(lgc):
    global lc
    lc=lgc
    global Cbuffer_to_server, CSimulator
    Cbuffer_to_server = globals()[lc.CLN_buffer_to_train]
    CSimulator=globals()[lc.CLN_simulator]

class client_datas:
    def __init__(self, process_working_dir, data_name,stock_list, StartI, EndI,logger,CLN_get_data, called_by):
        self.stock_list = stock_list
        self.data_name=data_name
        self.called_by=called_by
        self.process_working_dir = process_working_dir
        self.stock_working_dir = []
        self.logger=logger
        for stock in self.stock_list:
            one_stock_working_dir = os.path.join(self.process_working_dir, stock)
            if not os.path.exists(one_stock_working_dir): os.mkdir(one_stock_working_dir)
            self.stock_working_dir.append(one_stock_working_dir)
        num_stock = len(self.stock_list)
        self.l_i_env = []
        self.l_i_episode = []
        self.l_done_flag = [True for _ in range(num_stock)]
        self.l_s = [[] for _ in range(num_stock)]
        self.l_a = [0 for _ in range(num_stock)]
        self.l_ap = [[] for _ in range(num_stock)]
        # this is add to record the state value
        self.l_sv = [0.0 for _ in range(num_stock)]
        self.l_r = [[] for _ in range(num_stock)]
        self.l_t = [0 for _ in range(num_stock)]
        self.l_log_a_r_e = [[] for _ in range(num_stock)]
        self.l_log_stock_episode = [[] for _ in range(num_stock)]
        self.l_idx_valid_flag = [True for _ in range(num_stock)]  # for eval quiting
        self.l_i_episode = [0 for _ in range(len(self.stock_list))]

        self.l_i_episode_init_flag = [True for _ in range(num_stock)]  # for log purpose

        for stock in self.stock_list:
            i_env = CSimulator(self.data_name,stock,StartI, EndI,CLN_get_data,self.called_by)
            self.l_i_env.append(i_env)

    def eval_reset_data(self):
        self.l_idx_valid_flag = [True for _ in range(len(self.stock_list))]  # for eval quiting
        self.l_i_episode_init_flag = [True for _ in range(len(self.stock_list))]  # for log purpose
        self.l_i_episode=[0 for _ in range(len(self.stock_list))]
        # this is add to record the state value for are currently only used in eval process
        self.l_sv = [0.0 for _ in range(len(self.stock_list))]
        #for env in self.l_i_env:
        #    env.data_reset()
        self.l_done_flag = [True for _ in self.stock_list]   # this is ensure after this fun called, the first round will call reset data

    def worker_reset_data(self):
        self.l_idx_valid_flag = [True for _ in self.stock_list]  # for eval quiting
        self.l_i_episode_init_flag = [True for _ in range(len(self.stock_list))]  # for log purpose
        self.l_i_episode = [0 for _ in self.stock_list]
        self.l_t = [0 for _ in self.stock_list] # as after round save, but worker still work on and l_t will continue to add
        self.l_sv = [0.0 for _ in range(len(self.stock_list))]
        #for env in self.l_i_env:
        #    env.data_reset()
        self.l_done_flag = [True for _ in self.stock_list]         #this is to avoid unfinished reset
        #avoid un saved
        for idx,_ in enumerate(self.stock_list):
            if len (self.l_r[idx])!=0:
                self.logger.error("len (self.l_r[{0}])!=0".format(idx))
                assert len(self.l_r[idx]) == 0
            if len(self.l_log_a_r_e[idx]) != 0:
                self.logger.error("len(self.l_log_a_r_e[{0}]) != 0".format(idx))
                assert len(self.l_log_a_r_e[idx]) == 0
            if len(self.l_log_stock_episode[idx]) != 0:
                self.logger.error("len(self.l_log_stock_episode[{0}]) == 0".format(idx))
                assert len(self.l_log_stock_episode[idx]) == 0
class are_ssdi_handler:
    def __init__(self, process_name, process_working_dir, logger):
        self.process_name=process_name
        self.process_working_dir=process_working_dir
        self.ongoing_save_count = -1
        self.logger=logger
        self.log_are_column=["action", "reward", "episode", "day", "action_result"]

        #for ap_idx in range(lc.num_action): ### remove .num_action
        for ap_idx in range(lc.train_num_action):
            self.log_are_column.append("p{0}".format(ap_idx))
        self.log_are_column.append("state_value")
        #self.log_are_column.extend(["holding","potential_profit","trade_Nprice","trans_id"])
        self.log_are_column.extend(["holding", "trade_Nprice", "trans_id"])
        self.log_esdi_column=["episode", "stock", "length", "adjust_reward", "seconds"]

    def _eval_save(self,data,idx,eval_count):
        if eval_count ==0:
            del data.l_log_a_r_e[idx][:]
            del data.l_log_stock_episode[idx][:]
            return
        log_a_r_e_fn="{0}_T{1}.csv".format(lc.log_a_r_e_fn_seed,eval_count)
        log_e_s_d_i_fn="{0}_T{1}.csv".format(lc.log_e_s_d_i_fn_seed,eval_count)

        log_a_r_e_fn_fnwp=os.path.join(self.process_working_dir,data.stock_list[idx],log_a_r_e_fn )
        log_e_s_d_i_fnwp=os.path.join(self.process_working_dir,data.stock_list[idx],log_e_s_d_i_fn )
        df_log = pd.DataFrame(data.l_log_a_r_e[idx],columns=self.log_are_column)
        df_log.to_csv(log_a_r_e_fn_fnwp, index=False, float_format='%.4f')
        df_map = pd.DataFrame(data.l_log_stock_episode[idx],columns=self.log_esdi_column)
        df_map.to_csv(log_e_s_d_i_fnwp, index=False, float_format='%.4f')
        del data.l_log_a_r_e[idx][:]
        del data.l_log_stock_episode[idx][:]
        return

    def round_save(self, data, idx, flag_finished):
        self.finish_episode(data, idx, flag_finished)
        self.logger.info("{0} round saved".format(data.stock_list[idx]))
        if self.ongoing_save_count ==-1:
            self.logger.error("self.ongoing_save_count is -1")
            assert self.ongoing_save_count !=-1
        self._eval_save(data, idx, self.ongoing_save_count)

    def finish_episode(self, data,idx, flag_finished):
        #r_sum = sum([tr for tr in data.l_r[idx]])
        #assert r_sum==data.l_r[idx][-1]
        r_sum = data.l_r[idx][-1]
        data.l_log_stock_episode[idx].append([data.l_i_episode[idx], data.stock_list[idx], data.l_t[idx], r_sum, 0.0])
        self.logger.info("stock:{0} episode:{1} period_len:{2} reward:{3:.2f} {4} episode add to record"
                        .format(data.stock_list[idx], data.l_i_episode[idx],data.l_t[idx], r_sum,
                                "finished" if flag_finished else "unfinished"))
        data.l_i_episode[idx] += 1
        data.l_t[idx] = 0
        del data.l_r[idx][:]

    def start_round(self, save_count):
        self.ongoing_save_count=save_count

    def in_round(self, data, idx, a, ap, r, sv_dic, trans_id):
        item=[a, r, data.l_i_episode[idx], sv_dic["DateI"], sv_dic["action_return_message"]]
        #for ap_idx in range(lc.num_action):### remove .num_action
        for ap_idx in range(lc.train_num_action):
            item.append(ap[ap_idx])
        item.append(data.l_sv[idx])
        #item.extend([sv_dic["holding"], sv_dic["this_trade_day_Nprice"], trans_id])
        item.extend([sv_dic["holding"], sv_dic["Nprice"], trans_id])
        data.l_log_a_r_e[idx].append(item)

class transaction_id:
    not_in_transaction = "Not_in_trans"
    def __init__(self, stock, start_id=0):
        self.stock = stock
        self.current_counter = start_id
        self.flag_holding = False

    def get_transaction_id(self, flag_new_holding):
        if not self.flag_holding and not flag_new_holding:
            self.current_trans_id = self.not_in_transaction
        elif not self.flag_holding and flag_new_holding:
            self.current_counter += 1
            self.current_trans_id = "{0}_T{1}".format(self.stock, self.current_counter)
        elif self.flag_holding and flag_new_holding:
            # keep current trans _id
            pass
        elif self.flag_holding and not flag_new_holding:
            # keep current trans _id, but status change while self.flag_holding=flag_new_holding
            pass
        self.flag_holding = flag_new_holding
        return self.current_trans_id

    def reset_flag_holding(self): # to solve the new eval continue with the last trans_id
        self.flag_holding = False

class client_base(Process):
    def __init__(self, process_name, process_idx, lstock,SL_StartI,SL_EendI, L_output, D_share, E_stop, E_update_weight, E_worker_work,CLN_get_data):
        Process.__init__(self)
        self.process_name = process_name
        self.process_idx = process_idx
        self.L_output=L_output
        self.stock_list=lstock
        self.SL_StartI=SL_StartI
        self.SL_EendI=SL_EendI
        self.D_share=D_share
        self.E_stop=E_stop
        self.E_update_weight=E_update_weight
        self.E_worker_work=E_worker_work
        self.CLN_get_data=CLN_get_data
        self.process_working_dir=os.path.join(lc.system_working_dir,self.process_name)
        if not os.path.exists(self.process_working_dir): os.mkdir(self.process_working_dir)
        self.inp=pcom.name_pipe_cmd(self.process_name)

    def stack_l_state(self,l_state):
        l_lv,l_sv,l_av=[],[],[]
        for state in l_state:
            lv,sv,av=state
            l_lv.append(lv)
            l_sv.append(sv)
            l_av.append(av)
        stack_state=[np.concatenate(l_lv, axis=0), np.concatenate(l_sv, axis=0), np.concatenate(l_av, axis=0)]
        return stack_state


    def find_train_count_through_weight_fnwp(self, weight_fnwp):
        _, fn = os.path.split(weight_fnwp)
        regex = r'\w*_T(\d*).h5py'
        match = re.search(regex, fn)
        assert match, "faile to find train count in {0}".format(weight_fnwp)
        return int(match.group(1))

class Explore_process(client_base):
    def __init__(self, process_name, process_idx, data_name,learn_sl,SL_StartI, SL_EndI,L_output, D_share, E_stop, E_update_weight, E_worker_work,CLN_get_data):
        client_base.__init__(self, process_name, process_idx, learn_sl,SL_StartI, SL_EndI, L_output, D_share, E_stop, E_update_weight, E_worker_work,CLN_get_data)
        #self.init_data_explore(eval_loop_count =0) # not eval worker
        self.data_name=data_name
        flag_file_log = lc.l_flag_worker_log_file[process_idx]
        flag_screen_show = lc.l_flag_worker_log_screen[process_idx]
        self.logger= lcom.setup_logger(process_name,flag_file_log=flag_file_log, flag_screen_show=flag_screen_show)
        self.init_data_explore()
        self.E_update_weight=E_update_weight
        '''modify move_get actula to action common '''
        self.i_ac = actionOBOS(lc.train_action_type)

        self.max_record_sent_per_update_weight=int((lc.num_train_to_save_model*lc.batch_size/lc.num_workers)+20*lc.batch_size)
        self.max_record_sent_per_update_weight=int((lc.num_train_to_save_model*lc.batch_size/(lc.num_workers*lc.brain_buffer_reuse_times))+20*lc.batch_size)
        dirwp=os.path.join(sc.base_dir_RL_system, lc.RL_system_name,"record_send_buffer")
        if lc.flag_record_buffer_to_server:
            self.i_record_send_to_server=record_send_to_server(dirwp ,lc.flag_record_buffer_to_server)
    def init_data_explore(self):
        self.data = client_datas(self.process_working_dir, self.data_name,self.stock_list, self.SL_StartI,self.SL_EendI,self.logger,self.CLN_get_data,called_by="Explore")
        self.i_train_buffer_to_server = Cbuffer_to_server(len(self.stock_list))
        self.i_bs=buffer_series()
    def run(self):
        import tensorflow as tf
        #lcom.setup_tf_logger(self.process_name)
        tf.random.set_seed(2)
        from nets import Explore_Brain, init_gc,init_virtual_GPU
        init_gc(lc)    #init_gc(lgc)
        setproctitle.setproctitle("{0}_{1}".format(lc.RL_system_name, self.process_name))
        random.seed(2)
        np.random.seed(2)
        assert lc.l_percent_gpu_core_for_work[self.process_idx]!=0.0, "only work on GPU"
        virtual_GPU = init_virtual_GPU(lc.l_percent_gpu_core_for_work[self.process_idx])
        with tf.device(virtual_GPU):
            self.logger.info("{0} start".format(self.process_name))
            self.i_wb= locals()[lc.CLN_brain_explore]()
            self.logger.info(" wait for initial weight ")
            Ds={}
            Ds["worker_loop_count"]=-1
            Ds["accumulate_record_sent_per_print"] = 0
            Ds["accumulate_record_sent_per_update"] = 0
            Ds["flag_sent_enough_item"] = False
            Ds["This_round_received_weight_fnwp"]=""
            Ds["This_round_received_weight_fnwp"] = self.worker_check_update_weight()
            while len(Ds["This_round_received_weight_fnwp"])==0:
                Ds["This_round_received_weight_fnwp"] = self.worker_check_update_weight()
                time.sleep(1)
            else:
                self.logger.info("initial weight update from {0}".format(Ds["This_round_received_weight_fnwp"]))
            while not self.E_stop.is_set():
                if Ds["accumulate_record_sent_per_update"]<self.max_record_sent_per_update_weight:
                    Ds["worker_loop_count"] +=1
                    self.run_env_one_step(Ds["worker_loop_count"])
                    num_record_sent=self.worker_send_buffer_brain()
                    Ds["accumulate_record_sent_per_print"] +=num_record_sent
                    Ds["accumulate_record_sent_per_update"] +=num_record_sent
                else:
                    if not Ds["flag_sent_enough_item"]:
                        Ds["flag_sent_enough_item"]=True
                        self.logger.info("loop_count {0} sent enough record {1} start wait".
                                         format(Ds["worker_loop_count"],Ds["accumulate_record_sent_per_update"]))
                Ds["This_round_received_weight_fnwp"]=self.worker_check_update_weight()
                if len(Ds["This_round_received_weight_fnwp"])!=0:
                    #self.data.worker_reset_data()  # continue the original explore while weight reset
                    #current_len_buffer=self.i_train_buffer_to_server.get_len_train_buffer_to_server()
                    #self.i_train_buffer_to_server.empty_train_buffer_to_server() # keep the outbuffer while weight reset
                    #self.logger.info("at worker_loop_count {0} work reset data and  outbuffer left len {1}".
                    #                 format(Ds["worker_loop_count"],current_len_buffer ))
                    self.logger.info("loop_count {0} weight updated from {1} and worker start work".format(
                        Ds["worker_loop_count"],Ds["This_round_received_weight_fnwp"]))
                    Ds["accumulate_record_sent_per_update"]=0
                    Ds["flag_sent_enough_item"] = False
                if Ds["worker_loop_count"] %500==0 and Ds["worker_loop_count"]!=0:
                    self.logger.info("loop_count {0} accumulate record sent during this period {1} send buffer series {2}".
                                     format(Ds["worker_loop_count"],Ds["accumulate_record_sent_per_print"],self.i_bs.get_current() ))
                    Ds["accumulate_record_sent_per_print"] =0
                self.worker_name_pipe_cmd(Ds)
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
                s_, r, done, support_view_dic = i_env.step(a)
                self.data.l_done_flag[idx] = done
                a_onehot04,support_view_dic["old_ap"]=self.i_ac.I_A3C_worker_explorer(support_view_dic, self.data.l_ap[idx])
                self.clean_support_view_from_worker_to_server(support_view_dic)

                self.i_train_buffer_to_server.add_one_record(idx, s, a_onehot04, r, s_, done, support_view_dic)
                self.data.l_s[idx] = s_
        stacted_state = self.stack_l_state(self.data.l_s)
        self.data.l_a, self.data.l_ap, self.data.l_sv = self.i_wb.choose_action(stacted_state,"Explore")

    def clean_support_view_from_worker_to_server(self,support_view_dic):
        support_view_dic.pop("Flag_LastDay")
        support_view_dic.pop("Nprice")
        support_view_dic.pop("HFQRatio")
        support_view_dic.pop("Flag_Tradable")
        support_view_dic.pop("flag_all_period_explored")

        #support_view_dic.pop("Stock")  # add by Env need use by revorder state
        #support_view_dic.pop("DateI")  # add by Env need use by revorder state


        #support_view_dic.pop("action_return_message") #env
        #support_view_dic.pop("action_taken") #env
        #support_view_dic.pop("holding")    #av_state
        #remove "flag_force_sell" #support_view_dic.pop("flag_force_sell")    #av_state
        #support_view_dic.pop("old_ap")   #Explore_process

        # TD_intergrated will be add after add this fun
        #support_view_dic.pop(""SdisS_"")
        # support_view_dic.pop("_support_view_dic")   #TD_intergrated

        assert len(support_view_dic.keys())==6,support_view_dic.keys()

    def worker_send_buffer_brain(self):
        if self.i_train_buffer_to_server.get_len_train_buffer_to_server() > lc.num_train_record_to_brain:
            train_buffer_to_server = self.i_train_buffer_to_server.get_train_buffer_to_server()
            ## add sent explore brain train_count to each record to set
            length_sent = len(train_buffer_to_server)
            train_buffer_to_send = list(train_buffer_to_server)
            if lc.flag_record_buffer_to_server:
                self.i_record_send_to_server.saver(train_buffer_to_send)
            self.L_output.append([self.process_idx, self.i_bs.set_get_next(), train_buffer_to_send])
            self.i_train_buffer_to_server.empty_train_buffer_to_server()
            return length_sent
        return 0

    def worker_check_update_weight(self):
        if self.E_update_weight.is_set():
            weight_fnwp=self.D_share["weight_fnwp"]
            self.i_wb.load_weight(weight_fnwp)
            #self.logger.info("weight update from {0}".format(weight_fnwp))
            self.E_update_weight.clear()
            while not self.E_worker_work.is_set():
                time.sleep(1)
            return weight_fnwp
        else:
            return ""

    #def worker_name_pipe_cmd(self, worker_loop_count):
    def worker_name_pipe_cmd(self, Ds):
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
class Eval_process(client_base):
    def __init__(self, process_name, process_idx, data_name,learn_sl,SL_StartI, SL_EndI,L_output, E_stop,CLN_get_data):
        client_base.__init__(self, process_name, process_idx, learn_sl,SL_StartI, SL_EndI,L_output, None, E_stop, None, None,CLN_get_data)
        self.data_name = data_name
        flag_file_log=lc.l_flag_eval_log_file[process_idx]
        flag_screen_show=lc.l_flag_eval_log_screen[process_idx]
        self.logger= lcom.setup_logger(process_name,flag_file_log=flag_file_log, flag_screen_show=flag_screen_show)
        self.E_one_round_finished = Event()
        self.current_eval_count = lc.start_eval_count // lc.num_train_to_save_model + 1
        self.eval_loop_count = self.current_eval_count * lc.num_train_to_save_model

        self.init_data_eval()
        self.i_ac = actionOBOS(lc.train_action_type)
        self.initialed_prepare_summary_are_1ET=False

    def init_data_eval(self):
        self.data = client_datas(self.process_working_dir, self.data_name, self.stock_list, self.SL_StartI,self.SL_EendI,self.logger,self.CLN_get_data,called_by="Eval")
        self.i_are_ssdi=are_ssdi_handler(self.process_name, self.process_working_dir,self.logger)
        self.l_i_tran_id=[transaction_id(stock, start_id=0) for stock in self.stock_list]

    def run(self):
        import tensorflow as tf
        #lcom.setup_tf_logger(self.process_name)
        tf.random.set_seed(2)
        from nets import Explore_Brain, init_gc,init_virtual_GPU
        init_gc(lc)  #init_gc(lgc)
        setproctitle.setproctitle("{0}_{1}".format(lc.RL_system_name, self.process_name))
        self.logger.info("start at eval loop count {0}".format(self.eval_loop_count))
        assert lc.l_percent_gpu_core_for_eva[int(self.process_name[-1])]!=0.0, "Only Support GPU"
        virtual_GPU = init_virtual_GPU(lc.l_percent_gpu_core_for_eva[int(self.process_name[-1])])
        with tf.device(virtual_GPU):
            self.i_eb = locals()[lc.CLN_brain_explore]()
            while not self.E_stop.is_set():
                self.eval_name_pipe_cmd("Waiting for evaluation")
                model_weight_fnwp = self.eval_init_round()
                if len(model_weight_fnwp)==0:
                    time.sleep(30)
                    continue
                self.i_eb.load_weight(model_weight_fnwp)
                # these to make every eval round same
                random.seed(3)
                np.random.seed(3)
                while not self.E_one_round_finished.is_set() and not self.E_stop.is_set():
                    self.run_env_one_step()
                    self.eval_check_round_finished()
                    self.eval_name_pipe_cmd("Evaluation ongoing")
                else:
                    self.logger.info("finish eval_loop_count {0} ".format(self.eval_loop_count))
                    temp_save_fininshed__eval_loop_count=self.eval_loop_count
                    if self.E_one_round_finished.is_set():
                        self.eval_end_round()
                        if not self.initialed_prepare_summary_are_1ET:
                            #self.i_prepare_summary_are_1ET = summary_are_1ET(lc.RL_system_name, "Eval_0")
                            self.i_prepare_summary_are_1ET = ana_reward_data_A3C_worker_interface(lc.RL_system_name, self.process_name)
                            self.initialed_prepare_summary_are_1ET=True
                        #self.i_prepare_summary_are_1ET._get_are_summary_1ET(temp_save_fininshed__eval_loop_count)
                        self.i_prepare_summary_are_1ET._get_are_summary_1ET(temp_save_fininshed__eval_loop_count)
            self.logger.info("stopped")

    def run_env_one_step(self):
        for idx, i_env in enumerate(self.data.l_i_env):
            if not self.data.l_idx_valid_flag[idx]:
                continue
            if self.data.l_done_flag[idx]:
                try:
                    s, support_view_dic = i_env.reset()
                except Exception as e:
                    self.data.l_idx_valid_flag[idx]=False
                    self.logger.error("idx {0} {1} {2} at reset".format(idx,self.data.stock_list[idx],e))
                    continue
                self.__env_done_fun(idx,s, support_view_dic)
            else:
                s = self.data.l_s[idx]
                a = self.data.l_a[idx]
                ap =self.data.l_ap[idx]
                s_, r, done, support_view_dic = i_env.step(a)
                self.data.l_done_flag[idx] = done
                self.data.l_s[idx] = s_
                self.data.l_t[idx] += 1
                self.data.l_r[idx].append(r)
                trans_id = self.l_i_tran_id[idx].get_transaction_id(
                    flag_new_holding=True if support_view_dic["holding"] > 0 else False)
                actual_action = self.i_ac.I_A3C_worker_eval(support_view_dic)
                self.i_are_ssdi.in_round(self.data, idx, actual_action, ap, r, support_view_dic, trans_id)

        stacted_state = self.stack_l_state(self.data.l_s)
        self.data.l_a, self.data.l_ap,self.data.l_sv = self.i_eb.choose_action(stacted_state,"Eval")

    def __env_done_fun(self, idx, s, support_view_dic):
        self.data.l_s[idx] = s
        self.data.l_done_flag[idx] = False
        if support_view_dic["flag_all_period_explored"]:
            self.i_are_ssdi.round_save(self.data, idx, flag_finished=True)
            self.data.l_idx_valid_flag[idx] = False
            self.l_i_tran_id[idx].reset_flag_holding()  # to solve the new eval continue with the last trans_id
        else:
            if self.data.l_i_episode_init_flag[idx]:
                self.data.l_i_episode_init_flag[idx] = False
            else:
                #This is to reset trans_id at each reset
                #trans_id = self.l_i_tran_id[idx].get_transaction_id(flag_new_holding=False)
                self.l_i_tran_id[idx].get_transaction_id(flag_new_holding=False)
                self.i_are_ssdi.finish_episode(self.data, idx, flag_finished=True)


    #def find_model_surfix(self, eval_loop_count):
    #    l_model_fn = [fn for fn in os.listdir(lc.brain_model_dir) if "_T{0}.".format(eval_loop_count) in fn]
    #    if len(l_model_fn) == 2:
    #        regex = r'\w*(_\d{4}_\d{4}_T\d*).h5'
    #        match = re.search(regex, l_model_fn[0])
    #        return match.group(1)
    #    else:
    #        return None

    def eval_init_round(self):
        #found_model_surfix=self.find_model_surfix(self.eval_loop_count)
        found_model_surfix = find_model_surfix(lc.brain_model_dir,self.eval_loop_count)

        if found_model_surfix is None:
            return ""
        weight_fn = "{0}{1}.h5".format(lc.actor_weight_fn_seed, found_model_surfix)
        weight_fnwp=os.path.join(lc.brain_model_dir,weight_fn)
        self.i_are_ssdi.start_round(self.eval_loop_count)
        self.data.eval_reset_data()
        return weight_fnwp

    def eval_name_pipe_cmd(self, eval_state):
        l_status =["Waiting for evaluation", "Evaluation ongoing"]
        if eval_state not in l_status:
            self.logger.error("eval_state not in {0}".format(l_status))
            assert eval_state in l_status
        cmd_list = self.inp.check_input_immediate_return()
        if cmd_list is not None:
            if cmd_list[0][:-1] == "status":
                print("|||{0} for {1}|||".format(eval_state,self.eval_loop_count))
            else:
                print("Unknown command: {0} receive from name pipe: {1}".format(cmd_list, self.inp.np_fnwp))

    def eval_check_round_finished(self):
        if not any(self.data.l_idx_valid_flag):
            #self.logger.info("finish eval loop_count:{0}".format(self.eval_loop_count))
            self.E_one_round_finished.set()

    def eval_end_round(self):
        self.E_one_round_finished.clear()
        self.current_eval_count += 1
        self.eval_loop_count = self.current_eval_count * lc.num_train_to_save_model
        self.data.eval_reset_data()
        self.logger.info("set self.current_eval_count:{0}".format(self.current_eval_count))
