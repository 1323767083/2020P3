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
from action_comm import actionOBOS
import DBI_Base
from miscellaneous import  find_model_surfix
class Client_Datas_Common:
    def __init__(self, lc, data_name, stock_list, StartI, EndI, logger, CLN_get_data):
        self.stock_list = stock_list
        self.data_name = data_name
        self.logger = logger
        self.l_i_env = []
        self.l_done_flag = [True for _ in self.stock_list]
        self.l_s = [[] for _ in self.stock_list]
        self.l_a = [0 for _ in self.stock_list]
        self.l_ap = [[] for _ in self.stock_list]
        self.l_sv = [0.0 for _ in self.stock_list]


    def stack_l_state(self, l_state):
        l_lv, l_sv, l_av = [], [], []
        for state in l_state:
            lv, sv, av = state
            l_lv.append(lv)
            l_sv.append(sv)
            l_av.append(av)
        stack_state = [np.concatenate(l_lv, axis=0), np.concatenate(l_sv, axis=0), np.concatenate(l_av, axis=0)]
        return stack_state

class Client_Datas_Explore(Client_Datas_Common):
    def __init__(self, lc, data_name, stock_list, StartI, EndI, logger, CLN_get_data, called_by):
        Client_Datas_Common.__init__(self,lc, data_name, stock_list, StartI, EndI, logger, CLN_get_data)
        self.called_by=called_by
        for stock in self.stock_list:
            i_env = globals()[lc.CLN_simulator](self.data_name, stock, StartI, EndI, CLN_get_data, lc, self.called_by)
            self.l_i_env.append(i_env)

    def worker_reset_data(self):
        self.l_sv = [0.0 for _ in range(len(self.stock_list))]
        self.l_done_flag = [True for _ in self.stock_list]  # this is to avoid unfinished reset

class Client_Datas_Eval(Client_Datas_Common):
    def __init__(self, lc,process_working_dir, data_name,stock_list, StartI, EndI,logger,CLN_get_data, called_by):
        Client_Datas_Common.__init__(self, lc, data_name, stock_list, StartI, EndI, logger, CLN_get_data)
        self.called_by=called_by
        for stock in self.stock_list:
            i_env = globals()[lc.CLN_simulator](self.data_name, stock, StartI, EndI, CLN_get_data, lc, self.called_by)
            self.l_i_env.append(i_env)
        self.process_working_dir = process_working_dir
        self.stock_working_dir = []
        for stock in self.stock_list:
            one_stock_working_dir = os.path.join(self.process_working_dir, stock)
            if not os.path.exists(one_stock_working_dir): os.mkdir(one_stock_working_dir)
            self.stock_working_dir.append(one_stock_working_dir)
        self.l_log_a_r_e = [[] for _ in self.stock_list]
        self.l_i_episode = [0 for _ in self.stock_list]
        self.l_i_episode_init_flag = [True for _ in self.stock_list]  # for log purpose
        self.l_t = [0 for _ in self.stock_list]
        self.l_r = [[] for _ in self.stock_list]
        self.l_idx_valid_flag = [True for _ in self.stock_list]  # for eval quiting

    def eval_reset_data(self):
        # this is add to record the state value for are currently only used in eval process
        self.l_sv = [0.0 for _ in range(len(self.stock_list))]
        self.l_done_flag = [True for _ in self.stock_list]   # this is ensure after this fun called, the first round will call reset data
        self.l_idx_valid_flag = [True for _ in range(len(self.stock_list))]  # for eval quiting
        self.l_i_episode_init_flag = [True for _ in range(len(self.stock_list))]  # for log purpose
        self.l_i_episode=[0 for _ in range(len(self.stock_list))]
        self.l_t = [0 for _ in self.stock_list]  # as after round save, but worker still work on and l_t will continue to add
        for idx,_ in enumerate(self.stock_list):
            if len (self.l_r[idx])!=0:
                self.logger.error("len (self.l_r[{0}])!=0".format(idx))
                assert len(self.l_r[idx]) == 0
            if len(self.l_log_a_r_e[idx]) != 0:
                self.logger.error("len(self.l_log_a_r_e[{0}]) != 0".format(idx))
                assert len(self.l_log_a_r_e[idx]) == 0
class WR_handler:
    def __init__(self, lc,process_name, process_group_name, logger):
        self.lc=lc
        self.process_name=process_name
        self.logger=logger
        self.log_WR_column=["BW","BZ","BR","NW","NZ","NR","NA"]
        self.log_WRs=[]
        self.log_PAs=[]

        self.WR_working_dir,self.PA_working_dir="",""
        for tag in ["WR","PA"]:
            wdir= lc.system_working_dir
            for sub_dir in ["Classifiaction",tag, process_group_name]:
                wdir = os.path.join(wdir, sub_dir)
                if not os.path.exists(wdir): os.mkdir(wdir)
            setattr(self,"{0}_working_dir".format(tag), wdir)


    def save(self, eval_loop_count):
        fnwp=os.path.join(self.WR_working_dir, "ET{0}.csv".format(eval_loop_count))
        pd.DataFrame(self.log_WRs, columns=self.log_WR_column).to_csv(fnwp,index=False)

        fnwp=os.path.join(self.PA_working_dir, "ET{0}.csv".format(eval_loop_count))
        pd.DataFrame(self.log_PAs).to_csv(fnwp,index=False, float_format="%.2f")

        self.reset_logs()
    def reset_logs(self):
        del self.log_WRs[:]
        del self.log_PAs[:]

    def Fabricate_PA(self, profit_log):
        return profit_log

    def Fabricate_WR(self, l_WR):
        # ["BW", "BZ", "BR", "NW", "NZ", "NR", "NA"]
        return [l_WR.count(0),l_WR.count(1),l_WR.count(2),l_WR.count(10),l_WR.count(11),l_WR.count(12),l_WR.count(-1)]

    def add_log(self,ll_log):
        l_WR,l_PA=ll_log
        self.log_WRs.append(self.Fabricate_WR(l_WR))
        self.log_PAs.append(self.Fabricate_PA(l_PA))


class are_ssdi_handler:
    def __init__(self, lc,process_name, process_working_dir, logger):
        self.lc=lc
        self.process_name=process_name
        self.process_working_dir=process_working_dir
        self.ongoing_save_count = -1
        self.logger=logger
        self.log_are_column=["action", "reward", "episode", "day", "action_result"]

        for ap_idx in range(lc.train_num_action):
            self.log_are_column.append("p{0}".format(ap_idx))
        self.log_are_column.append("state_value")
        self.log_are_column.extend(["holding", "trade_Nprice", "trans_id"])

    def _eval_save(self,data,idx,eval_count):
        if eval_count ==0:
            del data.l_log_a_r_e[idx][:]
            return
        log_a_r_e_fn="{0}_T{1}.csv".format(self.lc.log_a_r_e_fn_seed,eval_count)
        log_a_r_e_fn_fnwp=os.path.join(self.process_working_dir,data.stock_list[idx],log_a_r_e_fn )
        df_log = pd.DataFrame(data.l_log_a_r_e[idx],columns=self.log_are_column)
        df_log.to_csv(log_a_r_e_fn_fnwp, index=False, float_format='%.4f')
        del data.l_log_a_r_e[idx][:]
        return

    def round_save(self, data, idx, flag_finished):
        self.finish_episode(data, idx, flag_finished)
        self.logger.info("{0} round saved".format(data.stock_list[idx]))
        if self.ongoing_save_count ==-1:
            self.logger.error("self.ongoing_save_count is -1")
            assert self.ongoing_save_count !=-1
        self._eval_save(data, idx, self.ongoing_save_count)

    def finish_episode(self, data,idx, flag_finished):
        if len(data.l_r[idx])!=0:
            r_sum = data.l_r[idx][-1]
        else:
            r_sum=0
        self.logger.info("stock:{0} episode:{1} period_len:{2} reward:{3:.2f} {4} episode add to record"
                        .format(data.stock_list[idx], data.l_i_episode[idx],data.l_t[idx], r_sum,
                                "finished" if flag_finished else "unfinished"))
        data.l_i_episode[idx] += 1
        data.l_t[idx] = 0
        del data.l_r[idx][:]

    def start_round(self, save_count):
        self.ongoing_save_count=save_count

    def in_round(self, data, idx, a, ap, r, sv_dic, trans_id,holding_flag):
        item=[a, r, data.l_i_episode[idx], sv_dic["DateI"], sv_dic["action_return_message"]]
        for ap_idx in range(self.lc.train_num_action):
            item.append(ap[ap_idx])
        item.append(data.l_sv[idx])
        item.extend([holding_flag, sv_dic["Nprice"], trans_id])
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

