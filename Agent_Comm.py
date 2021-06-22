import os,re, random,time,setproctitle,pickle
import pandas as pd
import numpy as np
from multiprocessing import Process,Event
import pipe_comm as pcom
import logger_comm  as lcom
import config as sc
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
