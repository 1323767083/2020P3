from multiprocessing import Process, Pipe,Manager, Event,log_to_stderr,get_logger
import os,time, random, sys, pickle,re, shutil
import matplotlib.pyplot as plt
import setproctitle
import datetime as dt
import numpy as np
import pandas as pd
import logging
import config as sc
import logger_comm  as lcom
import pipe_comm as pcom
import Stocklist_comm as scom
import Buffer_comm as bcom
import env as env

from Buffer_comm import buffer_to_train,buffer_series
import A3C


class C_debug_reward(Process):
    def __init__(self, argv):
        Process.__init__(self)
        random.seed(2)
        np.random.seed(2)
        assert len(argv) == 4, "system_name stock quit_date, saved_train_count"
        system_name         = argv[0]
        self.stock =          argv[1]
        self.quit_date      = argv[2]
        self.train_count    = argv[3]
        param_fnwp = os.path.join(sc.base_dir_RL_system, system_name, "config.json")
        if not os.path.exists(param_fnwp):
            raise ValueError("{0} does not exisit".format(param_fnwp))
        self.lgc=sc.gconfig()
        self.lgc.read_from_json(param_fnwp)
        #set debug inform to config
        self.lgc.CLN_TDmemory="TD_memory_2S_nc"

        A3C.init_gc(self.lgc)
        for pack in [env,pcom,lcom,scom,bcom]:
            pack.init_gc(self.lgc)

        logging.getLogger().setLevel(logging.INFO)
        #set root level to Debug, other wise, no matter the setting on sub level, it will only show waring and error

        self.Manager = Manager()
        self.process_name="{0}_{1}".format(self.lgc.RL_system_name, "debug_main")
        self.inp = pcom.name_pipe_cmd(self.process_name)
        self.logger = lcom.setup_logger(self.process_name, flag_file_log=True, flag_screen_show=True)
        #set multiprocess logging
        log_to_stderr()
        logger = get_logger()
        logger.setLevel(logging.INFO)

        self.i_bs = buffer_series()

        self.tb = bcom.train_buffer(self.lgc.Buffer_nb_Features)

        self.dump_record_fnwp=os.path.join(self.lgc.record_train_buffer_dir,"{0}_tb.pickle".format(self.stock))
        #self.train_push_many = self.tb.train_push_many
        #self.get_buffer_size = self.tb.get_buffer_size




    def run(self):

        setproctitle.setproctitle(self.process_name)
        stock_list=[self.stock]
        L_output                   =   self.Manager.list()
        E_update_weight            =   self.Manager.Event()
        E_worker_work              =   self.Manager.Event()
        D_share                    =   self.Manager.dict()
        E_stop                     =   self.Manager.Event()
        p_name = "{0}_{1}".format("debug_explore", 0)

        ExploreP=A3C.Explore_process(p_name, 0, stock_list,L_output, D_share, E_stop, E_update_weight, E_worker_work)
        ExploreP.daemon = True
        ExploreP.start()

        self.init_explore_weight_update(self.train_count, E_update_weight,E_worker_work,D_share)
        flag_quit= False
        while not flag_quit:
            flag_quit=self.get_train_records(L_output)
            cmd_list = self.inp.check_input_immediate_return()
            if cmd_list is not None:
                if cmd_list[0][:-1] == "save":
                    flag_quit=True
                    print "command: {0} receive from name pipe: {1} quit to save".format(cmd_list, self.inp.np_fnwp)
                else:
                    print "command: {0} receive from name pipe: {1}, unknown".format(cmd_list, self.inp.np_fnwp)
            time.sleep(1)

        E_stop.set()
        ExploreP.join()
        self.save_record_buffer()

    def find_model_surfix(self, eval_loop_count):
        l_model_fn = [fn for fn in os.listdir(self.lgc.brain_model_dir) if "_T{0}.".format(eval_loop_count) in fn]
        if len(l_model_fn) == 2:
            regex = r'\w*(_\d{4}_\d{4}_T\d*).h5py'
            match = re.search(regex, l_model_fn[0])
            return match.group(1)
        else:
            return None
    def init_explore_weight_update(self, train_count, E_update,E_worker_work, D_share ):
        found_model_surfix = self.find_model_surfix(train_count)
        if  found_model_surfix is None:
            raise ValueError ( "can not find init train count saved weight {0}".format(train_count))
        else:
            actor_weight_fn = "{0}{1}.h5py".format(self.lgc.actor_weight_fn_seed, found_model_surfix)
            last_saved_weights_fnwp = os.path.join(self.lgc.brain_model_dir, actor_weight_fn)
        self.logger.info("init_weights ready at {0} start worker weight update".format(last_saved_weights_fnwp))
        D_share["weight_fnwp"]=last_saved_weights_fnwp
        E_update.set()
        while not E_update.is_set():
            time.sleep(1)
        E_worker_work.set()
        self.logger.info("finish explore init weight update and start worker work")

    def get_train_records(self, L_output):
        flag_quit=False
        if len(L_output)!=0:
            input_item=L_output.pop(0)
            worker_idx, bs, input_buffer=input_item
            if self.i_bs.valify(bs):
                self.i_bs.set(bs)
            else:
                self.i_bs.set(bs)
                self.logger.warn("from worker {0} received wrong order last series {1} this serise {2}"
                                 .format(worker_idx, self.i_bs.get_current(), bs))

            for buffer_item in input_buffer:
                #to_clean_dic = buffer_item[5][0,0]["_support_view_dic"]
                date_s=buffer_item[5][0,0]["date"]

                if date_s >= self.quit_date:
                    print "handling last {0} buffer to train".format(date_s)
                    flag_quit = True
                else:
                    print "\thandling {0} buffer to train".format(date_s)
            self.tb.train_push_many(input_buffer)

        return flag_quit

    def save_record_buffer(self):
        pickle.dump(self.tb.train_queue, open(self.dump_record_fnwp,"w"))

    def clean_support_view_from_worker_to_server(self,support_view_dic):
        support_view_dic.pop("action_return_message")
        support_view_dic.pop("action_taken")
        support_view_dic.pop("holding")
        support_view_dic.pop("idx_in_period")
        support_view_dic.pop("last_day_flag")
        support_view_dic.pop("period_idx")
        #support_view_dic.pop("potential_profit")
        support_view_dic.pop("stock_SwhV1")
        support_view_dic.pop("this_trade_day_Nprice")
        support_view_dic.pop("this_trade_day_hfq_ratio")
        # only keep 'date' 'stock' 'old_ap'

'''
def Ana_reward(argv):
    i=debug_reward(argv)
    print "here"
    if not os.path.exists(i.dump_record_fnwp):
        i.run()
    a = pickle.load(open(i.dump_record_fnwp, "r"))
    b = [item[0, 0] for item in a[4]]
    plt.plot(b)
    plt.show()
'''

if __name__ == '__main__':
    if sys.argv[1].startswith("C_"):
        globals()[sys.argv[1]](sys.argv[2:]).run()
    else:
        globals()[sys.argv[1]](sys.argv[2:])


    #debug_reward(sys.argv[1:]).run()


'''
import pickle
import matplotlib.pyplot as plt
fnwp="/home/rdchujf/n_workspace/RL/htryc/record_tb/SH600177_tb.pickle"
a=pickle.load(open(fnwp,"r"))
b=[item[0,0] for item in a[4]]
d=[item[0,0]["R_before_add_to_memory"] for item in a[9] if item[0,0]["date"]<"20170101"]
e=[item[0,0]["R_after_add_to_memory"] for item in a[9]]


fnwp="/home/rdchujf/n_workspace/RL/htryc/record_tb/SH600008_tb.pickle"
a=pickle.load(open(fnwp,"r"))
b=[item[0,0] for item in a[4]]
plt.plot(b)
'''