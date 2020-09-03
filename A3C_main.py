from multiprocessing import Process,Manager,log_to_stderr,get_logger
import os,time, random, sys
import numpy as np
import config as sc
import logger_comm  as lcom
import pipe_comm as pcom
#import Stocklist_comm as scom
import DBI_Base as  DBI_Base
import Buffer_comm as bcom
import av_state as av_state
import env as env
import logging
import setproctitle
from miscellaneous import getselected_item_name, create_system,remove_system_sub_dirs,start_tensorboard,copy_between_two_machine,create_eval_system
#from miscellaneous import sanity_check_config
from A3C_workers import Explore_process,Eval_process, init_A3C_worker
from A3C_brain import Train_Process, init_A3C_brain
import DB_main
class main(Process):
    def __init__(self, argv):
        Process.__init__(self)
        random.seed(2)
        np.random.seed(2)

        l_fun_menu = ["create_train", "remote_copy", "learn", "eval","create_eval","DataBase"]
        fun_selected = getselected_item_name(l_fun_menu, colum_per_row=1) if len(argv)==0 else argv[0]
        if fun_selected == "learn":
            self.run = self.main_learn
            system_name = argv[1] if len(argv) == 2 else getselected_item_name(os.listdir(sc.base_dir_RL_system))
            if input("Clean all the sub dir Enter Yes or no: ")== "Yes":
                remove_system_sub_dirs(os.path.join(sc.base_dir_RL_system, system_name))
            self.init_system(system_name)
        elif fun_selected == "eval":
            self.run = self.main_eval_only
            system_name = argv[1] if len(argv) == 2 else getselected_item_name(os.listdir(sc.base_dir_RL_system))

            self.init_system(system_name)
        elif fun_selected == "create_eval":
            create_eval_system()
        elif fun_selected == "create_train":
            create_system(sc.base_dir_RL_system)
        elif fun_selected == "remote_copy":
            #copy_between_two_machine().copy_between_two_machine("192.168.199.100","")
            copy_between_two_machine().copy_between_two_machine()
        elif fun_selected == "DataBase":
            DB_main.main(argv[1:])
        else:
            raise ValueError("not support arg {0}".format(argv[1]))

    def init_system(self, system_name):
        param_fnwp = os.path.join(sc.base_dir_RL_system, system_name, "config.json")
        if not os.path.exists(param_fnwp):
            raise ValueError("{0} does not exisit".format(param_fnwp))
        lgc=sc.gconfig()
        lgc.read_from_json(param_fnwp)
        self.lgc=lgc
        init_A3C_brain(lgc)
        init_A3C_worker(lgc)
        for pack in [env, pcom, lcom, bcom, av_state]:
            pack.init_gc(lgc)
        self.iSL=DBI_Base.StockList(lgc.SLName)
        logging.getLogger().setLevel(logging.INFO)
        #set root level to Debug, other wise, no matter the setting on sub level, it will only show waring and error
        self.Manager = Manager()
        self.process_name="{0}_{1}".format(self.lgc.RL_system_name, "main")
        self.logger = lcom.setup_logger(self.process_name, flag_file_log=True, flag_screen_show=True)
        #set multiprocess logging
        log_to_stderr()
        logger = get_logger()
        logger.setLevel(logging.INFO)

    def main_learn(self):
        setproctitle.setproctitle(self.process_name)
        inp = pcom.name_pipe_cmd("main_learn")
        self.learn_init()
        tensorboard_dir = os.path.join(sc.base_dir_RL_system, self.lgc.RL_system_name, "tensorboard")
        tensorboard_process = Process(target=start_tensorboard, args=(self.lgc.tensorboard_port, tensorboard_dir,))
        tensorboard_process.start()

        while True:
            cmd_list = inp.check_input_immediate_return()
            if cmd_list is not None:
                if cmd_list[0][:-1] == "stop":
                    break
                else:
                    print("Unknown command: {0} from name pipe: {1}".format(cmd_list, inp.np_fnwp))
            assert self.TrainP.is_alive(), "Train_process not alive"
            time.sleep(600)

        self.join_all()
        tensorboard_process.join()
    def main_eval_only(self):
        setproctitle.setproctitle("{0}_{1}".format(self.lgc.RL_system_name, "main_eval_only"))
        inp = pcom.name_pipe_cmd("main_eval_only")
        self.eval_init()
        while True:
            cmd_list = inp.check_input_immediate_return()
            if cmd_list is not None:
                if cmd_list[0][:-1] == "stop":
                    break
                else:
                    print("Unknown command: {0} from name pipe: {1}".format(cmd_list, inp.np_fnwp))
            time.sleep(600)
        self.join_eval()

    '''
    def start_brain_process(self):
        LL_output                   =   [self.Manager.list() for _ in range (self.lgc.num_workers)]
        E_stop_brain                =   self.Manager.Event()
        LE_update_weight            =   [self.Manager.Event() for _ in range (self.lgc.num_workers)]
        LE_worker_work              =   [self.Manager.Event() for _ in range(self.lgc.num_workers)]
        D_share                     =   self.Manager.dict()
        TrainP = Train_Process(self.lgc.server_process_name_seed, 0, LL_output,D_share,
                               E_stop_brain,LE_update_weight,LE_worker_work)
        TrainP.daemon=True
        TrainP.start()
        return TrainP, LL_output, D_share, E_stop_brain, LE_update_weight,LE_worker_work

    def start_worker_process(self, idx, Manager, data_name,learn_sl, SL_StartI, SL_EndI,L_output, D_share, E_update_weight,E_worker_work ):
        p_name = "{0}_{1}".format(self.lgc.client_process_name_seed, idx)
        E_stop = Manager.Event()
        ExploreP=Explore_process(p_name, idx, data_name,learn_sl,SL_StartI, SL_EndI,L_output, D_share, E_stop, E_update_weight, E_worker_work)
        ExploreP.daemon = True
        ExploreP.start()
        return ExploreP,E_stop

    def start_eval_process(self, idx, Manager,data_name,unl_eval_sl,SL_StartI, SL_EndI):
        E_stop_eval = Manager.Event()
        p_name = "{0}_{1}".format(self.lgc.eval_process_seed, idx)
        EvalP=Eval_process(p_name, idx, data_name,unl_eval_sl,SL_StartI, SL_EndI,None, E_stop_eval)
        EvalP.daemon = True
        EvalP.start()
        return EvalP,E_stop_eval
    '''

    def learn_init(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.lgc.get_CUDA_VISIBLE_DEVICES_str(self.lgc.Brian_core)

        self.LL_output                   =   [self.Manager.list() for _ in range (self.lgc.num_workers)]
        self.E_stop_brain                =   self.Manager.Event()
        self.LE_update_weight            =   [self.Manager.Event() for _ in range (self.lgc.num_workers)]
        self.LE_worker_work              =   [self.Manager.Event() for _ in range(self.lgc.num_workers)]
        self.D_share                     =   self.Manager.dict()
        self.TrainP = Train_Process(self.lgc.server_process_name_seed, 0, self.LL_output,self.D_share,
                               self.E_stop_brain,self.LE_update_weight,self.LE_worker_work)
        self.TrainP.daemon=True
        self.TrainP.start()

        self.l_Process_worker = []
        self.l_E_stop_worker = []
        for idx in range(self.lgc.num_workers):
            os.environ["CUDA_VISIBLE_DEVICES"] = self.lgc.get_CUDA_VISIBLE_DEVICES_str(self.lgc.l_work_core[idx])

            SL_idx, SL_StartI, SL_EndI = self.lgc.l_train_SL_param[idx]
            flag,sl_explore=self.iSL.get_sub_sl("Train", SL_idx)
            assert flag, "Get Stock list {0} tag=\"Train\" index={1}".format(self.lgc.SLName,idx)
            #ExploreP, E_stop_worker = self. start_worker_process(idx,self.Manager, self.lgc.data_name,learn_divided_sl,SL_StartI, SL_EndI,
            #                self.LL_output[idx],self.D_share,self.LE_update_weight[idx], self.LE_worker_work[idx] )

            p_name = "{0}_{1}".format(self.lgc.client_process_name_seed, idx)
            E_stop = self.Manager.Event()
            ExploreP = Explore_process(p_name, idx, self.lgc.data_name, sl_explore, SL_StartI, SL_EndI,
                    self.LL_output[idx], self.D_share, E_stop,self.LE_update_weight[idx], self.LE_worker_work[idx])
            ExploreP.daemon = True
            ExploreP.start()

            self.l_E_stop_worker.append(E_stop)
            self.l_Process_worker.append(ExploreP)
        self.eval_init()

    def eval_init(self):
        if self.lgc.flag_eval_unlearn:
            self.l_eval_unlearn_process=[]
            self.l_E_stop_eval_unlearn=[]
            for idx in range(self.lgc.eval_num_process):
                os.environ["CUDA_VISIBLE_DEVICES"] = self.lgc.get_CUDA_VISIBLE_DEVICES_str(self.lgc.l_eval_core[idx])
                SL_idx,SL_StartI, SL_EndI=self.lgc.l_eval_SL_param[idx]
                flag,sl_eval=self.iSL.get_sub_sl("Eval",SL_idx)
                assert flag,  "Get Stock list {0} tag=\"Eval\" index={1}".format(self.lgc.SLName,idx)
                #EvalP, E_stop_eval = self.start_eval_process(idx, self.Manager, self.lgc.data_name,l_eval_sl, SL_StartI, SL_EndI)

                E_stop_eval = self.Manager.Event()
                p_name = "{0}_{1}".format(self.lgc.eval_process_seed, idx)
                EvalP = Eval_process(p_name, idx, self.lgc.data_name,sl_eval, SL_StartI, SL_EndI, None, E_stop_eval)
                EvalP.daemon = True
                EvalP.start()
                self.l_eval_unlearn_process.append(EvalP)
                self.l_E_stop_eval_unlearn.append(E_stop_eval)


    def join_all(self):
        if self.l_Process_worker is not None:
            for E_stop_worker, process_worker in zip(self.l_E_stop_worker, self.l_Process_worker):
                E_stop_worker.set()
                process_worker.join()
        if  self.E_stop_brain is not None and self.TrainP is not None:
            self.E_stop_brain.set()
            self.TrainP.join()
        self.join_eval()
    def join_eval(self):
        if self.lgc.flag_eval_unlearn:
            for idx in range(self.lgc.eval_num_process):
                self.l_E_stop_eval_unlearn[idx].set()
                self.l_eval_unlearn_process[idx].join()

if __name__ == '__main__':
    main(sys.argv[1:]).run()
