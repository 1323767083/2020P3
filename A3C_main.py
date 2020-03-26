from multiprocessing import Process,Manager,log_to_stderr,get_logger
import os,time, random, sys
import numpy as np
import config as sc
import logger_comm  as lcom
import pipe_comm as pcom
import Stocklist_comm as scom
import Buffer_comm as bcom

import env as env
import logging
import setproctitle
from miscellaneous import getselected_item_name, create_system,remove_system_sub_dirs,start_tensorboard,copy_between_two_machine,create_eval_system
#from miscellaneous import sanity_check_config
from A3C_workers import Explore_process,Eval_process, init_A3C_worker
from A3C_brain import Train_Process, init_A3C_brain
class main(Process):
    def __init__(self, argv):
        Process.__init__(self)
        random.seed(2)
        np.random.seed(2)

        l_fun_menu = ["create_train", "remote_copy", "learn", "eval","create_eval"]
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
            create_eval_system("T5_V2_")
        elif fun_selected == "create_train":
            create_system(sc.base_dir_RL_system)
        elif fun_selected == "remote_copy":
            copy_between_two_machine().copy_between_two_machine("192.168.199.100","")
        else:
            raise ValueError("not support arg {0}".format(argv[1]))

    def init_system(self, system_name):
        param_fnwp = os.path.join(sc.base_dir_RL_system, system_name, "config.json")
        if not os.path.exists(param_fnwp):
            raise ValueError("{0} does not exisit".format(param_fnwp))
        lgc=sc.gconfig()
        lgc.read_from_json(param_fnwp)
        #sanity_check_config(lgc)
        self.lgc=lgc
        #manually_select_GPU='s'
        #while manually_select_GPU not in ['n','0', '1']:
        #    manually_select_GPU=raw_input("Manually set GPU used (0) (1) or (n)ot, following the config:")
        #else:
        #    if manually_select_GPU in ['0','1']:
        #        lgc.Brian_core="GPU_{0}".format(manually_select_GPU)
        #        lgc.l_work_core=["GPU_{0}".format(manually_select_GPU) for _ in lgc.l_work_core]
        #        lgc.l_eval_core=["GPU_{0}".format(manually_select_GPU) for _ in lgc.l_eval_core]
        #        print "lgc.Brian_core, lgc.l_work_core, lgc.l_eval_core set to ", "GPU_{0}".format(manually_select_GPU)
        #manually_set_tensor_board_port= input("Manually set tensorboard port, or (0) for following config:")
        #if manually_set_tensor_board_port != 0:
        #    lgc.tensorboard_port=manually_set_tensor_board_port
        #    print "tensorboard port set to {0}".format(manually_set_tensor_board_port)
        init_A3C_brain(lgc)
        init_A3C_worker(lgc)
        for pack in [env,pcom,lcom,scom,bcom]:
            pack.init_gc(lgc)

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

    def start_worker_process(self, idx, Manager, data_name,learn_sl, L_output, D_share, E_update_weight,E_worker_work ):
        p_name = "{0}_{1}".format(self.lgc.client_process_name_seed, idx)
        E_stop = Manager.Event()
        ExploreP=Explore_process(p_name, idx, data_name,learn_sl,L_output, D_share, E_stop, E_update_weight, E_worker_work)
        ExploreP.daemon = True
        ExploreP.start()
        return ExploreP,E_stop

    def start_eval_process(self, idx, Manager,data_name,unl_eval_sl):
        E_stop_eval = Manager.Event()
        p_name = "{0}_{1}".format(self.lgc.eval_process_seed, idx)
        EvalP=Eval_process(p_name, idx, data_name,unl_eval_sl,None, E_stop_eval)
        EvalP.daemon = True
        EvalP.start()
        return EvalP,E_stop_eval

    def learn_init(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.lgc.get_CUDA_VISIBLE_DEVICES_str(self.lgc.Brian_core)
        i_d_sl=scom.divide_sl_to_process(self.lgc.data_name)
        learn_divided_sl = i_d_sl.get_sl_full_divided(self.lgc.train_data_base, self.lgc.train_data_index, self.lgc.num_workers)

        self.TrainP, self.LL_output, self.D_share, self.E_stop_brain, self.LE_update_weight, self.LE_worker_work,  \
            = self.start_brain_process()
        self.l_Process_worker = []
        self.l_E_stop_worker = []
        for idx in range(self.lgc.num_workers):
            os.environ["CUDA_VISIBLE_DEVICES"] = self.lgc.get_CUDA_VISIBLE_DEVICES_str(self.lgc.l_work_core[idx])
            ExploreP, E_stop_worker = self. start_worker_process(idx,self.Manager, self.lgc.data_name,learn_divided_sl[idx],
                            self.LL_output[idx],self.D_share,self.LE_update_weight[idx], self.LE_worker_work[idx] )
            self.l_E_stop_worker.append(E_stop_worker)
            self.l_Process_worker.append(ExploreP)
        self.eval_init()

    def eval_init(self):
        if self.lgc.flag_eval_unlearn:
            l_eval_sl=scom.prepare_eval_sl(self.lgc.l_eval_data_name, self.lgc.eval_data_base,self.lgc.eval_data_index,
                                           self.lgc.eval_num_stock_per_process)
            self.l_eval_unlearn_process=[]
            self.l_E_stop_eval_unlearn=[]
            for idx in range(self.lgc.eval_num_process):
                os.environ["CUDA_VISIBLE_DEVICES"] = self.lgc.get_CUDA_VISIBLE_DEVICES_str(self.lgc.l_eval_core[idx])
                EvalP, E_stop_eval = self.start_eval_process(idx, self.Manager, self.lgc.l_eval_data_name[idx],l_eval_sl[idx])
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
