from multiprocessing import Process,Manager,log_to_stderr,get_logger
import os,time, random, sys, shutil
import numpy as np
import config as sc
import logger_comm  as lcom
import pipe_comm as pcom
import DBI_Base as  DBI_Base
import logging
import setproctitle
from miscellaneous import getselected_item_name, create_system,remove_system_sub_dirs,start_tensorboard,copy_between_two_machine
from Agent_Explore import AgentMain, Agent_Sub
from Agent_Eval import EvalMain,EvalSub
from Brain import Train_Process
import DB_main

class Remove_DNFN:
    SubDNs,SubDN_Tags,FNPip_Tags,FNLog_Tags=[],[],[],[]
    def __init__(self, lc):
        self.lc=lc
        self.system_dir = os.path.join(sc.base_dir_RL_system, self.lc.RL_system_name)
    def RemoveDN(self):
        ToRemove_SubDNs=[]
        ToRemove_SubDNs.extend(self.SubDNs)
        for Tag in self.SubDN_Tags:
            ToRemove_SubDNs.extend([dn for dn in os.listdir(self.system_dir) if  Tag in dn])
        for sub_dir in ToRemove_SubDNs:
            directory_to_remove = os.path.join(self.system_dir, sub_dir)
            if os.path.exists(directory_to_remove):
                shutil.rmtree(directory_to_remove)
                print("Removed ", directory_to_remove)
    def RemoveFN(self):
        ToRemove_fnwps=[]
        for sub_dir, Tags in zip(["name_pipe", "log"], [self.FNPip_Tags, self.FNLog_Tags]):
            dnwp = os.path.join(self.system_dir, sub_dir)
            for Tag in Tags:
                if os.path.exists(dnwp):
                    ToRemove_fnwps.extend([os.path.join(dnwp, fn) for fn in os.listdir(dnwp) if Tag in fn])
        for fnwp in ToRemove_fnwps:
            os.remove(fnwp)
            print ("Removed {0}".format(fnwp))
    def Remove(self):
        self.RemoveDN()
        self.RemoveFN()

class Remove_DNFN_Explore_Agent(Remove_DNFN):
    SubDNs,SubDN_Tags,FNPip_Tags,FNLog_Tags=[],["ExploreAgent"],["ExploreAgent"],["ExploreAgent"]

class Remove_DNFN_Eval_Agent(Remove_DNFN):
    SubDNs,SubDN_Tags,FNPip_Tags,FNLog_Tags=["analysis","CC","WR"],["EvalAgent"],["EvalAgent"],["EvalAgent"]

class Remove_DNFN_Main(Remove_DNFN):
    SubDNs, SubDN_Tags, FNPip_Tags, FNLog_Tags = [], [], [], []
    def __init__(self, lc,fun_label):
        Remove_DNFN.__init__(self,lc)
        if "learn" in fun_label:
            self.SubDNs.extend(["model", "record_state", "tensorboard", "record_tb"])
            self.SubDN_Tags.append(self.lc.server_process_name_seed)
            self.FNPip_Tags.append(self.lc.server_process_name_seed)
            self.FNLog_Tags.append(self.lc.server_process_name_seed)
        self.SubDN_Tags.append(fun_label)
        self.FNPip_Tags.append(fun_label)
        self.FNLog_Tags.append(fun_label)

class main(Process):
    def __init__(self, argv):
        Process.__init__(self)
        random.seed(2)
        np.random.seed(2)

        l_fun_menu = ["learn", "eval","learneval","create_train", "remote_copy", "DataBase"]
        fun_selected = getselected_item_name(l_fun_menu, colum_per_row=1,flag_sort=False) if len(argv)==0 else argv[0]
        if fun_selected in ["learn", "eval","learneval"]:
            self.fun_label = fun_selected
            self.system_name = argv[1] if len(argv) == 2 else getselected_item_name(os.listdir(sc.base_dir_RL_system))
            param_fnwp = os.path.join(sc.base_dir_RL_system, self.system_name, "config.json")
            if not os.path.exists(param_fnwp):
                raise ValueError("{0} does not exisit".format(param_fnwp))
            self.lc = sc.gconfig()
            self.lc.read_from_json(param_fnwp)

            if input("Clean all the sub dir Enter Yes or no: ") == "Yes":
                Remove_DNFN_Main(self.lc,fun_selected).Remove()
                if "learn" in fun_selected:
                    Remove_DNFN_Explore_Agent(self.lc).Remove()
                if "eval" in fun_selected:
                    Remove_DNFN_Eval_Agent(self.lc).Remove()

        elif fun_selected == "create_train":
            create_system(sc.base_dir_RL_system)
            self.run=self.fake_run
        elif fun_selected == "remote_copy":
            #copy_between_two_machine().copy_between_two_machine("192.168.199.100","")
            copy_between_two_machine().copy_between_two_machine()
            self.run = self.fake_run
        elif fun_selected == "DataBase":
            DB_main.main(argv[1:])
            self.run = self.fake_run
        else:
            raise ValueError("not support arg {0}".format(argv[1]))


    def fake_run(self):
        return
    def run(self):
        assert self.fun_label in ["learn", "eval","learneval"],"fun_label received is {0}".format(self.fun_label)
        logging.getLogger().setLevel(logging.INFO)
        #set root level to Debug, other wise, no matter the setting on sub level, it will only show waring and error
        self.Manager = Manager()
        self.process_name = self.fun_label
        self.logger = lcom.setup_logger(self.lc,self.process_name, flag_file_log=True, flag_screen_show=True)
        #set multiprocess logging
        log_to_stderr()
        logger = get_logger()
        logger.setLevel(logging.INFO)

        setproctitle.setproctitle("{0}_{1}".format(self.lc.RL_system_name, self.process_name))
        inp = pcom.name_pipe_cmd(self.lc,self.process_name)
        getattr(self,"{0}_init".format(self.fun_label))()
        while True:
            cmd_list = inp.check_input_immediate_return()
            if cmd_list is not None:
                if cmd_list[0][:-1] == "stop":
                    break
                else:
                    print("Unknown command: {0} from name pipe: {1}".format(cmd_list, inp.np_fnwp))
            if "learn" in self.fun_label:
                assert self.TrainP.is_alive(), "{0} process not alive".format(self.fun_label)
            time.sleep(60)
        getattr(self, "{0}_join".format(self.fun_label))()

    def learn_init(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.lc.get_CUDA_VISIBLE_DEVICES_str(self.lc.Brian_core)

        self.LL_output                   =   [self.Manager.list() for _ in range (self.lc.num_workers)]
        self.E_Stop_Brain                =   self.Manager.Event()
        self.E_Update_Weight            =   self.Manager.Event()
        self.D_share                     =   self.Manager.dict()
        self.TrainP = Train_Process(self.lc,self.LL_output,self.D_share,self.E_Stop_Brain,self.E_Update_Weight)
        self.TrainP.daemon=True
        self.TrainP.start()

        os.environ["CUDA_VISIBLE_DEVICES"] = self.lc.get_CUDA_VISIBLE_DEVICES_str(self.lc.work_core)

        self.L_E_Weight_Updated_Agent = [self.Manager.Event() for _ in range(self.lc.num_workers)]
        self.L_Agent2GPU=self.Manager.list()
        self.LL_GPU2Agent=[self.Manager.list() for _ in range(self.lc.num_workers)]

        self.E_stop_AgentMain = self.Manager.Event()
        self.AgentMainP = AgentMain( self.lc,self.LL_output, self.D_share, self.E_stop_AgentMain, self.E_Update_Weight,
                                     self.L_E_Weight_Updated_Agent ,self.L_Agent2GPU,self.LL_GPU2Agent)
        self.AgentMainP.daemon = True
        self.AgentMainP.start()

        self.L_Process_AgentSub = []
        self.L_E_Stop_AgentSub = [self.Manager.Event() for _ in range(self.lc.num_workers)]

        for idx in list(range(self.lc.num_workers)):
            AgentP = Agent_Sub(self.lc,idx, self.LL_output[idx],self.L_Agent2GPU, self.LL_GPU2Agent[idx],
                                       self.L_E_Stop_AgentSub[idx],self.L_E_Weight_Updated_Agent[idx])
            AgentP.daemon = True
            AgentP.start()
            self.L_Process_AgentSub.append(AgentP)

        tensorboard_dir = os.path.join(sc.base_dir_RL_system, self.lc.RL_system_name, "tensorboard")
        self.tensorboard_process = Process(target=start_tensorboard, args=(self.lc.tensorboard_port, tensorboard_dir,))
        self.tensorboard_process.start()

    def eval_init(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.lc.get_CUDA_VISIBLE_DEVICES_str(self.lc.eval_core)

        #total_num_eval_process=self.lc.eval_num_process_group*self.lc.eval_num_process_each_group
        total_num_eval_process = len(self.lc.l_eval_num_process_group) * self.lc.eval_num_process_each_group

        self.L_E_Start1Round = [self.Manager.Event() for _ in range(total_num_eval_process)]
        self.L_Eval2GPU = self.Manager.list()
        self.LL_GPU2Eval = [self.Manager.list() for _ in range(total_num_eval_process)]
        self.Share_eval_loop_count = self.Manager.Value('i', self.lc.start_eval_count)
        self.E_stop_EvalMain = self.Manager.Event()
        self.EvalMainP = EvalMain(self.lc, self.E_stop_EvalMain, self.L_E_Start1Round, self.L_Eval2GPU,
                                  self.LL_GPU2Eval, self.Share_eval_loop_count)
        self.EvalMainP.daemon = True
        self.EvalMainP.start()

        self.L_Process_Evalsub = []
        self.L_E_Stop_Evalsub = [self.Manager.Event() for _ in range(total_num_eval_process)]

        process_idx=0
        for process_group_idx in self.lc.l_eval_num_process_group:
            for _ in range(self.lc.eval_num_process_each_group):
                print ("Start Eval Process Group {0} Process idx {1}".format(process_group_idx,process_idx))
                Eval_P = EvalSub(self.lc, process_group_idx,process_idx, self.L_Eval2GPU, self.LL_GPU2Eval[process_idx], self.L_E_Stop_Evalsub[process_idx],
                                 self.L_E_Start1Round[process_idx], self.Share_eval_loop_count)
                Eval_P.daemon = True
                Eval_P.start()
                self.L_Process_Evalsub.append(Eval_P)
                process_idx+=1



    def learn_join(self):
        if self.L_E_Stop_AgentSub is not None and self.L_Process_AgentSub is not None:
            for E_Stop, P_Agent in zip(self.L_E_Stop_AgentSub, self.L_Process_AgentSub):
                E_Stop.set()
                P_Agent.join()
        if self.E_stop_AgentMain is not None and  self.AgentMainP is not None:
            self.E_stop_AgentMain.set()
            self.AgentMainP.join()
        if self.E_Stop_Brain is not None and self.TrainP is not None:
            self.E_Stop_Brain.set()
            self.TrainP.join()
        self.tensorboard_process.join()


    def eval_join(self):
        if self.L_Process_Evalsub is not None and self.L_E_Stop_Evalsub is not None:
            for E_Stop_Evalsub, P_Evalsub in zip(self.L_E_Stop_Evalsub, self.L_Process_Evalsub):
                E_Stop_Evalsub.set()
                P_Evalsub.join()
        if self.E_stop_EvalMain is not None and  self.EvalMainP is not None:
            self.E_stop_EvalMain.set()
            self.EvalMainP.join()


    def learneval_init(self):
        self.learn_init()
        self.eval_init()

    def learneval_join(self):
        self.learn_join()
        self.eval_join()


if __name__ == '__main__':
    main(sys.argv[1:]).run()

