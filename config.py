import os,json,re
from collections import OrderedDict
from action_comm import  actionOBOS
cwd = os.getcwd()

if "/home/" in cwd or "/mnt/" in cwd:
    base_dir="/home/rdchujf"
elif "/content/" in cwd:
    base_dir = "/content/drive"
else:
    raise ValueError( "not support run program in {0}".format(cwd))

#Directory
qz_rar_dir="/home/rdchujf/rar_quanzheng"
base_dir_RL_data                            = os.path.join(base_dir,"n_workspace","data/RL_data")
base_dir_RL_system                          = os.path.join(base_dir,"n_workspace","RL")
for dir in [base_dir_RL_data,base_dir_RL_system]:
    if not os.path.exists(dir): raise ValueError("{0} not exists".format(dir))


l_GPU_size=[11019,12196]

class gconfig_specific:
    """
    @DynamicAttrs
    """
class gconfig_data:
    def __init__(self):
        ###general input
        self.RL_system_name =  float("nan") #"RL_try"
        self.SLName= ""
        self.data_name = float("nan") #
        self.system_type = float("nan") #"CTE"
        ###class name
        self.CLN_brain_train = float("nan") #"Train_Brian"
        self.CLN_brain_explore = float("nan") #"Explore_Brain"
        self.CLN_buffer_to_train = float("nan") #"buffer_to_train"
        self.CLN_simulator = float("nan") #"Simulator"
        self.CLN_trainer = float("nan") #"PG_trainer",
        self.CLN_env_account = float("nan") #"env_account"
        self.CLN_env_read_data = float("nan") #"R_T5"
        self.CLN_TDmemory = float("nan") #"TD_memory"
        self.CLN_GenStockList = float("nan") #"API_SH_sl"

        self.train_scale_factor= float("nan")
        self.train_shift_factor= float("nan")
        self.train_flag_clip=float("nan")
        self.train_flag_punish_no_action=float("nan")
        self.eval_scale_factor= float("nan")
        self.eval_shift_factor= float("nan")
        self.eval_flag_clip=float("nan")
        self.eval_flag_punish_no_action=float("nan")

        self.agent_method_sv= float("nan") #"CNN"
        self.agent_method_joint_lvsv= float("nan") #"CNN"
        self.agent_method_apsv= float("nan") #"HP"


        ###train_brain
        self.Brian_core = ""
        self.Brian_gpu_percent = float("nan") #0.8
        self.flag_brain_log_file = float("nan") #True
        self.flag_brain_log_screen = float("nan") #True
        #self.start_train_count = float("nan") #0
        self.num_train_to_save_model = float("nan") #1000
        self.load_AIO_fnwp = ""
        self.load_config_fnwp = ""
        self.load_weight_fnwp = ""

        ###BUFFER
        self.num_train_record_to_brain = float("nan") #100
        self.TDn = float("nan") #5
        self.Buffer_nb_Features = float("nan") #6
        self.brain_buffer_reuse_times = float("nan") #1

        # explore class
        self.l_flag_worker_log_file = [float("nan")] #[True, True, True]
        self.l_flag_worker_log_screen = [float("nan")] #[False, False, False]
        self.num_workers = float("nan") #3
        self.work_core = "" #"GPU_0"
        self.percent_gpu_core_for_work = float("nan") #0.2
        self.CLN_env_get_data_train = ""
        self.train_SL_param=[]

        # eval class
        self.l_flag_eval_log_file = [float("nan")] #[True, True]
        self.l_flag_eval_log_screen = [float("nan")] #[False, False]
        self.l_eval_SL_param = [[0, 20000000, 20000000]]
        self.l_CLN_env_get_data_eval=""
        self.start_eval_count = float("nan") #0
        self.eval_core = "" #"GPU_1"
        self.percent_gpu_core_for_eva = float("nan") #0.2
        self.l_eval_num_process_group=[2]
        self.eval_num_process_per_group=3

        # loss WEIGHT
        self.LOSS_POLICY = float("nan") #1.0
        self.LOSS_V = float("nan") #0.5
        self.LOSS_ENTROPY = float("nan") #0.01
        self.LOSS_clip = float("nan") #0.2
        self.LOSS_sqr_threadhold = float("nan") #10

        # optimizer
        self.Brain_optimizer = float("nan") #"Adam"
        self.Brain_leanring_rate = float("nan") #1e-4
        self.Brain_gamma = float("nan") #0.95
        self.batch_size = float("nan") #500

        # net config
        self.net_config = {} #{}
        # ENV CONFIG
        self.env_min_invest_per_round = float("nan") #100000
        self.env_max_invest_per_round = float("nan") #500000
        self.evn_eval_rest_total_times = float("nan") #500

        # train
        self.train_data_base = float("nan") #10
        self.train_data_index = float("nan") #1
        # eval
        self.eval_data_base = float("nan") #5
        self.eval_data_index = float("nan") #1
        self.eval_num_stock_per_process = float("nan") #100
        # action realted
        self.action_type_dict = float("nan") #{0: "buy", 1: "sell", 2: "no_action"}
        # debug
        self.flag_record_state = float("nan") #True
        self.flag_record_buffer_to_server = float("nan") #False
        self.flag_record_sim = float("nan") #False

        self.CLN_record_variable = float("nan") #"record_variable"
        self.tensorboard_port = float("nan") #6006
        # LHF
        self.LHP = float("nan") #0
        self.LNB = float("nan")  # 0
        # P2
        self.P2_current_phase = float("nan") #""
        self.P2_sell_system_name = float("nan") #""
        self.P2_sell_model_tc = float("nan") #-1
        # reward related
        self.CLN_AV_state = ""
        self.CLN_AV_Handler=""

        # value set by config
        self.Dict_specifc_param = {} #{}
        self.train_action_type = float("nan") #""  # "OB,"OS,"BS"
        self.train_num_action = float("nan") #

        self.OB_AV_shape=()
        self.OS_AV_shape=()
        self.raw_AV_shape=()

        self.Plen=float("nan")

        self.Max_TotalMoney=float("nan")
        self.low_profit_threadhold=float("nan")
        self.CC_strategy_fun=""

        #new add param, have default value here
        self.flag_train_random_explore=True
        self.flag_train_store_AIO_model=True
        self.train_random_explore_prob_buy=0.2
        self.train_total_los_clip=0
class gconfig(gconfig_data):
    def __init__(self):
        gconfig_data.__init__(self)
        # seed
        self.client_process_name_seed = "ExploreAgent"
        self.server_process_name_seed = "TrainBrain"
        self.eval_process_seed = "EvalAgent"
        self.actor_model_AIO_fn_seed = "train_model_AIO"
        self.actor_config_fn_seed = "config"
        self.actor_weight_fn_seed = "weight"
        self.log_a_r_e_fn_seed = "log_a_r_e"
        self.log_e_s_d_i_fn_seed = "log_s_s_d_i"
        self.command_pipe_seed = "pipe.command"
        self.specific_param=gconfig_specific()
        self.account_inform_titles=["TransIDI", "Holding_Gu", "Holding_Invest", "Holding_HRatio", "Holding_NPrice",
                               "Buy_Times", "Buy_Invest", "Buy_NPrice", "Sell_Return", "Sell_Earn","Sell_NPrice"]

        self.simulator_inform_titles=["DateI","StockI","Eval_Profit"]
        self.PSS_inform_titles =["AcutalAction"]
    def read_from_json(self, param_fnwp):
        param = json.load(open(param_fnwp, "r"), object_pairs_hook=OrderedDict)
        for item in list(param.keys()):
            sitem = str(item)
            if not sitem.startswith("======="):
                self.__dict__[sitem] = param[sitem]

        base_dir,fn=os.path.split(param_fnwp)
        assert fn=="config.json"
        base_dir, self.RL_system_name = os.path.split(base_dir)

        self.sanity_check_convert_enhance()

    def sanity_check_convert_enhance(self):
        self.Brian_gpu_percent = l_GPU_size[int(self.Brian_core[-1])]*self.Brian_gpu_percent
        self.percent_gpu_core_for_work=l_GPU_size[int(self.work_core[-1])]*self.percent_gpu_core_for_work
        self.percent_gpu_core_for_eva=l_GPU_size[int(self.eval_core[-1])]*self.percent_gpu_core_for_eva
        assert self.P2_current_phase in  ["Train_Sell","Train_Buy"]
        assert self.env_max_invest_per_round==self.env_min_invest_per_round,"Only support single buy"

        if self.load_AIO_fnwp!="" and self.load_config_fnwp!="" and self.load_weight_fnwp!="":
            fn = os.path.basename(self.load_AIO_fnwp)
            start_train_count_indication1=int(re.findall(r'\w+T(\d+).h5', fn)[0])
            fn = os.path.basename(self.load_weight_fnwp)
            start_train_count_indication2=int(re.findall(r'\w+T(\d+).h5', fn)[0])
            assert start_train_count_indication1==start_train_count_indication2
            #self.start_train_count=start_train_count_indication1

        new_action_type_dict={}
        for item in list(self.action_type_dict.keys()):
            new_action_type_dict[int(item)]=self.action_type_dict[item]
        self.action_type_dict=new_action_type_dict

        # convert unicode to string
        for item in list(self.__dict__.keys()):
            # convert unicode to string
            if type(self.__dict__[str(item)]) is str:
                # print str(item), gc.__dict__[str(item)]
                self.__dict__[str(item)] = str(self.__dict__[str(item)])
            # convert list with item in list is unicode
            elif type(self.__dict__[str(item)]) is list:
                if type(self.__dict__[str(item)][0]) == str:
                    self.__dict__[str(item)] = [str(iitem) for iitem in self.__dict__[str(item)]]
            else:
                continue

        #enhancement
        self.system_working_dir = os.path.join(base_dir_RL_system, self.RL_system_name)
        if not os.path.exists(self.system_working_dir): os.mkdir(self.system_working_dir)
        self.log_dir=os.path.join(self.system_working_dir,"log")
        if not os.path.exists(self.log_dir): os.mkdir(self.log_dir)
        self.brain_model_dir = os.path.join(self.system_working_dir, "model")
        if not os.path.exists(self.brain_model_dir): os.mkdir(self.brain_model_dir)
        self.tensorboard_dir = os.path.join(self.system_working_dir, "tensorboard")
        if not os.path.exists(self.tensorboard_dir): os.mkdir(self.tensorboard_dir)
        self.record_variable_dir = os.path.join(self.system_working_dir, "record_state")
        if not os.path.exists(self.record_variable_dir): os.mkdir(self.record_variable_dir)
        self.record_train_buffer_dir = os.path.join(self.system_working_dir, "record_tb")
        if not os.path.exists(self.record_train_buffer_dir): os.mkdir(self.record_train_buffer_dir)

        if self.brain_buffer_reuse_times == 1:
            setattr(self, "CLN_brain_buffer", "brain_buffer")
        else:
            setattr(self, "CLN_brain_buffer", "brain_buffer_reuse")

        assert self.CLN_trainer == "PPO_trainer",self.CLN_trainer

        assert self.agent_method_sv in ["CNN","CNN2D","CNN2Dvalid","CNN2DV2","CNN2DV3","CNN2DV4"]   #remove "RNN","RCN"
        assert self.agent_method_joint_lvsv in ["CNN","CNN2D","CNN2Dvalid","CNN2DV2","CNN2DV3","CNN2DV4"] #remove "RNN","RCN"
        assert self.agent_method_apsv in ["HP"]
        if self.CLN_AV_Handler=="AV_Handler":
            self.OS_AV_shape = (self.LHP + 1,)
            self.OB_AV_shape = (self.LNB + 1,)
        elif self.CLN_AV_Handler=="AV_Handler_AV1":
            self.OS_AV_shape = (1,)
            self.OB_AV_shape = (1,)
        len_inform=len(self.account_inform_titles) + len(self.simulator_inform_titles) + len(self.PSS_inform_titles)
        self.raw_AV_shape = (self.LNB + 1 + 2 + self.LHP + 1 + 2 + 1+1 +len_inform,)
        self.PLen = self.LHP + self.LNB

        l_specific_param_title=[]

        if self.system_type == "LHPP2V2":
            assert self.P2_current_phase == "Train_Sell"
            self.train_action_type = "OS"
            self.train_num_action = 2
            assert self.net_config["dense_prob"][-1] == self.train_num_action
            actionOBOS(self.train_action_type).sanity_check_action_config(self)
            # specific parm
            for item_title in l_specific_param_title:
                assert item_title in list(self.Dict_specifc_param.keys())
                setattr(self.specific_param,item_title,self.Dict_specifc_param[item_title])

        elif self.system_type == "LHPP2V3":   #V3 means buy policy
            assert self.P2_current_phase == "Train_Buy"
            self.train_action_type = "OB"
            self.train_num_action = 2
            assert self.net_config["dense_prob"][-1] == self.train_num_action
            actionOBOS(self.train_action_type).sanity_check_action_config(self)
            # 8.specific param
            for item_title in l_specific_param_title:
                assert item_title in list(self.Dict_specifc_param.keys())
                setattr(self.specific_param,item_title,self.Dict_specifc_param[item_title])

        else:
            assert False, "not support type: {0}".format(self.system_type)

    def get_CUDA_VISIBLE_DEVICES_str(self, core_str):
        if core_str.startswith("GPU"):
           return core_str[-1]
        elif core_str=="CPU":
            return ""
        else:
            raise ValueError("unkown selected core {0}".format(core_str))
