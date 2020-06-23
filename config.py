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
#-- rar base dir
qz_rar_dir="/home/rdchujf/rar_quanzheng"
#--data source_base dir
base_dir_source_trade_by_trade              = os.path.join(base_dir, "Stk_TradeByTrade")
base_dir_qz_1                               = os.path.join(base_dir, "Stk_qz")
base_dir_qz_2                               = os.path.join(base_dir, "Stk_qz_2")
base_dir_RL_data                            = os.path.join(base_dir,"n_workspace","data/RL_data")
base_dir_RL_system                          = os.path.join(base_dir,"n_workspace","RL")
for dir in [base_dir_source_trade_by_trade,base_dir_qz_1,base_dir_qz_2,base_dir_RL_data,base_dir_RL_system]:
    if not os.path.exists(dir): raise ValueError("{0} not exists".format(dir))

#Param
#--qz
qz_sh_avail_start = "20130415"
qz_sz_avail_start = "20130104"
qz_avail_end = "20171229"

V2_start_tims_s = "20180101"
V2_end_tims_s = "20190731"

#qz data fils
missing_data_fnwp = os.path.join(base_dir_qz_1,"missingdate.csv")
zero_len_data_fnwp = os.path.join(base_dir_qz_1,"zero_data.csv")


#data set param
RL_data_skip_days = 100
RL_data_least_length = 23
RL_da_dan_threadhold = 1000000
RL_xiao_dan_threadhold = 100000

l_GPU_size=[11019,12196]
conf= \
    {
        "=======General=======": "=======================",
        #"RL_system_name": "OS1",
        "data_name": "T5",
        "system_type": "LHPP2V2",
        "=======Class=======": "=======================",
        "CLN_brain_train": "Train_Brian",
        "CLN_brain_explore": "Explore_Brain",
        #"CLN_brain_buffer":"brain_buffer",
        "CLN_buffer_to_train": "buffer_to_train",
        "CLN_env_account": "env_account",
        "CLN_env_read_data": "R_T5_skipSwh_balance",
        "CLN_env_get_data_train": "env_get_data_LHP_train",
        "CLN_env_get_data_eval": "env_get_data_LHP_eval",
        "CLN_AV_state":"Phase_State_V2",
        "CLN_GenStockList": "API_SH_sl",
        "=======system type related class": "=======================",
        "CLN_simulator": "Simulator_LHPP2V2",
        "CLN_trainer": "LHPP2V2_PPO_trainer",
        "CLN_TDmemory": "TD_memory_LHPP2V2",
        "=======CLN_agent detailed===": "=======================",
        "agent_method_sv": "CNN",
        "agent_method_joint_lvsv": "CNN",
        "agent_method_apsv": "HP",
        "=======action related=======": "=======================",
        "action_type_dict": {"0": "buy", "1": "no_action", "2": "sell", "3": "no_action"},
        #"method_name_of_choose_action_for_train": "choose_action_LHPP2V2",
        #"method_name_of_choose_action_for_eval": "choose_action_LHPP2V2",
        "=======Train_record_realted=======": "=======================",
        "TD_get_verified_record": "",
        "TD_get_after_buy_accumulate_R":"",
        "Optimize_accumulate_reward_method": "",
        "=======Class_eval_train=======": "=======================",
        "train_scale_factor":20,
        "train_shift_factor":0.01,
        "train_flag_clip":True,
        "train_flag_punish_no_action":False,
        "eval_scale_factor": 1,
        "eval_shift_factor": 0,
        "eval_flag_clip": False,
        "eval_flag_punish_no_action": False,

        #"train_reward_type": "scaler_clip",
        #"train_reward_scaler_factor":1,
        #"eval_reward_type": "scaler",
        #"eval_reward_scaler_factor":1,
        "=======Train_brain=======": "=======================",
        "Brian_core": "GPU_0",
        "Brian_gpu_percent": 0.6,
        "flag_brain_log_file": False,
        "flag_brain_log_screen": True,
        "start_train_count": 0,
        "num_train_to_save_model": 1000,
        "load_AIO_fnwp": "",
        "load_config_fnwp": "",
        "load_weight_fnwp": "",
        "=======Buffer=======": "=======================",
        "num_train_record_to_brain": 100,
        "TDn": 1,
        "Buffer_nb_Features": 6,
        "brain_buffer_reuse_times": 4,
        "=======Explore Procee=======": "=======================",
        "l_work_core": ["GPU_0"],
        "l_percent_gpu_core_for_work": [0.4],
        "l_flag_worker_log_file": [False],
        "l_flag_worker_log_screen": [True],
        "num_workers": 1,
        "=======Eval process=======": "=======================",
        "l_eval_core": ["GPU_1"],
        "l_percent_gpu_core_for_eva": [0.4],
        "l_flag_eval_log_file": [False],
        "l_flag_eval_log_screen": [True],
        "l_eval_data_name": ["T5"],
        "start_eval_count": 0,
        "eval_num_process": 1,
        "=======LOSS=======": "=======================",
        "LOSS_POLICY": 1.0,
        "LOSS_V": 0.5,
        "LOSS_ENTROPY": 0.01,
        "LOSS_clip":0.2,
        "LOSS_sqr_threadhold":10,
        "=======Optimizer=======": "=======================",
        "Brain_optimizer": "Adam",
        "Brain_leanring_rate": 1e-4,
        "Brain_gamma": 0.95,
        "batch_size": 500,
        "=======Net=======": "=======================",
        "net_config": {
            "lv_shape": [20, 16],
            "sv_shape": [20, 25, 2],
            "flag_l_level": [
                "C",
                "I",
                "I",
                "C"
            ],
            "l_kernel_l": [
                3,
                0,
                0,
                3
            ],
            "l_filter_l": [
                64,
                128,
                256,
                512
            ],
            "l_maxpool_l": [
                2,
                2,
                2,
                2
            ],
            "flag_s_level": [
                "C",
                "C",
                "C",
                "C"
            ],
            "s_kernel_l": [
                3,
                3,
                3,
                3
            ],
            "s_filter_l": [
                32,
                64,
                128,
                256
            ],
            "s_maxpool_l": [
                2,
                2,
                2,
                2
            ],
            "dense_l": [
                512
            ],
            "dense_prob": [
                256,
                128,
                64,
                32,
                2
            ],
            "dense_advent": [
                256,
                128,
                64,
                32,
                1
            ],
            "ms_param_TimeDistributed": [
                100,
                256
            ],
            "ms_param_CuDNNLSTM": [
                100,
                200
            ]
        },
        "=======Simulator=======": "=======================",
        "env_min_invest_per_round": 100000,
        "env_max_invest_per_round": 100000,
        "env_flag_random_start_in_episode_for_explore": True,
        "env_flag_random_start_in_episode_for_eval": True,
        "evn_eval_rest_total_times": 200,
        "=======train Data=======": "=======================",
        "train_data_base": 2,
        "train_data_index": 1,
        "=======Eval Data=======": "=======================",
        "eval_data_base": 4,
        "eval_data_index": 0,
        "eval_num_stock_per_process": 64,
        "=======debug=======": "=======================",
        "flag_record_state": True,
        "flag_record_buffer_to_server": False,
        "flag_record_sim": False,
        "CLN_record_variable": "record_variable2",
        "record_checked_threahold": 500,
        "tensorboard_port": 6002,
        "=======system_type_specific LHP=======": "=======================",
        "LHP": 5,
        "=======system_type_specific 2P=======": "=======================",
        "P2_current_phase": "Train_Sell",
        "P2_sell_system_name": "",
        "P2_sell_model_tc": "",
        "=======trial=======": "=======================",
        "Dict_specifc_param": {
        }
    }
class gconfig_specific:
    """
    @DynamicAttrs
    """
class gconfig_data:
    def __init__(self):

        ###general input
        self.RL_system_name =  float("nan") #"RL_try"
        self.data_name = float("nan") #"T5"
        self.system_type = float("nan") #"CTE"
        ###class name
        self.CLN_brain_train = float("nan") #"Train_Brian"
        self.CLN_brain_explore = float("nan") #"Explore_Brain"
        #self.CLN_brain_buffer = float("nan") #"brain_buffer"
        self.CLN_buffer_to_train = float("nan") #"buffer_to_train"
        self.CLN_simulator = float("nan") #"Simulator"
        self.CLN_trainer = float("nan") #"PG_trainer",

        self.CLN_env_account = float("nan") #"env_account"

        self.train_scale_factor= float("nan")
        self.train_shift_factor= float("nan")
        self.train_flag_clip=float("nan")
        self.train_flag_punish_no_action=float("nan")
        self.eval_scale_factor= float("nan")
        self.eval_shift_factor= float("nan")
        self.eval_flag_clip=float("nan")
        self.eval_flag_punish_no_action=float("nan")

        #self.train_reward_type=float("nan") #"scaler" "scaler_clip"
        #self.train_reward_scaler_factor=float("nan") #1
        #self.eval_reward_type = float("nan")  # "scaler" "scaler_clip"
        #self.eval_reward_scaler_factor=float("nan") #1

        self.CLN_env_read_data = float("nan") #"R_T5"
        self.CLN_env_get_data_train = float("nan") #"env_get_data_base"
        self.CLN_env_get_data_eval = float("nan") #"env_get_data_base"
        self.CLN_TDmemory = float("nan") #"TD_memory"
        self.CLN_GenStockList = float("nan") #"API_SH_sl"


        self.agent_method_sv= float("nan") #"CNN"
        self.agent_method_joint_lvsv= float("nan") #"CNN"
        self.agent_method_apsv= float("nan") #"HP"


        ###train_brain
        ##core
        self.Brian_core = ""
        self.Brian_gpu_percent = float("nan") #0.8
        ##log
        self.flag_brain_log_file = float("nan") #True
        self.flag_brain_log_screen = float("nan") #True
        # other
        self.start_train_count = float("nan") #0
        self.num_train_to_save_model = float("nan") #1000

        self.load_AIO_fnwp = ""
        self.load_config_fnwp = ""
        self.load_weight_fnwp = ""
        #
        ###BUFFER
        self.num_train_record_to_brain = float("nan") #100
        self.TDn = float("nan") #5
        self.Buffer_nb_Features = float("nan") #6
        self.brain_buffer_reuse_times = float("nan") #1

        # explore class
        self.l_work_core = ["",""] #["GPU_0", "GPU_0", "GPU_0"]
        self.l_percent_gpu_core_for_work = [float("nan")] #[0.2, 0.2, 0.2]
        self.l_flag_worker_log_file = [float("nan")] #[True, True, True]
        self.l_flag_worker_log_screen = [float("nan")] #[False, False, False]
        self.num_workers = float("nan") #3

        # eval class
        self.l_eval_core = ["",""] #["GPU_1", "GPU_1"]
        self.l_percent_gpu_core_for_eva = [float("nan")] #[0.2, 0.2]
        self.l_flag_eval_log_file = [float("nan")] #[True, True]
        self.l_flag_eval_log_screen = [float("nan")] #[False, False]
        self.l_eval_data_name =[float("nan")] # ["T5", "T5", "T5_V2_"]
        self.start_eval_count = float("nan") #0
        self.eval_num_process = float("nan") #2

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
        self.env_flag_random_start_in_episode_for_explore = float("nan") #False
        self.env_flag_random_start_in_episode_for_eval = float("nan") #False
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
        # self.num_action=3
        #self.method_name_of_choose_action_for_train = float("nan") #""
        #self.method_name_of_choose_action_for_eval = float("nan") #""
        # debug
        # self.flag_recorder=True
        # "flag_recorder": True,
        self.flag_record_state = float("nan") #True
        self.flag_record_buffer_to_server = float("nan") #False
        self.flag_record_sim = float("nan") #False

        self.CLN_record_variable = float("nan") #"record_variable"
        self.record_checked_threahold = float("nan") #0
        self.tensorboard_port = float("nan") #6006
        # LHF
        self.LHP = float("nan") #0
        # P2
        self.P2_current_phase = float("nan") #""
        self.P2_sell_system_name = float("nan") #""
        self.P2_sell_model_tc = float("nan") #-1
        # reward related
        self.TD_get_verified_record = ""
        self.TD_get_after_buy_accumulate_R = ""
        self.Optimize_accumulate_reward_method = ""
        self.CLN_AV_state = ""

        # value set by config
        self.Dict_specifc_param = {} #{}
        self.train_action_type = float("nan") #""  # "OB,"OS,"BS"
        self.train_num_action = float("nan") #

        self.OB_AV_shape=()
        self.OS_AV_shape=()
        self.raw_AV_shape=()

class gconfig(gconfig_data):
    def __init__(self):
        gconfig_data.__init__(self)
        # seed
        self.client_process_name_seed = "Explore_worker"
        self.server_process_name_seed = "Train_Brain"
        self.eval_process_seed = "Eval"
        self.actor_model_AIO_fn_seed = "train_model_AIO"
        self.actor_config_fn_seed = "config"
        self.actor_weight_fn_seed = "weight"
        self.log_a_r_e_fn_seed = "log_a_r_e"
        self.log_e_s_d_i_fn_seed = "log_s_s_d_i"
        self.command_pipe_seed = "pipe.command"
        self.specific_param=gconfig_specific()

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
        #sanity check
        assert len(self.l_work_core) == self.num_workers
        assert len(self.l_percent_gpu_core_for_work) == self.num_workers
        assert len(self.l_flag_worker_log_file) == self.num_workers
        assert len(self.l_flag_worker_log_screen) == self.num_workers

        assert len(self.l_eval_core)==self.eval_num_process
        assert len(self.l_percent_gpu_core_for_eva) == self.eval_num_process
        assert len(self.l_flag_eval_log_file) == self.eval_num_process
        assert len(self.l_flag_eval_log_screen) == self.eval_num_process


        self.Brian_gpu_percent = l_GPU_size[int(self.Brian_core[-1])]*self.Brian_gpu_percent

        self.l_percent_gpu_core_for_work = [l_GPU_size[int(work_core[-1])]*percent_gpu_core
                            for work_core,percent_gpu_core in zip(self.l_work_core,self.l_percent_gpu_core_for_work)]

        self.l_percent_gpu_core_for_eva =  [l_GPU_size[int(eval_core[-1])]*percent_gpu_core
                            for eval_core,percent_gpu_core in zip(self.l_eval_core,self.l_percent_gpu_core_for_eva)]

        assert self.env_max_invest_per_round>=self.env_min_invest_per_round
        assert self.P2_current_phase in  ["Train_Sell","Train_Buy"]

        self.flag_eval_unlearn = True if self.eval_num_process!=0 else False
        self.flag_multi_buy =True if self.env_max_invest_per_round/self.env_min_invest_per_round >1 else False
        self.times_to_buy =self.env_max_invest_per_round/self.env_min_invest_per_round

        if self.load_AIO_fnwp!="" and self.load_config_fnwp!="" and self.load_weight_fnwp!="":
            fn = os.path.basename(self.load_AIO_fnwp)
            start_train_count_indication1=int(re.findall(r'\w+T(\d+).h5py', fn)[0])
            fn = os.path.basename(self.load_weight_fnwp)
            start_train_count_indication2=int(re.findall(r'\w+T(\d+).h5py', fn)[0])
            assert start_train_count_indication1==start_train_count_indication2
            self.start_train_count=start_train_count_indication1

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
        l_specific_param_title=[]
        # 1.Simulator get data
        assert self.CLN_env_get_data_train == "env_get_data_LHP_train"
        assert self.CLN_env_get_data_eval == "env_get_data_LHP_eval"

        # 2.Simulator
        assert self.CLN_simulator == "Simulator_intergrated"  # "Simulator_LHPP2V8"#"Simulator_LHPP2V2"
        assert not self.flag_multi_buy
        assert self.LHP != 0
        assert self.env_flag_random_start_in_episode_for_eval == True

        # 3.TD_buffer
        assert self.CLN_TDmemory == "TD_memory_integrated"

        # 5.net_agent
        assert self.agent_method_sv in ["RNN", "CNN", "RCN"]
        assert self.agent_method_joint_lvsv in ["RNN", "CNN", "RCN"]
        assert self.agent_method_apsv in ["HP", "HP_SP", "HP_DAV"]

        self.flag_sv_stop_gradient, self.flag_sv_joint_state_stop_gradient = [False, True] \
            if "_SP" in self.agent_method_apsv else [False, False]  ## can not be [True True] situation

        if self.system_type == "LHPP2V2":    #V2 means reuse to multibuy part av to encode signle buy holding duration
            # 0.Train Phase
            assert self.P2_current_phase == "Train_Sell"
            # 6.net_trainer
            assert "LHPP2V2_" in self.CLN_trainer
            # 7.action
            self.train_action_type = "OS"
            self.train_num_action = 2
            assert self.net_config["dense_prob"][-1] == self.train_num_action
            actionOBOS(self.train_action_type).sanity_check_action_config(self)
            # 8.specific parm
            for item_title in l_specific_param_title:
                assert item_title in list(self.Dict_specifc_param.keys())
                setattr(self.specific_param,item_title,self.Dict_specifc_param[item_title])
            assert self.CLN_AV_state == "Phase_State_V2"
            self.OS_AV_shape = (self.LHP + 1,)
            self.raw_AV_shape =(self.LHP + 1,)

        elif self.system_type == "LHPP2V3":   #V3 means buy policy
            # 0.Train Phase
            assert self.P2_current_phase == "Train_Buy"
            # 6.net_trainer  # this is to include support for V3 V32 and V33
            assert "LHPP2V3" in self.CLN_trainer
            # 7.action
            self.train_action_type = "OB"
            self.train_num_action = 2
            assert self.net_config["dense_prob"][-1] == self.train_num_action
            actionOBOS(self.train_action_type).sanity_check_action_config(self)
            # 8.specific param
            assert self.CLN_AV_state in ["Phase_State_V3__1","Phase_State_V3__2"]
            l_specific_param_title.append("LNB")
            for item_title in l_specific_param_title:
                assert item_title in list(self.Dict_specifc_param.keys())
                setattr(self.specific_param,item_title,self.Dict_specifc_param[item_title])

            if self.CLN_AV_state =="Phase_State_V3__1":
                self.OB_AV_shape = (self.specific_param.LNB + 1,)
            else:
                self.OB_AV_shape = (1,)
            self.OS_AV_shape =(self.LHP + 1,)
            self.raw_AV_shape = (self.specific_param.LNB + 1 + self.LHP + 1,)

        elif self.system_type in ["LHPP2V8"]:
            # 0.Train Phase
            assert self.P2_current_phase == "Train_Buy"
            # 6.net_trainer  # this is to include support for V3 V32 and V33
            assert self.system_type in self.CLN_trainer
            # 7.action
            self.train_action_type = "B32"
            self.train_num_action = 3
            assert self.net_config["dense_prob"][-1] == self.train_num_action
            assert self.net_config["dense_advent"][-1] == 1
            actionOBOS(self.train_action_type).sanity_check_action_config(self)
            # 8.specific param
            assert self.CLN_AV_state == "Phase_State_V8"
            l_specific_param_title.append("LNB")
            l_specific_param_title.append("LNT")
            for item_title in l_specific_param_title:
                assert item_title in list(self.Dict_specifc_param.keys())
                setattr(self.specific_param,item_title,self.Dict_specifc_param[item_title])
            self.OS_AV_shape = (self.LHP + 1,)
            self.OB_AV_shape = (self.specific_param.LNT+1+self.specific_param.LNB+1+1,)
            self.raw_AV_shape = (self.specific_param.LNT + 1 + self.specific_param.LNB + 1 + self.LHP + 1,)
        else:
            assert False, "not support type: {0}".format(self.system_type)

    def get_CUDA_VISIBLE_DEVICES_str(self, core_str):
        if core_str.startswith("GPU"):
           return core_str[-1]
        elif core_str=="CPU":
            return ""
        else:
            raise ValueError("unkown selected core {0}".format(core_str))

