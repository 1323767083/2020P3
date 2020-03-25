# purpose

    To merge the enviroment, enviroment only have several senario
    1. normal
    2. have limit holding period
    3. have limit not buy period
    
    To merge the TD buffer, all should have the SdisS_
    1. normal, which not check the support view
    2. customized:
        1. check successful sell
        2. check successful buy
        3. check foresell
    
    To simplify A3C_worker
    1. the support of different action format should not in the ac_worker coder
    
    To simplify  agent and trainer, baisc rule:
    1. each senario OB, OS should have only one catagory  like 2(OS)  3(OB)
    2. in one senario, different model structure have a small number like:
        a.31 (OS state value solution)
        b.32 (OS P Q value solution)
        c.33 (OS Q table soultion)
    3. 0 will be the formal solution, like
        a. 30 is OB formal solution
    4. T0, T1 will be trail solution
        a. 31T0 is a trial for 31

#optimize trainer structure (Done)
## base_trainer only includes three function 
    
        def select_optimizer(self, name, learning_rate):
        def load_train_model(self, fnwps):
        def _vstack_states(self,i_train_buffer):

##move other method to  LHP_trainer_fun
        def _prepare_build_train_model(self):
        def extract_y(self, y):
        def join_loss_policy_part(self,y_true,y_pred):
        def join_loss_entropy_part(self, y_true, y_pred):
        def join_loss_entropy_part_old(self, y_true, y_pred):
        def join_loss_sv_part(self, y_true, y_pred):
        def join_loss_selection(self, masktype):
        def M_policy_loss_selection(self, masktype):
        def M_value_loss_selection(self, masktype):
        def M_entropy_selection(self,  masktype):
        def M_state_value_selection(self,masktype):
        def M_advent_selection(self, masktype):
        def M_advent_low_selection (self, masktype):
        def M_advent_high_selection (self, masktype):
##divide __init__
        to base_trainer
            def __init__(self, class_name_policy_agent):
            self.gammaN = lc.Brain_gamma ** lc.TDn
            self.i_policy_agent = General_agent(class_name_policy_agent)
            self.build_predict_model=self.i_policy_agent.build_predict_model
            if lc.flag_record_state:
                self.rv = globals()[lc.CLN_record_variable](lc)

        to LHP_trainer_fun
            def __init__(self, class_name_policy_agent):
            self.join_loss=self.join_loss_selection("NM")
            self.M_policy_loss = self.M_policy_loss_selection("NonMask")
            self.M_value_loss = self.M_value_loss_selection("NonMask")
            self.M_entropy = self.M_entropy_selection("NonMask")
            self.M_state_value=self.M_state_value_selection("NonMask")
            self.M_advent = self.M_advent_selection("NonMask")
            self.M_advent_low = self.M_advent_low_selection("NonMask")
            self.M_advent_high = self.M_advent_high_selection("NonMask")
    
            self.comile_metrics=[self.M_policy_loss, self.M_value_loss,self.M_entropy,self.M_state_value,self.M_advent,
                                 self.M_advent_low,self.M_advent_high]
    
            self.load_jason_custom_objects={"softmax": softmax,"tf":tf, "concatenate":concatenate,"lc":lc}
            self.load_model_custom_objects={"join_loss": self.join_loss, "tf":tf,"concatenate":concatenate,
                                            "M_policy_loss":self.M_policy_loss,"M_value_loss":self.M_value_loss,
                                            "M_entropy":self.M_entropy,"M_state_value":self.M_state_value,
                                            "M_advent":self.M_advent,"M_advent_low":self.M_advent_low,
                                            "M_advent_high":self.M_advent_high,"lc":lc}
##    create net_trainer_legacy and move two class in
        class PG_trainer(base_trainer):
        class PPO_trainer(base_trainer):


##    review nets_trainer_LHPPP2V3 interit base_trainer

# change Simulator_LHPP2V32 to Simulator_LHPP2V3 (Done)
* change class name  from Simulator_LHPP2V32 to Simulator_LHPP2V3
* config.py 
    * LHPP2V3 LHPP2V33 from Simulator_LHPP2V2 to Simulator_LHPP2V3
    * LHPP2V32 from Simulator_LHPP2V32 to Simulator_LHPP2V3
* check impact LHPP2V3 LHPP2V33
* change import related LHPP2V32
* modify new Simulator_LHPP2V3
    * implement not force buy in step_OB_train by add conditional check if lc.specific_param.BB_NBD!=0:
    * implement limit no action before buy in step_OB_eval, this need to be tested 
        set Done_flag to be True, so that it trigger next round, and since each reset will count in 
        the class env_get_data_LHP_eval_nc  method reset_get_data, the actual eval success buy time will be reduced

# change TD_memory_LHPP2V32 to TD_memory_LHPP2V3 (Done)
* change class name TD_memory_LHPP2V32 to TD_memory_LHPP2V3
* config.py 
    * LHPP2V3 LHPP2V32 LHPP2V33 to TD_memory_LHPP2V3
* check impact on: cause dy support_view_dic["SdisS_"]
    * LHPP2V3_PPO_trainer1
    * LHPP2V32_PPO_trainer1
    * LHPP2V33_PPO_trainer1
    * recorder.py store load two place
        change if self.lc.system_type in ["LHPP2V3", "LHPP2V32", "LHPP2V33"]:  
        #to if self.lc.self.CLN_TDmemory == "TD_memory_LHPP2V3":

# how to imporve TD_memory (Done)
* TD_memory_2S_nc 
    * add "SdisS_" support in TD_memory_2S_nc
    * modify nets_trainer_LHPP2V2.py nets_trainer_LHPP2V2_base.py
        * remove OB related function in nets_trainer_LHPP2V2_base.py and nets_trainer_LHPP2V2.py
        * modify normal_accumulate_reward and OS_ForceSell_accumulate_reward to support SdisS_:
        * Open question question should the mask consideing tinpai and exceed holding be considered in V3 trainer?
            not need in V3 these unsuccessful sell need to be punished as the buy action need to reduce
* TD_memory_2S_nc_LHPP2V4
    * TD_memory_2S_nc_LHPP2V4 is removed
    * self.system_type == "LHPP2V4" will use            
        * assert self.CLN_TDmemory=="TD_memory_LHPP2V3"
        * assert self.CLN_simulator == "Simulator_LHPP2V3"
        * add specific_param in confif accordingly
        * LHPP2V4_Q_trainer1 add SdisS_ support accordingly and also add sanity check Tinpai,Exceed_limit removed by TD_memory_LHPP2V3 should not met in trainer


# seperate normal_agent (Done)

# remove sim_param (Done)

# remove unused system type to simplify the logic (Done)
* system type to removed
    * "Normal_PG":
    * "Normal_PPO":
    * "LHP_PG":
* system type explain
    normal  continue in simulator
    LHP      LHP means limit holding period:   This is one phase four action  very prelimary idea
    LHPP2    P2 means 2 phase,  OB and OS
    LHPP2V2  V2 means reuse to multibuy part av to encode signle buy holding duration
* impact files
    * net_agent.py & nets_agent_legacy.py
        * (remove) LHP and Normal is implemented together
    * nets.py     //Choose action
        * (remove) LHP and Normal is implemented  explore_brain choose action 
    * env_get_data.py
        * (remove) LHP and Normal are seperated implemented and not inherited (class env_get_data_base and class env_get_data_LHP_eval)
    * env.py
        * (Keep) simulator (normal implementation)  Simulator_LHP( LHP implementation) are inherited
    * nets_trainer_legacy.py   
        * (remove) PG_trainer and PPO_trainer in net_trainer_legacy.py for normal implentation
    * nets_trainer_LHPP2V2.py
        * (keep) LHP_PG_trainer and LHP_PPO_trainer in net_trainer_LHPP2V2.py for LHP implentation are inheritated
    * config 
        * (remove) Normal_PG, Normal_PPO, LHP_PG
* use mark ID is __system_type_remove__

# merge "LHPP2_PG" and "LHPP2_PPO" in config.py with mark ID is __merge_LHPP2_PG_PPO__ (Done)

# simplify V2_trainer (Done)
* remove     
    * def _prepare_build_train_model(self) 
    * add it direct in LHP_PG_trainer_fun and LHP_PPO_trainer_fun
* remove system_type LHPP2 
    * remove system_type LHPP2 in config.py
    * rename LHP_PG_trainer to  LHPP2V2_PG_trainer_base 
    * rename LHP_PPO_trainer to  LHPP2V2_PPO_trainer_base
* create LHPP2V2_PG_trainer and LHPP2V2_PPO_trainer
    * introduce "mask_method","accumulate_reward_method","mask_code" in specific param for P2V2
    "mask_code" should check string whether include "PVE"; 'P' means mask prob, 'V' mask state value 'E' mask entropy 
* understand origin V2
Trainer| phase |method mask | method accumulate_reward | mask
--- | --- |--- | --- | ---
LHP_PG_trainer | not applicable | _get_LHP_mask | LHP_accumulate_reward | "MPE"
LHP_PPO_trainer | not applicable |_get_LHP_mask |LHP_accumulate_reward | "MPE"
LHPP2_PG_trainer|"Train_Sell"|_get_OS_mask|LHP_accumulate_reward|"MPE"
LHPP2_PPO_trainer|"Train_Sell"|_get_OS_mask|LHP_accumulate_reward|"MPE"
LHPP2_PG_trainer2|"Train_Sell"|_get_OS_mask|LHP_accumulate_reward|"MPVE"
LHPP2_PPO_trainer2|"Train_Sell"|_get_OS_mask|LHP_accumulate_reward|"MPVE"
LHPP2_PG_trainer3|"Train_Sell"|_get_OS_mask_force_sell|LHPP2_OS_ForceSell_accumulate_reward|"MPE"
LHPP2_PPO_trainer3|"Train_Sell"|_get_OS_mask_force_sell|LHPP2_OS_ForceSell_accumulate_reward|"MPE"
LHPP2_PG_trainer4|"Train_Sell"|_get_OS_mask_force_sell|LHPP2_OS_ForceSell_accumulate_reward|"MPVE"
LHPP2_PPO_trainer4|"Train_Sell"|_get_OS_mask_force_sell|LHPP2_OS_ForceSell_accumulate_reward|"MPVE"
LHPP2_PPO_trainer5|"Train_Sell"|_get_OS_mask_force_sell|LHPP2_OS_ForceSell_accumulate_reward|"MP"


#whether possible to centerlize the action related function 

# create action class
    function 1, action04 to train action ([])
    function 2, train action ([]) to action04 
    function 2, env get actual action?
        why different simulator has different behavior on this funtion 
class name | env get actual action | holding control | beofore buy control | fabric AV | note
--- | --- | --- |--- | --- | ---  
Simulator | no(Yes on (*) | no | no |1.0 if holding > 0.0 else 0.0, potential_profit, support_view_dic["stock_SwhV1"]
Simulator_LHP | Yes | Yes | No |1.0 if holding > 0.0 else 0.0, potential_profit, support_view_dic["stock_SwhV1"]
Simulator_LHP2V2 | Yes | Yes | No | lhd=[0 for _ in range(lc.LHP+1)], lhd[self.LHP_CHP]=1, lav=lhd+[potential_profit]
Simulator_LHP2V3 | Yes | Yes | Yes | lhd=[0 for _ in range(lc.LHP+1)], lhd[self.LHP_CHP]=1, lav=lhd+[potential_profit]
        (*) change  env get actual action, choose no action from 1, 3 base on support_view["holding"] not from holding control
    move get_actual_action_from_support_view from simulators calss to action common class '''modify move_get actula to action common
    
    
#modify LHP2V2 according to new action_num set up
   also use marked         ### remove .num_action
* Input (Done)
* get_masks check logic (Done)  ### caused by 1. remove .num_action 2. add get_actual_action 3. add TD_memory_LHPP2V2
    * _get_OS_mask_force_sell  
        * origin function mask 
            *flag_force_sell= True sell
            *Fail sell, which l_support_view[idx][0, 0]["action_return_message"] in ["No_holding", "Tinpai"]:
        * as new situation Fail sell will be change to no action in get actual action from action_comm
            the new function only need to mask the flag_force_sell=True
    * _get_OS_mask
        * origin function mask 
            *Fail sell, which l_support_view[idx][0, 0]["action_return_message"] in ["No_holding", "Tinpai"]:
        * as new situation Fail sell will be change to no action in get actual action from action_comm
            the new function do nothing, keep like n_a
    * _get_LHP_mask
        as this function trainer has been removed, so this only comment for remove
    
    
* extract_y  (Done)
* choose action (Done)

* while train_num change to 2 , the buy action should be removed in the td buffer (Done)
    * add TD_memory_LHPP2V2, copy from TD_memory_LHPP2V3
    * change marked between 
        * '''# Start different compare with TD_memory_LHPP2V3  
        * '''# End different compare with TD_memory_LHPP2V3


# config.py
    * remove 
        * num_action 
            marked         ### remove .num_action
        * specific param related number of action 
    # introduce
        * trained_num_action  
        * reference_num_action
        * trained_action_type  "OB, OS, SB"
        * flag_use_ref_anumb_action
        
        '''
        * BS_num_action  This is for normal 4
        * OB_num_action  This is for only buy 2
        * OS_num_action  This is for only sell 2
        *modify config to ensure 
            * LHPP2 use BS_num_action and net_config["dense_prob"][-1]==BS_num_action
            * LHPP2V2 use OS_num_action and net_config["dense_prob"][-1]==OS_num_action
            * LHPP2V3 LHPP2V32 LHPP2V33 use OB_num_action and net_config["dense_prob"][-1]==OB_num_action
        '''
* go through 
    * net_trainer
        * LHPP2
        * LHPP2V2 (Done)
            * Input_a, Input_mask
            * Extract_y
        * LHPP2V3 (Done)
            * Input_a
            * Extract_y
        * LHPP2V32 (Done)
            * Input_a
            * Extract_y
        * LHPP2V33 (Done)
            * Input_a
            * Extract_y
        * LHPP2V4 (Done)
            * Input_a
            * Extract_y

    * nets.Explore_Brain choose_action (Done)
    
    * A3C_worker
        * eval are_ssi_handler
            * vresult.py (Done)
            * vresult_data_pbsv.py (Done)
        * worker before send to server 
            introduce action_comm marked '''introduce action_comm (Done)
remove:
    * LHPP2V3_num_action find in folder  ### marked remove LHPP2V3_num_action (Done)
    * LHPP2V4_num_action find in folder  ### marked remove LHPP2V4_num_action (Done)  
        
    
# change class name Simulator_LHP merge with Simulator_LHPP2V2 and rename to Simulator_LHPP2V2 
# rename TD_memory_2S_nc to TD_memory_2S_nc_old 

# how to handle 
* recorder_old.py
* TD_memory_old:
* train_buffer_reuse:

   