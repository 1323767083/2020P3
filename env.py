import os
import random
import numpy as np
from data_common import API_SH_SZ_total_sl,API_SH_sl, API_SZ_sl,API_trade_date,ginfo_one_stock,hfq_toolbox
from data_T5 import R_T5,R_T5_scale,R_T5_balance, R_T5_skipSwh,R_T5_skipSwh_balance
import config as sc
from recorder import record_sim_stock_data
from env_get_data import *
from env_ar import *
from av_state import Phase_State_V8,Phase_State_V3__1,Phase_State_V3__2,Phase_State_V2
from action_comm import actionOBOS

def init_gc(lgc):
    global lc
    lc=lgc
    env_get_data_init(lgc)
    env_ar_init(lgc)
    global Caccount                         #,Creward,Cenv_read_data
    Caccount=globals()[lc.CLN_env_account]


class Simulator_intergrated:            #Simulator_LHPP2V8:
    def __init__(self, data_name, stock,called_by):
        assert not lc.flag_multi_buy
        self.called_by = called_by
        self.data_name = data_name
        self.i_PSS = globals()[lc.CLN_AV_state]()
        self.i_account = Caccount()
        if self.called_by == "explore":
            assert lc.CLN_env_get_data_train == "env_get_data_LHP_train"
            #self.i_reward = env_reward_basic(lc.train_reward_scaler_factor,lc.train_reward_type,lc)
            self.i_reward = env_reward_basic(lc.train_scale_factor, lc.train_shift_factor, lc.train_flag_clip,
                                             lc.train_flag_punish_no_action)
            self.i_get_data = globals()[lc.CLN_env_get_data_train](self.data_name, stock,
                            flag_episode_random_start=lc.env_flag_random_start_in_episode_for_explore)
        else:
            assert self.called_by == "eval"
            assert lc.CLN_env_get_data_eval == "env_get_data_LHP_eval"
            #self.i_reward = env_reward_basic(lc.eval_reward_scaler_factor,lc.eval_reward_type,lc)
            self.i_reward = env_reward_basic(lc.eval_scale_factor, lc.eval_shift_factor, lc.eval_flag_clip,
                                             lc.eval_flag_punish_no_action)
            self.i_get_data = globals()[lc.CLN_env_get_data_eval](self.data_name, stock,
                                flag_episode_random_start=lc.env_flag_random_start_in_episode_for_eval)

        if not self.i_get_data.flag_proper_data_avaliable:
            raise ValueError("{0} {1} does not exisit".format(self.data_name, stock))

        if lc.flag_record_sim:
            self.i_record_sim_stock_data=record_sim_stock_data(os.path.join(sc.base_dir_RL_system,
                                                        lc.RL_system_name,"record_sim"), stock,lc.flag_record_sim)
    def reset(self):
        self.i_PSS.reset()
        self.i_account.reset()  # clear account inform if last period not sold out finally
        state, support_view_dic = self.i_get_data.reset_get_data()
        support_view_dic["action_return_message"] = "from reset"  # True?
        self.i_PSS.fabricate_av_and_update_support_view(state, support_view_dic,False)
        #support_view_dic["holding"] = 0
        if lc.flag_record_sim:
            self.i_record_sim_stock_data.saver([state, support_view_dic],support_view_dic["date"])
        return state, support_view_dic

    def step_comm(self, action_str):
        state, support_view_dic, _ = self.i_get_data.next_get_data()
        if lc.flag_record_sim:
            self.i_record_sim_stock_data.saver([state, support_view_dic],support_view_dic["date"])
        if action_str == "buy":  # This is only support single buy
            support_view_dic["action_taken"] = "Buy"
            flag_operation_result, return_message = self.i_account.buy(support_view_dic["this_trade_day_Nprice"],
                                                                       support_view_dic["this_trade_day_hfq_ratio"],
                                                                       support_view_dic["stock"],
                                                                       support_view_dic["date"])
            if flag_operation_result and return_message == "Success":
                reward = self.i_reward.Sucess_buy
            elif not flag_operation_result and return_message == "Tinpai":
                reward = self.i_reward.Tinpai
            elif not flag_operation_result and return_message == "Exceed_limit":
                reward = self.i_reward.Fail_buy
            else:
                raise ValueError("unexpected buy return message {0}".format(return_message))
        elif action_str == "sell":  # sell
            support_view_dic["action_taken"] = "Sell"
            flag_operation_result, return_message, current_profit = self.i_account.sell(
                support_view_dic["this_trade_day_Nprice"],
                support_view_dic["this_trade_day_hfq_ratio"],
                support_view_dic["stock"],
                support_view_dic["date"])
            if flag_operation_result  and return_message == "Success":
                reward = self.i_reward.Success_sell(current_profit)
            elif not flag_operation_result  and return_message == "Tinpai":
                reward = self.i_reward.Tinpai
            elif not flag_operation_result  and return_message == "No_holding":
                reward = self.i_reward.Fail_sell
            else:
                raise ValueError("unexpected sell return message {0}".format(return_message))
        elif action_str == "no_action":
            support_view_dic["action_taken"] = "No_action"
            return_message = "No_action"  # if want no action always no action as result
            reward = self.i_reward.No_action
        elif action_str == "no_trans":
            support_view_dic["action_taken"] = "No_trans"
            return_message = "No_trans"  # if want no action always no action as result
            reward = self.i_reward.No_action
        else:
            raise ValueError("does not support action {0}".format(action_str))
        return return_message,reward,state, support_view_dic

    def step(self,action):
        adj_action=self.i_PSS.check_need_force_state(action)
        flag_force_sell=True if adj_action==2 and action!=2 else False
        adj_action_str=lc.action_type_dict[adj_action]
        return_message,reward,state, support_view_dic=self.step_comm(adj_action_str)
        support_view_dic["action_return_message"]=return_message
        Done_flag=self.i_PSS.update_phase_state(support_view_dic,adj_action,return_message)
        self.i_PSS.fabricate_av_and_update_support_view(state, support_view_dic,flag_force_sell)
        return state, reward,Done_flag,support_view_dic


    def data_reset(self): # this is copy from old code at 20200207, fill right, but not tested
        if self.called_by=="explore":
            assert False, "{0}  does not support data_reset as only called by explore process not eval process".format(self.__class__.__name__)
        else:
            assert self.called_by=="eval"
            self.i_get_data.data_reset()



