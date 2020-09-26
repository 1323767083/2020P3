import os
#from data_common import API_SH_SZ_total_sl,API_SH_sl, API_SZ_sl,API_trade_date,ginfo_one_stock,hfq_toolbox
#from data_T5 import R_T5,R_T5_scale,R_T5_balance, R_T5_skipSwh,R_T5_skipSwh_balance
import config as sc
from recorder import record_sim_stock_data
#from env_get_data import *
from env_ar import *
from av_state import Phase_State_V8,Phase_State_V3__1,Phase_State_V3__2,Phase_State_V2
from action_comm import actionOBOS
from DBTP_Reader import DBTP_Train_Reader, DBTP_Eval_Reader,DBTP_Continue_Reader

def init_gc(lgc):
    global lc
    lc=lgc
    env_ar_init(lgc)

class Simulator_intergrated:
    def __init__(self, data_name, stock,StartI, EndI,CLN_get_data,calledby):
        assert not lc.flag_multi_buy
        self.calledby=calledby
        self.i_PSS = globals()[lc.CLN_AV_state]()
        self.i_account = globals()[lc.CLN_env_account]()
        if self.calledby == "Explore":
            #assert lc.CLN_env_get_data_train == "DBTP_Train_Reader"
            assert CLN_get_data == "DBTP_Train_Reader"
            self.i_reward = env_reward_basic(lc.train_scale_factor, lc.train_shift_factor, lc.train_flag_clip,
                                             lc.train_flag_punish_no_action)
            #self.i_get_data = globals()[lc.CLN_env_get_data_train](data_name, stock,StartI, EndI,lc.PLen)
            self.i_get_data = globals()[CLN_get_data](data_name, stock,StartI, EndI,lc.PLen)
        else:
            assert self.calledby == "Eval"
            #assert lc.CLN_env_get_data_eval == "DBTP_Eval_Reader"
            assert CLN_get_data in ["DBTP_Eval_Reader", "DBTP_Continue_Reader"]
            self.i_reward = env_reward_basic(lc.eval_scale_factor, lc.eval_shift_factor, lc.eval_flag_clip,
                                             lc.eval_flag_punish_no_action)
            #self.i_get_data = globals()[lc.CLN_env_get_data_eval](data_name, stock,StartI, EndI,lc.PLen,
            #                                                      eval_reset_total_times=lc.evn_eval_rest_total_times)
            self.i_get_data = globals()[CLN_get_data](data_name, stock, StartI, EndI, lc.PLen,
                                                              eval_reset_total_times=lc.evn_eval_rest_total_times)

        if lc.flag_record_sim:
            self.i_record_sim_stock_data=record_sim_stock_data(os.path.join(sc.base_dir_RL_system,
                                                        lc.RL_system_name,"record_sim"), stock,lc.flag_record_sim)
    def reset(self):
        self.i_PSS.reset()
        self.i_account.reset()  # clear account inform if last period not sold out finally
        state, support_view_dic = self.i_get_data.reset_get_data()
        support_view_dic["action"] = "Reset"  # True?
        support_view_dic["action_return_message"] = "from reset"  # True?
        self.i_PSS.fabricate_av_and_update_support_view(state, support_view_dic)
        #support_view_dic["holding"] = 0
        if lc.flag_record_sim:
            self.i_record_sim_stock_data.saver([state, support_view_dic],str(support_view_dic["DateI"]))
        return state, support_view_dic

    def step_comm(self, action_str):
        state, support_view_dic, _ = self.i_get_data.next_get_data()

        support_view_dic["action_taken"] = action_str
        if not support_view_dic["Flag_Tradable"]:
            reward = self.i_reward.Tinpai
            return_message="Tinpai"
            return return_message, reward, state, support_view_dic


        if action_str == "Buy":  # This is only support single buy
            flag_operation_result, return_message = self.i_account.buy(support_view_dic["Nprice"],
                                                                       support_view_dic["HFQRatio"],
                                                                       support_view_dic["Stock"],
                                                                       support_view_dic["DateI"])
            if flag_operation_result and return_message == "Success":
                reward = self.i_reward.Sucess_buy
            elif not flag_operation_result and return_message == "Tinpai":
                #reward = self.i_reward.Tinpai
                assert False,"This should be check ahead by Flag_Tradable"
            elif not flag_operation_result and return_message == "Exceed_limit":
                reward = self.i_reward.Fail_buy
            else:
                raise ValueError("unexpected buy return message {0}".format(return_message))
        elif action_str == "Sell":  # sell
            flag_operation_result, return_message, current_profit = self.i_account.sell(
                support_view_dic["Nprice"],
                support_view_dic["HFQRatio"],
                support_view_dic["Stock"],
                support_view_dic["DateI"])
            if flag_operation_result  and return_message == "Success":
                reward = self.i_reward.Success_sell(current_profit)
            elif not flag_operation_result  and return_message == "Tinpai":
                #reward = self.i_reward.Tinpai
                assert False, "This should be check ahead by Flag_Tradable"
            elif not flag_operation_result  and return_message == "No_holding":
                reward = self.i_reward.Fail_sell
            else:
                raise ValueError("unexpected sell return message {0}".format(return_message))
        elif action_str == "No_action":
            return_message = "No_action"  # if want no action always no action as result
            reward = self.i_reward.No_action
        elif action_str == "No_trans":
            return_message = "No_trans"  # if want no action always no action as result
            reward = self.i_reward.No_action
        else:
            raise ValueError("does not support action {0}".format(action_str))

        if lc.flag_record_sim:
            self.i_record_sim_stock_data.saver([state, support_view_dic],str(support_view_dic["DateI"]))
        return return_message,reward,state, support_view_dic

    def step(self,action):
        adj_action = self.i_PSS.check_need_force_state(action)
        if self.calledby=="Eval":
            if adj_action==0 and action!=0: # if force buy happen in eval, keep original action
                adj_action=action
        adj_action_str = lc.action_type_dict[adj_action]
        return_message,reward,state, support_view_dic=self.step_comm(adj_action_str)
        support_view_dic["action_return_message"]=return_message
        Done_flag=self.i_PSS.update_phase_state(support_view_dic,adj_action,return_message)
        #self.i_PSS.fabricate_av_and_update_support_view(state, support_view_dic,flag_force_sell)
        self.i_PSS.fabricate_av_and_update_support_view(state, support_view_dic)
        return state, reward,Done_flag,support_view_dic

