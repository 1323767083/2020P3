import os
import config as sc
from recorder import record_sim_stock_data
from State import Phase_State, AV_Handler
from action_comm import actionOBOS
from DBTP_Reader import DBTP_Train_Reader, DBTP_Eval_Reader,DBTP_DayByDay_reader
from DBI_Base import hfq_toolbox
class env_account:
    param_shou_xu_fei = 0.00025
    param_yin_hua_shui = 0.001
    param_communicatefei = 1
    param_guohufei = 1
    def __init__(self,lc):
        self.i_hfq_tb = hfq_toolbox()
        self._account_reset()
        assert lc.env_min_invest_per_round<=lc.env_max_invest_per_round
        self.invest_per_term=lc.env_min_invest_per_round
        self.max_num_invest=int(lc.env_max_invest_per_round/lc.env_min_invest_per_round)
    #### common function
    def _sell_stock_cost(self, volume, price):
        tmp_total_money = volume * price
        sell_cost = tmp_total_money * (self.param_shou_xu_fei + self.param_yin_hua_shui) + \
                    self.param_communicatefei + \
                    self.param_guohufei * int(volume / 1000.0)
        return sell_cost

    def _buy_stock_cost(self, volume, price):
        tmp_total_money = volume * price
        buy_cost = tmp_total_money * self.param_shou_xu_fei + \
                   self.param_communicatefei + \
                   self.param_guohufei * int(volume / 1000.0)
        return buy_cost

    def _account_reset(self):
        self.volume_gu = 0
        self.total_invest = 0.0
        self.Hratio = 1.0
        self.buy_times = 0

    def reset(self):
        self._account_reset()

    def buy(self, trade_Nprice, trade_hfq_ratio):
        if self.buy_times < self.max_num_invest:
            volume_gu = int(self.invest_per_term * 0.995 / (trade_Nprice * 100)) * 100
            assert volume_gu != 0, "{0} can not buy one hand".format(self.invest_per_term)
            buy_cost= self._buy_stock_cost(volume_gu, trade_Nprice)
            current_holding_volume_gu = self.i_hfq_tb.get_update_volume_on_hfq_ratio_change(old_hfq_ratio=self.Hratio,
                                                                                new_hfq_ratio=trade_hfq_ratio,
                                                                                old_volume=self.volume_gu)
            self.volume_gu = current_holding_volume_gu + volume_gu
            self.Hratio    = trade_hfq_ratio
            self.total_invest += volume_gu * trade_Nprice + buy_cost
            self.buy_times += 1
            return True, "Success"
        else:
            return False, "Exceed_limit"

    def sell(self, trade_Nprice,trade_hfq_ratio):
        if self.volume_gu != 0:
            current_holding_volume_gu = self.i_hfq_tb.get_update_volume_on_hfq_ratio_change \
                (old_hfq_ratio=self.Hratio, new_hfq_ratio=trade_hfq_ratio, old_volume=self.volume_gu)
            sell_cost = self._sell_stock_cost(current_holding_volume_gu, trade_Nprice)
            total_money_back = current_holding_volume_gu * trade_Nprice - sell_cost
            total_money_invest = self.total_invest
            profit = total_money_back / total_money_invest - 1.0
            self._account_reset()
            return True, "Success", profit
        else:
            return False, "No_holding", 0.0

    def eval(self):
        return 1 if self.volume_gu!=0 else 0

class env_reward:
    def __init__(self,scale_factor,shift_factor, flag_clip, flag_punish_no_action):
        self.scale_factor,self.shift_factor, self.flag_clip=scale_factor, shift_factor, flag_clip
        if flag_punish_no_action:
            self.Sucess_buy = 0.0
            self.Fail_buy,self.Fail_sell,self.No_action,self.Tinpai   = -0.0001,-0.0001,-0.0001,-0.0001
        else:
            self.Sucess_buy = 0.0
            self.Fail_buy,self.Fail_sell,self.No_action,self.Tinpai   = 0.0,0.0,0.0,0.0

    def Success_sell(self,profit):
        raw_profit = (profit - self.shift_factor) * self.scale_factor
        if self.flag_clip:
            return  1 if raw_profit > 1 else -1 if raw_profit < -1 else raw_profit
        else:
            return raw_profit

    def hist_scale(self):
        return -0.3*self.scale_factor, 0.3*self.scale_factor, 0.01*self.scale_factor

class Simulator_intergrated:
    def __init__(self, data_name, stock,StartI, EndI,CLN_get_data,lc,calledby):
        self.lc=lc
        self.calledby=calledby
        self.i_PSS = globals()[lc.CLN_AV_state](self.lc,self.calledby)
        self.i_account = globals()[lc.CLN_env_account](self.lc)
        if self.calledby == "Explore":
            assert CLN_get_data == "DBTP_Train_Reader"
            self.i_reward = env_reward(lc.train_scale_factor, lc.train_shift_factor, lc.train_flag_clip,
                                             lc.train_flag_punish_no_action)
            self.i_get_data = globals()[CLN_get_data](data_name, stock,StartI, EndI,lc.PLen)
        else:
            assert self.calledby == "Eval"
            assert CLN_get_data in ["DBTP_Eval_Reader","DBTP_DayByDay_reader"]
            self.i_reward = env_reward(lc.eval_scale_factor, lc.eval_shift_factor, lc.eval_flag_clip,
                                             lc.eval_flag_punish_no_action)
            self.i_get_data = globals()[CLN_get_data](data_name, stock, StartI, EndI, lc.PLen,
                                                              eval_reset_total_times=lc.evn_eval_rest_total_times)
        if lc.flag_record_sim:
            self.i_record_sim_stock_data=record_sim_stock_data(os.path.join(sc.base_dir_RL_system,
                                                        lc.RL_system_name,"record_sim"), stock,lc.flag_record_sim)
    def reset(self):
        self.i_account.reset()  # clear account inform if last period not sold out finally
        state, support_view_dic = self.i_get_data.reset_get_data()
        support_view_dic["action"] = "Reset"  # True?
        support_view_dic["action_return_message"] = "from reset"  # True?
        raw_av=self.i_PSS.reset_phase_state()
        state.append(raw_av)
        if self.lc.flag_record_sim:
            self.i_record_sim_stock_data.saver([state, support_view_dic],str(support_view_dic["DateI"]))
        return state, support_view_dic

    def step_comm(self, adj_action):
        action_str = self.lc.action_type_dict[adj_action]
        state, support_view_dic, _ = self.i_get_data.next_get_data()
        support_view_dic["action_taken"] = action_str

        if not support_view_dic["Flag_Tradable"]:
            reward = self.i_reward.Tinpai
            return_message="Tinpai"
            return return_message, reward, state, support_view_dic

        if action_str == "Buy":  # This is only support single buy
            flag_operation_result, return_message = self.i_account.buy(support_view_dic["Nprice"],support_view_dic["HFQRatio"])
            if flag_operation_result and return_message == "Success":
                reward = self.i_reward.Sucess_buy
            elif not flag_operation_result and return_message == "Tinpai":
                assert False,"This should be check ahead by Flag_Tradable"
            elif not flag_operation_result and return_message == "Exceed_limit":
                reward = self.i_reward.Fail_buy
            else:
                raise ValueError("unexpected buy return message {0}".format(return_message))
        elif action_str == "Sell":  # sell
            flag_operation_result, return_message, current_profit = self.i_account.sell(support_view_dic["Nprice"],support_view_dic["HFQRatio"])
            if flag_operation_result  and return_message == "Success":
                reward = self.i_reward.Success_sell(current_profit)
            elif not flag_operation_result  and return_message == "Tinpai":
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

        if self.lc.flag_record_sim:
            self.i_record_sim_stock_data.saver([state, support_view_dic],str(support_view_dic["DateI"]))
        return return_message,reward,state, support_view_dic

    '''
    def step(self,action):
        adj_action = self.i_PSS.check_need_force_state(action)
        if self.calledby=="Eval":
            if adj_action==0 and action!=0: # if force buy happen in eval, keep original action
                adj_action=action
        #TODO check this check should included in the state param setting
        adj_action_str = self.lc.action_type_dict[adj_action]
        return_message,reward,state, support_view_dic=self.step_comm(adj_action_str)
        support_view_dic["action_return_message"]=return_message
        Done_flag=self.i_PSS.update_phase_state(support_view_dic,adj_action,return_message)
        self.i_PSS.fabricate_av_and_update_support_view(state, support_view_dic)
        return state, reward,Done_flag,support_view_dic
    '''

    def step(self,action):
        adj_action = self.i_PSS.check_need_force_state(action)
        return_message,reward,state, support_view_dic=self.step_comm(adj_action)
        support_view_dic["action_return_message"]=return_message
        Done_flag, raw_av,actual_action=self.i_PSS.update_phase_state(adj_action, return_message)
        state.append(raw_av)
        return state, reward,Done_flag,support_view_dic,actual_action