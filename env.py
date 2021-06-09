import os
import config as sc
from recorder import record_sim_stock_data
from State import *
from action_comm import actionOBOS
from DBTP_Reader import DBTP_Train_Reader,DBTP_Eval_WR_Reader
from DBI_Base import hfq_toolbox
#roughly buy 0.0003
#roughly sell 0.0013

class env_account:
    param_yin_hua_shui          = 0.001
    param_guo_hu_fei            = 0.0002
    param_jin_shou_fei          = 0.0000487
    param_quan_shang_yong_jin   = 0.00025
    def __init__(self,lc, stock):
        self.lc, self.stock=lc, stock
        self.i_hfq_tb = hfq_toolbox()
        self.TransIDI = -1
        self._account_reset()
        assert lc.env_min_invest_per_round<=lc.env_max_invest_per_round
        self.invest_per_term=lc.env_min_invest_per_round
        self.max_num_invest=int(lc.env_max_invest_per_round//lc.env_min_invest_per_round)

    #Following is caculated according SH market, SZ market seems no guohufei , as the difference is small so not considered here
    def _buy_stock_cost(self, gu, Nprice):
        temp_trans_money = gu * Nprice
        temp_guo_hu_fei = gu * self.param_guo_hu_fei
        temp_quan_shang_yong_jin = temp_trans_money*self.param_quan_shang_yong_jin

        yi_hua_sui = 0
        guo_hu_fei = 1 if temp_guo_hu_fei<1 else temp_guo_hu_fei
        jin_shou_fei = temp_trans_money*self.param_jin_shou_fei
        quan_shang_yong_jin = 5 if temp_quan_shang_yong_jin<5 else temp_quan_shang_yong_jin

        return guo_hu_fei+jin_shou_fei+quan_shang_yong_jin

    def _sell_stock_cost(self, gu, Nprice):
        temp_trans_money = gu * Nprice
        temp_guo_hu_fei = gu * self.param_guo_hu_fei
        temp_quan_shang_yong_jin = temp_trans_money * self.param_quan_shang_yong_jin

        yi_hua_sui = temp_trans_money*self.param_yin_hua_shui
        guo_hu_fei = 1 if temp_guo_hu_fei<1 else temp_guo_hu_fei
        jin_shou_fei = temp_trans_money*self.param_jin_shou_fei
        quan_shang_yong_jin = 5 if temp_quan_shang_yong_jin<5 else temp_quan_shang_yong_jin

        return yi_hua_sui + guo_hu_fei + jin_shou_fei + quan_shang_yong_jin


    def _account_reset(self):
        self._clean_holding()
        self._clean_step_result()
        self.TransIDI += 1

    def _clean_holding(self):
        self.Holding_Gu = 0
        self.Holding_Invest = 0.0
        self.Holding_HRatio = 1.0
        self.Holding_NPrice =0.0
        self.Buy_Times = 0.0

    def _clean_step_result(self):
        self.Buy_Invest = 0.0
        self.Buy_NPrice=0.0
        self.Sell_Return = 0.0
        self.Sell_NPrice = 0.0
        self.Sell_Earn=0.0
        self.Tinpai_huaizhang = 0.0


    def reset(self):
        self._account_reset()

    def buy(self, trade_Nprice, trade_hfq_ratio):
        self._clean_step_result()
        if self.Buy_Times >= self.max_num_invest:
            return False, "Exceed_limit"
        else:
            gu = int(self.invest_per_term * 0.999 / (trade_Nprice * 100)) * 100
            assert gu != 0, "{0} can not buy one hand {1}".format(self.invest_per_term,self.stock)
            buy_cost= self._buy_stock_cost(gu, trade_Nprice)
            self.Buy_Invest = gu * trade_Nprice + buy_cost
            self.Buy_NPrice =trade_Nprice

            current_holding_gu = self.i_hfq_tb.get_update_volume_on_hfq_ratio_change(old_hfq_ratio=self.Holding_HRatio,
                                                                                new_hfq_ratio=trade_hfq_ratio,
                                                                                old_volume=self.Holding_Gu)
            self.Holding_Gu = current_holding_gu + gu
            self.Holding_Invest += self.Buy_Invest
            self.Buy_Times += 1
            self.Holding_HRatio    = trade_hfq_ratio
            self.Holding_NPrice= trade_Nprice
            return True, "Success"

    def sell(self, trade_Nprice,trade_hfq_ratio):
        self._clean_step_result()
        if self.Holding_Gu != 0:
            current_holding_gu = self.i_hfq_tb.get_update_volume_on_hfq_ratio_change \
                (old_hfq_ratio=self.Holding_HRatio, new_hfq_ratio=trade_hfq_ratio, old_volume=self.Holding_Gu)
            sell_cost = self._sell_stock_cost(current_holding_gu, trade_Nprice)
            self.Sell_Return = current_holding_gu * trade_Nprice - sell_cost
            self.Sell_Earn=self.Sell_Return-self.Holding_Invest
            self.Sell_NPrice =trade_Nprice
            profit = self.Sell_Earn / self.Holding_Invest
            self._clean_holding()
            return True, "Success", profit
        else:
            return False, "No_holding", 0.0

    def no_action(self):
        self._clean_step_result()

    def eval_profit(self, Close_NPrice):
        if self.Holding_NPrice==0:
            return 0
        else:
            if Close_NPrice==0:  #only first day or first days
                assert self.Holding_NPrice==0, "should be the first day of period with out holding in DBTP_reader could set close price is 0"
                return 0
            else:
                return Close_NPrice/self.Holding_NPrice-1

    def get_inform_for_AV(self):
        return [getattr(self,title) for title in self.lc.account_inform_titles]

class Reward_intergrated:
    def __init__(self,Choice_reward_function,scale_factor,shift_factor, flag_clip):
        self.Choice_reward_function,self.scale_factor, self.shift_factor, self.flag_clip \
            = Choice_reward_function,scale_factor, shift_factor, flag_clip
        self.Choice_reward_function,self.scale_factor= Choice_reward_function,scale_factor
        self.YNprice, self.YHFQratio, self.YFlag_Tradable, self.Yaction= float("NaN"),float("NaN"),False,float("NaN")
        self.TNprice, self.THFQratio, self.TFlag_Tradable, self.Taction= float("NaN"),float("NaN"),False,float("NaN")
        self.ifhqconvert=hfq_toolbox()
        if self.Choice_reward_function == "legacy":
            self.flag_RW = False
            rewardfun = lambda x: 0.0
            self.Success_sell = self.legacy_Success_sell
            self.Sucess_buy = rewardfun
            self.Fail_buy = rewardfun
            self.Fail_sell = rewardfun
            self.No_action = rewardfun
            self.Tinpai = rewardfun

        if self.Choice_reward_function == "RightWrong":
            self.flag_RW=True
            rewardfun = self.get_reward_01
            self.Success_sell = lambda x: rewardfun()
            self.Sucess_buy = lambda x: rewardfun()
            self.Fail_buy = lambda x: rewardfun()
            self.Fail_sell = lambda x: rewardfun()
            self.No_action = lambda x: rewardfun()
            self.Tinpai = lambda x: rewardfun()

        elif self.Choice_reward_function == "RightWrong_on_real":
            self.flag_RW = True
            rewardfun = self.get_reward_real
            self.Success_sell = lambda x: rewardfun()
            self.Sucess_buy = lambda x: rewardfun()
            self.Fail_buy = lambda x: rewardfun()
            self.Fail_sell = lambda x: rewardfun()
            self.No_action = lambda x: rewardfun()
            self.Tinpai = lambda x: rewardfun()

        else:
            assert False


    def update_today(self,Nprice,HFQratio,Flag_Tradable, Acutal_action):
        self.YNprice, self.YHFQratio,self.YFlag_Tradable, self.Yaction = self.TNprice, self.THFQratio,self.TFlag_Tradable,self.Taction
        self.TNprice, self.THFQratio,self.TFlag_Tradable,self.Taction = Nprice, HFQratio, Flag_Tradable,Acutal_action

    #legacy for vresult_data_reward.py
    def hist_scale(self):
        return -0.3*self.scale_factor, 0.3*self.scale_factor, 0.01*self.scale_factor


    #two check function for WR Eval
    def check_right_or_wrong(self):
        assert self.flag_RW
        if self.YFlag_Tradable and self.TFlag_Tradable:
            YHPrice = self.ifhqconvert.get_hfqprice_from_Nprice(self.YNprice, self.YHFQratio)
            THPrice = self.ifhqconvert.get_hfqprice_from_Nprice(self.TNprice, self.THFQratio)
            if self.Yaction==0:
                #return "BW" if YHPrice<THPrice else "BZ" if YHPrice==THPrice else "BR"
                return 0 if YHPrice>THPrice else 1 if YHPrice==THPrice else 2
            else:
                # return "NW" if YHPrice<THPrice else "NZ" if YHPrice==THPrice else "NR"
                return 10 if YHPrice>THPrice else 11 if YHPrice==THPrice else 12
        else:
            return -1

    def check_profit(self):
        # roughly buy 0.0003
        # roughly sell 0.0013

        assert self.flag_RW
        if self.YFlag_Tradable and self.TFlag_Tradable:
            YHPrice = self.ifhqconvert.get_hfqprice_from_Nprice(self.YNprice, self.YHFQratio)
            THPrice = self.ifhqconvert.get_hfqprice_from_Nprice(self.TNprice, self.THFQratio)
            if self.Yaction==0:
                shift_factor=0
                return THPrice/YHPrice-1.0016 +shift_factor      # [-5 to 5]  centralized at 0  should be in [-1 to 1]
            else:
                shift_factor=10
                return THPrice/YHPrice-1.0016 +shift_factor      # [5, 15]    centralized at 10  should be in [9 to 11]
        else:
            shift_factor = -10
            return shift_factor                              # [-10]      centralized at -10  should be ==-10

    # three reward function for Train
    def get_reward_01(self):
        raw_reward=self.check_right_or_wrong()
        adj_reward=raw_reward%10
        if self.Yaction==0:
            if adj_reward==0:
                return -1
            elif adj_reward==1:
                return 0
            elif adj_reward==2:
                return 1
            else:
                assert raw_reward <0,"raw_reward = {0} adj_reward={1}".format(raw_reward,adj_reward)
                return 0
        else:
            if adj_reward==0:
                return 1
            elif adj_reward==1:
                return 0
            elif adj_reward==2:
                return -1
            else:
                assert raw_reward <0,"raw_reward = {0} adj_reward={1}".format(raw_reward,adj_reward)
                return 0

    def get_reward_real(self):
        if self.YFlag_Tradable and self.TFlag_Tradable:
            YHPrice = self.ifhqconvert.get_hfqprice_from_Nprice(self.YNprice, self.YHFQratio)
            THPrice = self.ifhqconvert.get_hfqprice_from_Nprice(self.TNprice, self.THFQratio)
            raw_profit=(THPrice / YHPrice - 1.0016)*self.scale_factor
            raw_profit= -1.0 if raw_profit <-1.0 else 1.0 if raw_profit>1.0 else  raw_profit
            return  raw_profit if self.Yaction==0 else -raw_profit
        else:
            return 0


    def legacy_Success_sell(self, profit):
        raw_profit = (profit - self.shift_factor) * self.scale_factor
        if self.flag_clip:
            return 1 if raw_profit > 1 else -1 if raw_profit < -1 else raw_profit
        else:
            return raw_profit

class Simulator_intergrated:
    def __init__(self, data_name, stock,StartI, EndI,CLN_get_data,lc,calledby):
        self.lc=lc
        self.calledby=calledby
        self.i_PSS = globals()[lc.CLN_AV_state](self.lc,self.calledby)
        if self.calledby == "Explore":
            assert CLN_get_data == "DBTP_Train_Reader"
            self.i_account = globals()[lc.CLN_env_account](self.lc, stock)
            self.i_reward = Reward_intergrated(lc.Choice_reward_function,lc.train_scale_factor, lc.train_shift_factor, lc.train_flag_clip)
            self.i_get_data = globals()[CLN_get_data](data_name, stock,StartI, EndI,lc.PLen)
        else:
            assert self.calledby == "Eval"
            assert CLN_get_data in ["DBTP_Eval_WR_Reader"]
            self.i_reward = Reward_intergrated(lc.Choice_reward_function,lc.eval_scale_factor, lc.eval_shift_factor, lc.eval_flag_clip)
            self.i_get_data = globals()[CLN_get_data](data_name, stock, StartI, EndI, lc.PLen,
                                                              eval_reset_total_times=lc.evn_eval_rest_total_times)

            self.i_account = globals()[lc.CLN_env_account](self.lc, stock)

        if lc.flag_record_sim:
            self.i_record_sim_stock_data=record_sim_stock_data(os.path.join(sc.base_dir_RL_system,
                                                        lc.RL_system_name,"record_sim"), stock,lc.flag_record_sim)

    def reset(self):
        self.i_account.reset()  # clear account inform if last period not sold out finally
        state, support_view_dic = self.i_get_data.reset_get_data()
        self.i_reward.update_today(support_view_dic["Nprice"], support_view_dic["HFQRatio"],
                              support_view_dic["Flag_Tradable"], 1) # 1 means not buy take no action
        support_view_dic["action"] = "Reset"  # True?
        support_view_dic["action_return_message"] = "from reset"  # True?
        l_av_inform=self.i_account.get_inform_for_AV()
        l_av_inform.append(support_view_dic["DateI"])
        l_av_inform.append(int(support_view_dic["Stock"][2:]))
        l_av_inform.append(self.i_account.eval_profit(self.i_get_data.get_CloseNPrice()))
        if "Flag_Force_Next_Reset" in support_view_dic.keys():
            Flag_Force_Next_Reset=support_view_dic["Flag_Force_Next_Reset"]
        else:
            Flag_Force_Next_Reset=False
        raw_av=self.i_PSS.reset_phase_state(l_av_inform,Flag_Force_Next_Reset)
        state.append(raw_av)
        if self.lc.flag_record_sim:
            self.i_record_sim_stock_data.saver([state, support_view_dic],str(support_view_dic["DateI"]))
        return state, support_view_dic

    def step_comm(self, adj_action):
        action_str = self.lc.action_type_dict[adj_action]
        state, support_view_dic, _ = self.i_get_data.next_get_data()
        self.i_reward.update_today(support_view_dic["Nprice"],support_view_dic["HFQRatio"],support_view_dic["Flag_Tradable"],adj_action)
        support_view_dic["action_taken"] = action_str

        if not support_view_dic["Flag_Tradable"]:
            self.i_account.no_action()  #this is to clean the step inform like the buy inform  ToDo double check
            reward = self.i_reward.Tinpai(float("NaN"))
            return_message="Tinpai"
            return return_message, reward, state, support_view_dic

        if action_str == "Buy":  # This is only support single buy
            flag_operation_result, return_message = self.i_account.buy(support_view_dic["Nprice"],support_view_dic["HFQRatio"])
            if flag_operation_result and return_message == "Success":
                reward = self.i_reward.Sucess_buy(float("NaN"))
            elif not flag_operation_result and return_message == "Tinpai":
                assert False,"This should be check ahead by Flag_Tradable"
            elif not flag_operation_result and return_message == "Exceed_limit":
                reward = self.i_reward.Fail_buy(float("NaN"))
            else:
                raise ValueError("unexpected buy return message {0}".format(return_message))
        elif action_str == "Sell":  # sell
            flag_operation_result, return_message, current_profit = self.i_account.sell(support_view_dic["Nprice"],support_view_dic["HFQRatio"])
            if flag_operation_result  and return_message == "Success":
                reward = self.i_reward.Success_sell(current_profit)

            elif not flag_operation_result  and return_message == "Tinpai":
                assert False, "This should be check ahead by Flag_Tradable"
            elif not flag_operation_result  and return_message == "No_holding":
                reward = self.i_reward.Fail_sell(float("NaN"))
            else:
                raise ValueError("unexpected sell return message {0}".format(return_message))
        elif action_str == "No_action":
            return_message = "No_action"  # if want no action always no action as result
            self.i_account.no_action()
            reward = self.i_reward.No_action(float("NaN"))
        elif action_str == "No_trans":
            return_message = "No_trans"  # if want no action always no action as result
            self.i_account.no_action()
            reward = self.i_reward.No_action(float("NaN"))
        else:
            raise ValueError("does not support action {0}".format(action_str))
        if self.lc.flag_record_sim:
            self.i_record_sim_stock_data.saver([state, support_view_dic],str(support_view_dic["DateI"]))
        return return_message,reward,state, support_view_dic

    def step(self,Input_action):
        action, PSS_action = Input_action, 0
        adj_action = self.i_PSS.check_need_force_state(action)
        if adj_action!=action and action==0:
            PSS_action=0
        return_message,reward,state, support_view_dic=self.step_comm(adj_action)
        support_view_dic["action_return_message"]=return_message
        l_av_inform=self.i_account.get_inform_for_AV()
        l_av_inform.append(support_view_dic["DateI"])
        l_av_inform.append(int(support_view_dic["Stock"][2:]))
        l_av_inform.append(self.i_account.eval_profit(self.i_get_data.get_CloseNPrice()))
        if "Flag_Force_Next_Reset" in support_view_dic.keys():
            Flag_Force_Next_Reset=support_view_dic["Flag_Force_Next_Reset"]
        else:
            Flag_Force_Next_Reset=False
        Done_flag, raw_av, actual_action = self.i_PSS.update_phase_state(adj_action, return_message, l_av_inform, PSS_action,Flag_Force_Next_Reset)
        state.append(raw_av)
        return state, reward,Done_flag,support_view_dic,actual_action

    def WR_get_data(self):
        state, support_view_dic = self.i_get_data.reset_get_data()
        return state, support_view_dic
        #self.iRW.update_today(support_view_dic["Nprice"], support_view_dic["HFQRatio"],
        #                      support_view_dic["Flag_Tradable"], 1) # 1 means not buy take no action, similar as reset as it called like reset
