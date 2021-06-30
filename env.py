import os
import numpy as np
import config as sc
from action_comm import actionOBOS
from DBTP_Reader import DBTP_Train_Reader,DBTP_Eval_WR_Reader
from DBI_Base import hfq_toolbox
from State import AV_Handler_No_AccountInform
#roughly buy 0.0003
#roughly sell 0.0013
class Simulator_intergrated:
    def __init__(self, data_name, stock, StartI, EndI, CLN_get_data, lc, calledby):
        self.lc = lc
        self.calledby = calledby
        if self.calledby == "Explore":
            assert CLN_get_data == "DBTP_Train_Reader"
            self.i_get_data = globals()[CLN_get_data](data_name, stock, StartI, EndI, lc.PLen)
            self.Choice_reward_function, self.scale_factor, self.shift_factor, self.flag_clip =\
                lc.Choice_reward_function, lc.train_scale_factor, lc.train_shift_factor,lc.train_flag_clip
        else:
            assert self.calledby == "Eval"
            assert CLN_get_data in ["DBTP_Eval_WR_Reader"]
            self.i_get_data = globals()[CLN_get_data](data_name, stock, StartI, EndI, lc.PLen,
                                                      eval_reset_total_times=lc.evn_eval_rest_total_times)
            self.Choice_reward_function, self.scale_factor, self.shift_factor, self.flag_clip = \
                lc.Choice_reward_function, lc.eval_scale_factor, lc.eval_shift_factor,lc.eval_flag_clip

        self.rewardfun=getattr(self,self.Choice_reward_function)

        assert lc.CLN_AV_Handler in ["AV_Handler_No_AccountInform"]
        self.i_av = globals()[lc.CLN_AV_Handler](lc)
        assert self.lc.LNB==1 and self.lc.LHP==1
        self.ifhqconvert=hfq_toolbox()

    def reset(self):
        state, support_view_dic = self.i_get_data.reset_get_data()
        if "Flag_Force_Next_Reset" in support_view_dic.keys():
            Flag_Force_Next_Reset=support_view_dic["Flag_Force_Next_Reset"]
        else:
            Flag_Force_Next_Reset=False

        self.CuP=self.i_av.P_init
        self.CuPs_idx = [0 for _ in range(self.i_av.P_Num)]
        raw_av=self.i_av.fabricate_av(self.CuPs_idx, self.CuP,Flag_Force_Next_Reset,False, False)
        state.append(raw_av)

        self.YHprice, self.YHFQratio, self.YFlag_Tradable, self.Yaction= float("NaN"),float("NaN"),False,float("NaN")
        self.THFQratio,self.TFlag_Tradable,self.Taction = support_view_dic["HFQRatio"], support_view_dic["Flag_Tradable"], float("NaN")
        self.THprice = self.ifhqconvert.get_hfqprice_from_Nprice(support_view_dic["Nprice"], self.THFQratio) if self.TFlag_Tradable else float("NaN")
        self.raw_profit=float("NaN")

        return state, support_view_dic

    def step(self, Input_action):
        state, support_view_dic, _ = self.i_get_data.next_get_data()
        if self.CuP!=self.i_av.P_NB:
            assert self.CuP==self.i_av.P_HP
            assert Input_action in [2,3],Input_action  # fored by Direct sell model or fail in NB phase take action=3

        if "Flag_Force_Next_Reset" in support_view_dic.keys():   #in the Train TP reader no this item in eavl WR TP Reader this is set
            Flag_Force_Next_Reset=support_view_dic["Flag_Force_Next_Reset"]
        else:
            Flag_Force_Next_Reset=False

        if Input_action == self.i_av.CuPs_exit_actions[self.CuP] and support_view_dic["Flag_Tradable"]:
            raw_av = self.i_av.fabricate_av(self.CuPs_idx, self.CuP,Flag_Force_Next_Reset,flag_CuP_finished=True,flag_CuP_Successed=True)
            self.CuP += 1  # the fabricate_av should always have current working CuP as input change to next phase use two flag to identify
            Flag_Done= True if self.CuP==self.i_av.P_END else False  #next phase is end True or continue in next phase False
        else:
            self.CuPs_idx[self.CuP] += 1
            raw_av = self.i_av.fabricate_av(self.CuPs_idx, self.CuP, Flag_Force_Next_Reset,flag_CuP_finished=True,flag_CuP_Successed=False)
            #following two line change for LNB and LHP all is 1 and the not buy in LNB phase still need run 1 step no_action in LHP to get the potential profit for no_action
            self.CuP += 1
            Flag_Done = True if self.CuP==self.i_av.P_END else False  #exceed limitation  Error

        state.append(raw_av)

        self.YHFQratio,self.YFlag_Tradable, self.Yaction = self.THFQratio,self.TFlag_Tradable,self.Taction
        self.YHprice = self.THprice
        self.THFQratio,self.TFlag_Tradable,self.Taction = support_view_dic["HFQRatio"], support_view_dic["Flag_Tradable"], Input_action

        self.THprice = self.ifhqconvert.get_hfqprice_from_Nprice(support_view_dic["Nprice"], self.THFQratio) if self.TFlag_Tradable else float("NaN")
        #self.raw_profit = self.THprice/self.YHprice-1.0016 if  self.YFlag_Tradable and self.TFlag_Tradable else float("NaN")
        self.raw_profit = self.THprice / self.YHprice - 1.0016 if self.YFlag_Tradable and self.TFlag_Tradable else float(
            "NaN")
        reward=self.rewardfun()

        return state, reward, Flag_Done, support_view_dic, Input_action

    def WR_get_data(self):
        state, support_view_dic = self.i_get_data.reset_get_data()
        return state, support_view_dic

    #two check function for WR Eval
    def check_right_or_wrong(self):
        if self.YFlag_Tradable and self.TFlag_Tradable:
            if self.Yaction==0:
                #return "BW" if YHPrice<THPrice else "BZ" if YHPrice==THPrice else "BR"
                return 0 if self.raw_profit<0 else 1 if self.raw_profit==0 else 2
            elif self.Yaction==1:
                # return "NW" if YHPrice<THPrice else "NZ" if YHPrice==THPrice else "NR"
                return 10 if self.raw_profit<0 else 11 if self.raw_profit==0 else 12
            else:
                assert self.Yaction!=self.Yaction   #reset
                return -10
        else:
            return -1

    def check_profit(self):
        if self.YFlag_Tradable and self.TFlag_Tradable:
            if self.Yaction==0:
                return self.raw_profit +0      # [-5 to 5]  centralized at 0  should be in [-1 to 1]
            elif self.Yaction==1:
                return self.raw_profit +10      # [5, 15]    centralized at 10  should be in [9 to 11]
            else:
                assert self.Yaction!=self.Yaction   #reset
                return -10
        else:
            return -10                              # [-10]      centralized at -10  should be ==-10

    # three reward function for Train
    def RightWrong(self):
        if self.YFlag_Tradable and self.TFlag_Tradable:
            if self.Yaction == 0:
                # return "BW" if YHPrice<THPrice else "BZ" if YHPrice==THPrice else "BR"
                return -1 if self.raw_profit < 0 else 0 if self.raw_profit == 0 else 1
            elif self.Yaction==1:
                # return "NW" if YHPrice<THPrice else "NZ" if YHPrice==THPrice else "NR"
                return 1 if self.raw_profit < 0 else 0 if self.raw_profit == 0 else -1
            else:
                assert self.Yaction!=self.Yaction   #reset
                return 0
        else:
            return 0

    def RightWrong_OnlyBuy(self):
        if self.YFlag_Tradable and self.TFlag_Tradable:
            if self.Yaction == 0:
                # return "BW" if YHPrice<THPrice else "BZ" if YHPrice==THPrice else "BR"
                return -1 if self.raw_profit < 0 else 0 if self.raw_profit == 0 else 1
            elif self.Yaction==1:
                # return "NW" if YHPrice<THPrice else "NZ" if YHPrice==THPrice else "NR"
                return -0.0002
            else:
                assert self.Yaction!=self.Yaction   #reset
                return 0
        else:
            return 0


    def RightWrong_on_real(self):
        if self.YFlag_Tradable and self.TFlag_Tradable:
            adj_raw_profit=(self.raw_profit+0.0016-self.shift_factor)*self.scale_factor
            clip_adj_raw_profit= -1.0 if adj_raw_profit <-1.0 else 1.0 if adj_raw_profit>1.0 else  adj_raw_profit
            if self.Yaction == 0:
                return  clip_adj_raw_profit
            elif self.Yaction==1:
                return -clip_adj_raw_profit
            else:
                assert self.Yaction!=self.Yaction   #reset
                return 0
        else:
            return 0

    def RightWrong_on_real_OnlyBuy(self):
        if self.YFlag_Tradable and self.TFlag_Tradable:
            adj_raw_profit=(self.raw_profit+0.0016-self.shift_factor)*self.scale_factor
            clip_adj_raw_profit= -1.0 if adj_raw_profit <-1.0 else 1.0 if adj_raw_profit>1.0 else  adj_raw_profit
            if self.Yaction == 0:
                return  clip_adj_raw_profit
            elif self.Yaction==1:
                return 0
            else:
                assert self.Yaction!=self.Yaction   #reset
                return 0
        else:
            return 0


