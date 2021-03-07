import os, random
import DBI_Base
import pandas as pd
import numpy as np
from State import *
from Buy_Strategies import Buy_Strategies
class Eval_CC_base:
    seed_CC_Sub="EvalGroup{0}"
    def __init__(self,system_working_dir):
        self.CC_log_dir = os.path.join(system_working_dir, "CC")
        if not os.path.exists(self.CC_log_dir): os.mkdir(self.CC_log_dir)

    def get_SL_fnwp(self):
        return os.path.join(self.CC_log_dir,"SL_in_order.csv")

    def get_money_in_hand(self,ET,group_idx):
        return os.path.join(self.CC_log_dir, self.seed_CC_Sub.format(group_idx),"ET{0}_money_in_hand.csv".format(ET))

    def get_Record_fnwp(self,ET, group_idx):
        return os.path.join(self.CC_log_dir,self.seed_CC_Sub.format(group_idx), "ET{0}.csv".format(ET))

    def get_action_decision_log_fnwp(self,ET, group_idx):
        return os.path.join(self.CC_log_dir,self.seed_CC_Sub.format(group_idx), "ET{0}_action_decision.csv".format(ET))

class Eval_CC(Eval_CC_base):
    def __init__(self,lc):
        Eval_CC_base.__init__(self,lc.system_working_dir)
        self.lc = lc
        self.iSL = DBI_Base.StockList(self.lc.SLName)
        self.l_CC_GroupIdx, self.ll_CC_ProcessIdx=self.Get_V3EvalCC_ProcessIdx_Range()

        if len(self.l_CC_GroupIdx)!=0:
            self.StartEndP =[]
            start_idx=0
            total_sl=[]
            for sl in [self.iSL.Get_Eval_SubProcess_SL(lc, self.l_CC_GroupIdx[0], process_idx) for process_idx in self.ll_CC_ProcessIdx[0]]:
                total_sl.extend(sl)
                self.StartEndP.append(start_idx)
                start_idx+=len(sl)
            else:
                self.StartEndP.append(start_idx)

            self.l_CC_last_eval_date=[99999999 for _ in self.l_CC_GroupIdx ] #999999 us make the first round start trigger the print

            self.lll_CC_OutBuffer=[[[] for _ in self.ll_CC_ProcessIdx[location_idx]] for location_idx, _ in enumerate(self.l_CC_GroupIdx)]

            self.log_titles = self.lc.account_inform_titles + self.lc.simulator_inform_titles + self.lc.PSS_inform_titles
            self.l_df = [pd.DataFrame(columns=self.log_titles) for _ in self.l_CC_GroupIdx]
            self.dfMoney_titles=["DateI","Money_in_hand","Eval_holding","Sell_money_on_the_way","Eval_Ttotal","Tinpai_huaizhang"]
            self.l_dfMoney=[pd.DataFrame(columns=self.dfMoney_titles)  for _ in self.l_CC_GroupIdx]
            self.ADlog_titles = ["DateI", "not_buy_due_limit", "not_buy_due_low_profit", "sell_due_low_profit","multibuy"]
            self.l_dfADlog = [pd.DataFrame(columns=self.ADlog_titles)   for _ in self.l_CC_GroupIdx]

            for group_idx in self.l_CC_GroupIdx:
                sub_dir=os.path.join(self.CC_log_dir,self.seed_CC_Sub.format(group_idx))
                if not os.path.exists(sub_dir): os.mkdir(sub_dir)

            self.l_TotalInvest=[self.lc.l_CC_group_invest_total_money[GroupIdx] for GroupIdx in self.l_CC_GroupIdx]
            self.i_cav = globals()[lc.CLN_AV_Handler](lc)

            pd.DataFrame(total_sl, columns=["Stock"]).to_csv(self.get_SL_fnwp(), index=False)
            self.fakeap=[[np.NaN,np.NaN] for _ in total_sl]
            self.fakesv=[[np.NaN,np.NaN] for _ in total_sl]

            self.istrategy = Buy_Strategies(self.lc)

    def Is_V3EvalCC(self, Cln_DBTP_Reader,calledby):
        return self.lc.system_type=="LHPP2V3" and calledby=="Eval" and Cln_DBTP_Reader=="DBTP_Eval_CC_Reader"

    def Is_ProcessIdx_CCProcessIdx(self, ProcessIdx):
        return any([ProcessIdx in l_CC_ProcessIdx for l_CC_ProcessIdx in  self.ll_CC_ProcessIdx])

    def get_Group_location_idx(self, ProcessIdx):
        return [ProcessIdx in l_CC_ProcessIdx for l_CC_ProcessIdx in self.ll_CC_ProcessIdx].index(True)


    def Stop_Record_on_ET(self, ET):
        self.l_TotalInvest = [self.lc.l_CC_group_invest_total_money[GroupIdx] for GroupIdx in self.l_CC_GroupIdx]
        if hasattr(self, "l_df"):
            for location_group_idx, group_idx in enumerate(self.l_CC_GroupIdx):
                self.l_df[location_group_idx].to_csv(self.get_Record_fnwp(ET,group_idx), index=False, float_format='%.2f')
                self.l_df[location_group_idx].drop(self.l_df[location_group_idx].index, inplace=True) #delete all row
                self.l_df[location_group_idx].reset_index(inplace=True, drop=True)  #clean the empty index after drop
        if hasattr(self, "l_dfADlog"):
            for location_group_idx, group_idx in enumerate(self.l_CC_GroupIdx):
                self.l_dfADlog[location_group_idx].to_csv(self.get_action_decision_log_fnwp(ET,group_idx), index=False, float_format='%.2f')
                self.l_dfADlog[location_group_idx].drop(self.l_dfADlog[location_group_idx].index, inplace=True) #delete all row
                self.l_dfADlog[location_group_idx].reset_index(inplace=True, drop=True)  # clean the empty index after drop
        if hasattr(self, "l_dfMoney"):
            for location_group_idx, group_idx in enumerate(self.l_CC_GroupIdx):
                self.l_dfMoney[location_group_idx].to_csv(self.get_money_in_hand(ET,group_idx), index=False, float_format='%.2f')
                self.l_dfMoney[location_group_idx].drop(self.l_dfMoney[location_group_idx].index, inplace=True) #delete all row
                self.l_dfMoney[location_group_idx].reset_index(inplace=True, drop=True)  # clean the empty index after drop

    def CC_handler(self,process_idx, stacted_state,result,LL_GPU2Eval):
        location_group_idx=self.get_Group_location_idx(process_idx)
        self.lll_CC_OutBuffer[location_group_idx][process_idx - self.ll_CC_ProcessIdx[location_group_idx][0]] = [stacted_state, result]
        if any([len(item) == 0 for item in self.lll_CC_OutBuffer[location_group_idx]]):
            return False,np.NaN
        l_a_OB, l_a_OS, l_holding,L_Eval_Profit_low_flag,l_sell_return,l_buy_invest,l_DateI, l_log = [[] for _ in range(8)]
        l_holding_value=[]
        l_Tinpai_huaizhang=[]
        for CC_OutBuffer_item in  self.lll_CC_OutBuffer[location_group_idx]:
            [_, _, av],[l_a_OB_round, l_a_OS_round]=CC_OutBuffer_item
            l_a_OB.extend(l_a_OB_round)
            l_a_OS.extend(l_a_OS_round)
            l_holding.extend([self.i_cav.Is_Holding_Item(av_item) for av_item in  av ])
            l_sell_return.extend([self.i_cav.get_inform_item(av_item,"Sell_Return") for av_item in  av ])
            l_buy_invest.extend([self.i_cav.get_inform_item(av_item, "Buy_Invest") for av_item in av])
            L_Eval_Profit_low_flag.extend([True if self.i_cav.get_inform_item(av_item, "Eval_Profit")
                    <=self.lc.l_CC_group_low_profit_threadhold[self.l_CC_GroupIdx[location_group_idx]] else False for av_item in av])

            l_DateI.extend([self.i_cav.get_inform_item(av_item, "DateI") for av_item in av])
            l_log.extend([self.i_cav.get_inform_in_all(av_item) for av_item in av])
            l_holding_value.extend([self.i_cav.get_inform_item(av_item,"Holding_Gu")*
                                    self.i_cav.get_inform_item(av_item,"Holding_NPrice") for av_item in av])
            l_Tinpai_huaizhang.extend([self.i_cav.get_inform_item(av_item, "Tinpai_huaizhang") for av_item in av])
        assert len(set(l_DateI))==1,l_DateI
        if self.l_CC_last_eval_date[location_group_idx]>l_DateI[0]:
            print("CC group {0} End Evaluation at {1} and Start Evaluation at {2}   ".
                  format(location_group_idx, self.l_CC_last_eval_date[location_group_idx],l_DateI[0]))
        self.l_CC_last_eval_date[location_group_idx]=l_DateI[0]

        sell_money_on_the_way=sum(l_sell_return)
        buy_money_used=sum(l_buy_invest)
        Tinpai_huaizhang=sum(l_Tinpai_huaizhang)
        self.l_TotalInvest[location_group_idx] =self.l_TotalInvest[location_group_idx] -buy_money_used+sell_money_on_the_way
        eval_holding_value=sum(l_holding_value)

        l_moneylog=[[l_DateI[0], self.l_TotalInvest[location_group_idx], eval_holding_value, sell_money_on_the_way,
          self.l_TotalInvest[location_group_idx] + eval_holding_value, Tinpai_huaizhang]]

        num_stock_could_invest= int(self.l_TotalInvest[location_group_idx]//self.lc.l_CC_min_invest_per_round[self.l_CC_GroupIdx[location_group_idx]])

        l_a,l_ADlog = getattr(self.istrategy, self.lc.l_CC_group_strategy_fun[self.l_CC_GroupIdx[location_group_idx]])\
            (l_DateI[0],num_stock_could_invest,l_a_OB,l_a_OS,l_holding,L_Eval_Profit_low_flag)

        self.l_df[location_group_idx] = self.l_df[location_group_idx].append(
            pd.DataFrame(l_log, columns=self.log_titles), ignore_index=True)

        self.l_dfMoney[location_group_idx] = self.l_dfMoney[location_group_idx].\
            append(pd.DataFrame(l_moneylog,columns=self.dfMoney_titles),ignore_index=True)

        self.l_dfADlog[location_group_idx] = self.l_dfADlog[location_group_idx].\
            append(pd.DataFrame([l_ADlog],columns=self.ADlog_titles), ignore_index=True)


        for process_idx in self.ll_CC_ProcessIdx[location_group_idx]:
            idx=process_idx-self.ll_CC_ProcessIdx[location_group_idx][0]
            result=[l_a[self.StartEndP[idx]:self.StartEndP[idx+1]],
                    self.fakeap[self.StartEndP[idx]:self.StartEndP[idx+1]],
                    self.fakesv[self.StartEndP[idx]:self.StartEndP[idx+1]]]  #l_ap and l_sv not used in Eval CC as no are log needed
            LL_GPU2Eval[process_idx].append(result)
        for CC_OutBuffer_item in self.lll_CC_OutBuffer[location_group_idx]:
            del CC_OutBuffer_item[:]
        return True, list(set(l_DateI))[0]
    def Get_V3EvalCC_ProcessIdx_Range(self):
        L_CC_ProcessGroup=[process_group_idx for process_group_idx in self.lc.l_eval_num_process_group
                    if self.lc.l_CLN_env_get_data_eval[process_group_idx]=="DBTP_Eval_CC_Reader"]
        if len(L_CC_ProcessGroup)==0:
            return [],[]
        else:
            return L_CC_ProcessGroup, [list(range(self.lc.l_eval_num_process_group.index(CC_ProcessGroup)*self.lc.eval_num_process_each_group,
                   (self.lc.l_eval_num_process_group.index(CC_ProcessGroup)+1)*self.lc.eval_num_process_each_group)) for CC_ProcessGroup in L_CC_ProcessGroup]

    def Extract_V3EvalCC_MultiplexAction(self,MultiplexAction):
        action=MultiplexAction%10
        PSS_action=MultiplexAction//10%10
        return action, PSS_action

    def Fabricate_V3EvalCC_MultiplexAction(self,action,PSS_action):
        return PSS_action*10+action

