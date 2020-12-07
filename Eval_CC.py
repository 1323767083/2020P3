import os, random
import DBI_Base
import pandas as pd
import numpy as np
from State import *
class Eval_CC:
    #Max_TotalMoney=5000000
    #low_profit_threadhold = -0.02
    def __init__(self,lc):
        self.lc=lc
        self.iSL = DBI_Base.StockList(self.lc.SLName)
        self.CC_GroupIdx, self.l_CC_ProcessIdx=self.Get_V3EvalCC_ProcessIdx_Range()

        if len(self.l_CC_ProcessIdx)!=0:
            self.CC_OutBuffer=[[] for _ in self.l_CC_ProcessIdx]
            self.StartEndP =[]
            start_idx=0
            total_sl=[]
            for sl in [self.iSL.Get_Eval_SubProcess_SL(lc, self.CC_GroupIdx, process_idx) for process_idx in self.l_CC_ProcessIdx]:
                total_sl.extend(sl)
                self.StartEndP.append(start_idx)
                start_idx+=len(sl)
            else:
                self.StartEndP.append(start_idx)

            self.log_titles = self.lc.account_inform_titles + self.lc.simulator_inform_titles + self.lc.PSS_inform_titles
            self.df = pd.DataFrame(columns=self.log_titles)
            self.dfMoney_titles=["DateI","Money_in_hand","Eval_holding","Eval_Ttotal"]
            self.dfMoney=pd.DataFrame(columns=self.dfMoney_titles)
            self.CC_log_dir = os.path.join(self.lc.system_working_dir, "CC")
            if not os.path.exists(self.CC_log_dir): os.mkdir(self.CC_log_dir)

            self.TotalInvest=self.lc.Max_TotalMoney
            self.i_cav = globals()[lc.CLN_AV_Handler](lc)

            pd.DataFrame(total_sl, columns=["Stock"]).to_csv(self.get_SL_fnwp(), index=False)
            self.fakeap=[[np.NaN,np.NaN] for _ in total_sl]
            self.fakesv=[[np.NaN,np.NaN] for _ in total_sl]
    def Is_V3EvalCC(self, Cln_DBTP_Reader,calledby):
        return self.lc.system_type=="LHPP2V3" and calledby=="Eval" and Cln_DBTP_Reader=="DBTP_Eval_CC_Reader"

    def Is_ProcessIdx_CCProcessIdx(self, ProcessIdx):
        return ProcessIdx in self.l_CC_ProcessIdx

    def get_money_in_hand(self,ET):
        return os.path.join(self.CC_log_dir, "ET{0}_money_in_hand.csv".format(ET))

    def get_SL_fnwp(self):
        return os.path.join(self.CC_log_dir,"SL_in_order.csv")

    def get_Record_fnwp(self,ET):
        return os.path.join(self.CC_log_dir,"ET{0}.csv".format(ET))

    def get_action_decision_log_fnwp(self,ET):
        return os.path.join(self.CC_log_dir,"ET{0}_action_decision.csv".format(ET))

    def Stop_Record_on_ET(self, ET):
        self.TotalInvest = self.lc.Max_TotalMoney   #this is to make next round evaluation has money
        if hasattr(self, "df"):
            self.df.to_csv(self.get_Record_fnwp(ET), index=False, float_format='%.2f')
            self.df.drop(self.df.index, inplace=True) #delete all row
        if hasattr(self, "dfADlog"):
            self.dfADlog.to_csv(self.get_action_decision_log_fnwp(ET), index=False, float_format='%.2f')
            self.dfADlog.drop(self.dfADlog.index, inplace=True) #delete all row
        #self.dfMoney
        if hasattr(self, "dfMoney"):
            self.dfMoney.to_csv(self.get_money_in_hand(ET), index=False, float_format='%.2f')
            self.dfMoney.drop(self.dfMoney.index, inplace=True) #delete all row

    def Buy_Strategy_one_time(self,dateI,num_stock_could_invest,l_a_OB,l_a_OS,l_holding,L_Eval_Profit_low_flag):
        Sidxs_all       = set(list(range(len(l_a_OB))))
        Sidxs_OB_buy    = set([idx for idx, action in enumerate(l_a_OB) if action ==0])
        Sidxs_holding   = set([idx for idx, holding_flag in enumerate(l_holding) if holding_flag])
        Sidxs_OS_sell   = set([idx for idx, action in enumerate(l_a_OS) if action == 2])
        Sidxs_Eval_Low  = set([idx for idx, flag in enumerate(L_Eval_Profit_low_flag) if flag])

        Sidxs_buy = (Sidxs_OB_buy- Sidxs_Eval_Low) & (Sidxs_all-Sidxs_holding)
        Sidxs_sell = (Sidxs_OS_sell | Sidxs_Eval_Low)& Sidxs_holding
        Sidxs_not_buy_due_limit =set()  #empty set should create by set() not {}
        if len(Sidxs_buy)>num_stock_could_invest:
            lidxs_0=list(Sidxs_buy)
            random.shuffle(lidxs_0)
            Sidxs_buy=set(lidxs_0[:num_stock_could_invest])
            Sidxs_not_buy_due_limit=set(lidxs_0[num_stock_could_invest:])
        assert len(Sidxs_buy & Sidxs_sell)==0, "Sidxs_buy {0} & Sidxs_sell {1}".format(Sidxs_buy, Sidxs_sell)
        Sidxs_no_action_1 = Sidxs_all - Sidxs_holding -Sidxs_buy
        Sidxs_no_action_3 = Sidxs_holding - Sidxs_sell
        assert len(Sidxs_buy)+ len(Sidxs_no_action_1)+ len(Sidxs_sell)+len(Sidxs_no_action_3)==len(Sidxs_all)
        l_a = [0 for _ in range(len(l_a_OB))]
        for action, s_idx in enumerate([Sidxs_buy,Sidxs_no_action_1, Sidxs_sell, Sidxs_no_action_3]):
            for idx in list(s_idx):
                l_a[idx] = action

        l_not_buy_due_low_profit = list(Sidxs_Eval_Low & Sidxs_buy)
        l_sell_due_low_profit=list(Sidxs_Eval_Low & Sidxs_sell)
        l_not_buy_due_limit=list(Sidxs_not_buy_due_limit)
        l_not_buy_due_low_profit.sort()
        l_sell_due_low_profit.sort()
        l_not_buy_due_limit.sort()

        if not hasattr(self,"dfADlog"):
            self.ADlog_titles = ["DateI", "not_buy_due_limit", "not_buy_due_low_profit", "sell_due_low_profit"]
            self.dfADlog = pd.DataFrame(columns=self.ADlog_titles)

        l_ADlog=[dateI,
                "_".join(map(str, l_not_buy_due_limit)),
                "_".join(map(str, l_not_buy_due_low_profit)),
                "_".join(map(str, l_sell_due_low_profit))]
        self.dfADlog = self.dfADlog.append(pd.DataFrame([l_ADlog],columns=self.ADlog_titles), ignore_index=True)

        return l_a

    def Buy_Strategy_multi_time(self,dateI,num_stock_could_invest,l_a_OB,l_a_OS,l_holding,L_Eval_Profit_low_flag):
        Sidxs_all       = set(list(range(len(l_a_OB))))
        Sidxs_OB_buy    = set([idx for idx, action in enumerate(l_a_OB) if action ==0])
        Sidxs_holding   = set([idx for idx, holding_flag in enumerate(l_holding) if holding_flag])
        Sidxs_OS_sell   = set([idx for idx, action in enumerate(l_a_OS) if action == 2])
        Sidxs_Eval_Low  = set([idx for idx, flag in enumerate(L_Eval_Profit_low_flag) if flag])

        Sidxs_buy = (Sidxs_OB_buy- Sidxs_Eval_Low)
        Sidxs_not_buy_due_limit =set()
        if len(Sidxs_buy)>num_stock_could_invest:
            lidxs_0=list(Sidxs_buy)
            random.shuffle(lidxs_0)
            Sidxs_buy=set(lidxs_0[:num_stock_could_invest])
            Sidxs_not_buy_due_limit=set(lidxs_0[num_stock_could_invest:])

        #Sidxs_sell = ((Sidxs_OS_sell | Sidxs_Eval_Low)-Sidxs_buy)& Sidxs_holding
        Sidxs_sell = (Sidxs_OS_sell | Sidxs_Eval_Low) & Sidxs_holding - Sidxs_buy

        assert len(Sidxs_buy & Sidxs_sell)==0, "Sidxs_buy {0} & Sidxs_sell {1}".format(Sidxs_buy, Sidxs_sell)
        Sidxs_no_action_1 = Sidxs_all - Sidxs_holding -Sidxs_buy
        Sidxs_no_action_3 = Sidxs_holding - Sidxs_sell-Sidxs_buy # there is part of holding in buy action due to multi buy
        assert len(Sidxs_buy)+ len(Sidxs_no_action_1)+ len(Sidxs_sell)+len(Sidxs_no_action_3)==len(Sidxs_all)

        l_a = [0 for _ in range(len(l_a_OB))]
        for action, s_idx in enumerate([Sidxs_buy-Sidxs_holding,Sidxs_no_action_1, Sidxs_sell, Sidxs_no_action_3]):
            for idx in list(s_idx):
                l_a[idx] = self.Fabricate_V3EvalCC_MultiplexAction(action=action,PSS_action=0)
        for idx in list(Sidxs_buy &Sidxs_holding):
            l_a[idx] = self.Fabricate_V3EvalCC_MultiplexAction(action=0,PSS_action=1)

        l_not_buy_due_low_profit = list(Sidxs_Eval_Low & Sidxs_buy)
        l_sell_due_low_profit=list(Sidxs_Eval_Low & Sidxs_sell)
        l_not_buy_due_limit=list(Sidxs_not_buy_due_limit)
        l_multibuy= list(Sidxs_buy &Sidxs_holding)
        l_not_buy_due_low_profit.sort()
        l_sell_due_low_profit.sort()
        l_not_buy_due_limit.sort()
        l_multibuy.sort()

        if not hasattr(self,"dfADlog"):
            self.ADlog_titles = ["DateI", "not_buy_due_limit", "not_buy_due_low_profit", "sell_due_low_profit","multibuy"]
            self.dfADlog = pd.DataFrame(columns=self.ADlog_titles)

        l_ADlog=[dateI,
                "_".join(map(str, l_not_buy_due_limit)),
                "_".join(map(str, l_not_buy_due_low_profit)),
                "_".join(map(str, l_sell_due_low_profit)),
                "_".join(map(str, l_multibuy)) ]
        self.dfADlog = self.dfADlog.append(pd.DataFrame([l_ADlog],columns=self.ADlog_titles), ignore_index=True)

        return l_a

    def handler(self,process_idx, stacted_state,result,LL_GPU2Eval):
        self.CC_OutBuffer[process_idx - self.l_CC_ProcessIdx[0]] = [stacted_state, result]
        if any([len(item) == 0 for item in self.CC_OutBuffer]):
            return
        l_a_OB, l_a_OS, l_holding,L_Eval_Profit_low_flag,l_sell_return,l_buy_invest,l_DateI, l_log = [[] for _ in range(8)]
        l_holding_value=[]
        for CC_OutBuffer_item in  self.CC_OutBuffer:
            [_, _, av],[l_a_OB_round, l_a_OS_round]=CC_OutBuffer_item
            l_a_OB.extend(l_a_OB_round)
            l_a_OS.extend(l_a_OS_round)
            l_holding.extend([self.i_cav.Is_Holding_Item(av_item) for av_item in  av ])
            l_sell_return.extend([self.i_cav.get_inform_item(av_item,"Sell_Return") for av_item in  av ])
            l_buy_invest.extend([self.i_cav.get_inform_item(av_item, "Buy_Invest") for av_item in av])
            L_Eval_Profit_low_flag.extend([True if self.i_cav.get_inform_item(av_item, "Eval_Profit")
                                                   <=self.lc.low_profit_threadhold else False for av_item in av])
            l_DateI.extend([self.i_cav.get_inform_item(av_item, "DateI") for av_item in av])
            l_log.extend([self.i_cav.get_inform_in_all(av_item) for av_item in av])
            l_holding_value.extend([self.i_cav.get_inform_item(av_item,"Holding_Gu")*
                                    self.i_cav.get_inform_item(av_item,"Holding_NPrice") for av_item in av])
        assert len(set(l_DateI))==1,l_DateI
        print(l_DateI[0])
        self.df=self.df.append(pd.DataFrame(l_log,columns=self.log_titles), ignore_index=True)

        self.TotalInvest=self.TotalInvest-sum(l_buy_invest)+ sum(l_sell_return)
        eval_holding_value=sum(l_holding_value)
        self.dfMoney =self.dfMoney.append(pd.DataFrame([[l_DateI[0],self.TotalInvest, eval_holding_value,
                                        self.TotalInvest+eval_holding_value]],
                                        columns=self.dfMoney_titles),ignore_index=True)

        num_stock_could_invest= int(self.TotalInvest//self.lc.env_min_invest_per_round)

        l_a = getattr(self, self.lc.CC_strategy_fun)(l_DateI[0],num_stock_could_invest,l_a_OB,l_a_OS,l_holding,
                                                     L_Eval_Profit_low_flag)

        for process_idx in self.l_CC_ProcessIdx:
            idx=process_idx-self.l_CC_ProcessIdx[0]
            result=[l_a[self.StartEndP[idx]:self.StartEndP[idx+1]],self.fakeap,self.fakesv]  #l_ap and l_sv not used in Eval CC as no are log needed
            LL_GPU2Eval[process_idx].append(result)
        for CC_OutBuffer_item in self.CC_OutBuffer:
            del CC_OutBuffer_item[:]

    def Get_V3EvalCC_ProcessIdx_Range(self):
        L_CC_ProcessGroup=[idx for idx,process_group_idx in enumerate(self.lc.l_eval_num_process_group)
                    if self.lc.l_CLN_env_get_data_eval[process_group_idx]=="DBTP_Eval_CC_Reader"]
        if len(L_CC_ProcessGroup)==0:
            return np.NaN,[]
        elif len(L_CC_ProcessGroup)==1:
            return L_CC_ProcessGroup[0], list(range(L_CC_ProcessGroup[0]*self.lc.eval_num_process_per_group,
                                                    (L_CC_ProcessGroup[0]+1)*self.lc.eval_num_process_per_group))
        else:
            assert False, "DBTP_Eval_CC_Reader Only can be one progcess group DBTP_reader"

    def Extract_V3EvalCC_MultiplexAction(self,MultiplexAction):
        action=MultiplexAction%10
        PSS_action=MultiplexAction//10%10
        return action, PSS_action

    def Fabricate_V3EvalCC_MultiplexAction(self,action,PSS_action):
        return PSS_action*10+action

