import os,json, shutil,sys
from collections import OrderedDict
import config as sc
import pandas as pd
import numpy as np

import random
from DBI_Base import DBI_init,hfq_toolbox,StockList
import DBTP_Reader
from DBR_Reader import RawData

from State import AV_Handler_AV1
from Buy_Strategies import Buy_Strategies
#Strategy_config
{
    "strategy_name": "ZZZ",
    "strategy_fun": "Buy_Strategy_multi_time_Direct_sell",
    "RL_system_name": "SSS",
    "RL_Model_ET": "250",
    "GPU_mem":2600
}

#Experiement_config
{
    "total_invest": 250000,
    "min_invest": 50000,
    "StartI": 20201201,
    "EndI": 20210228
}
class ATFH:
    def __init__(self,Strategy_dir, experiment_name):
        self.AT_account_dir = os.path.join(Strategy_dir, experiment_name)
        if not os.path.exists(self.AT_account_dir): os.mkdir(self.AT_account_dir)
    def get_SL_fnwp(self):
        return os.path.join(self.AT_account_dir,"SL_in_order.csv")
    def get_account_fnwp(self):
        return os.path.join(self.AT_account_dir,"AT_account.csv")
    def get_account_backup_fnwp(self,DateI):
        desdir=os.path.join(self.AT_account_dir,str(DateI))
        if not os.path.exists(desdir): os.mkdir(desdir)
        return os.path.join(desdir,"AT_account_afterday_backup.csv")
    def get_aresult_fnwp(self,DateI):
        desdir=os.path.join(self.AT_account_dir,str(DateI))
        if not os.path.exists(desdir): os.mkdir(desdir)
        return os.path.join(desdir,"AT_StepResult.csv")
    def get_a2eDone_fnwp(self,DateI):
        desdir=os.path.join(self.AT_account_dir,str(DateI))
        if not os.path.exists(desdir): os.mkdir(desdir)
        return os.path.join(desdir,"AT_Action2exeDone.csv")
    def get_a2e_fnwp(self,DateI):
        desdir=os.path.join(self.AT_account_dir,str(DateI))
        if not os.path.exists(desdir): os.mkdir(desdir)
        return os.path.join(desdir,"AT_Action2exe.csv")
    def get_account_detail_fnwp(self):
        return os.path.join(self.AT_account_dir,"AT_AccountDetail.csv")
    def get_account_detail_backup_fnwp(self,DateI):
        desdir=os.path.join(self.AT_account_dir,str(DateI))
        if not os.path.exists(desdir): os.mkdir(desdir)
        return os.path.join(desdir, "AT_AccountDetail_backup.csv")
    def get_report_fnwp(self, DateI):
        desdir=os.path.join(self.AT_account_dir,str(DateI))
        if not os.path.exists(desdir): os.mkdir(desdir)
        return os.path.join(desdir, "Report.txt")

class Experiment_Config:
    AT_base_dir = "/home/rdchujf/n_workspace/AT"
    total_invest=float("NaN")
    min_invest=float("NaN")
    StartI=float("NaN")
    EndI=float("NaN")
    def __init__(self, portfolio_name, strategy_name, Experiement_name):
        self.Experiment_dir=os.path.join(self.AT_base_dir,portfolio_name, strategy_name,Experiement_name)
        self.Experiement_name = Experiement_name
        config_fnwp=os.path.join(self.Experiment_dir,"config.json")
        param = json.load(open(config_fnwp, "r"), object_pairs_hook=OrderedDict)
        for item in list(param.keys()):
            if not item.startswith("======="):
                self.__dict__[item] = param[item]

class Strategy_Config:
    AT_base_dir="/home/rdchujf/n_workspace/AT"
    #set default value for param
    strategy_fun=""
    RL_system_name=""
    RL_Model_ET=float("NaN")
    GPU_mem=float("NaN")
    def __init__(self,portfolio_name, strategy_name):
        #load strategy config
        self.Strategy_dir=os.path.join(self.AT_base_dir,portfolio_name, strategy_name)
        self.strategy_name = strategy_name
        config_fnwp=os.path.join(self.Strategy_dir,"config.json")
        param = json.load(open(config_fnwp, "r"), object_pairs_hook=OrderedDict)
        for item in list(param.keys()):
            if not item.startswith("======="):
                self.__dict__[item] = param[item]

        self.AT_log_dir=os.path.join(self.Strategy_dir, "Log")
        if not os.path.exists(self.AT_log_dir): os.mkdir(self.AT_log_dir)
        self.AT_model_dir=os.path.join(self.Strategy_dir,"model")
        if not os.path.exists(self.AT_model_dir): os.mkdir(self.AT_model_dir)
        self.RL_config_dir=os.path.join(self.Strategy_dir,"RL_config")
        if not os.path.exists(self.RL_config_dir): os.mkdir(self.RL_config_dir)

        #load selected RL config
        RL_config_fnwp=os.path.join(self.RL_config_dir, "config.json")
        if not os.path.exists(RL_config_fnwp):
            src_RL_config_fnwp=os.path.join(sc.base_dir_RL_system, self.RL_system_name, "config.json")
            assert os.path.exists(src_RL_config_fnwp),\
                "RL {0} config local copy does not exsit and original copy can not found".format(self.RL_system_name)
            shutil.copy(src_RL_config_fnwp,self.RL_config_dir)
        self.rlc = sc.gconfig()
        self.rlc.read_from_json(RL_config_fnwp,system_name=self.RL_system_name)

        weight_fns = [fn for fn in os.listdir(self.AT_model_dir) if fn.endswith(".h5") and "T{0}".
            format(self.RL_Model_ET) in fn]
        if len(weight_fns) != 1:
            src_weight_fns = [fn for fn in os.listdir(self.rlc.brain_model_dir) if fn.endswith(".h5") and "T{0}".
                format(self.RL_Model_ET) in fn]
            assert len(src_weight_fns) == 1, \
                "Model weight Local copy does not exsit and original copy can not found ET={0}".format(self.RL_Model_ET)
            shutil.copy(os.path.join(self.rlc.brain_model_dir, src_weight_fns[0]), self.AT_model_dir)
            weight_fns = [fn for fn in os.listdir(self.AT_model_dir) if fn.endswith(".h5") and "T{0}".
                format(self.RL_Model_ET) in fn]
        assert len(weight_fns)==1,"{0} has more than one weight file {1}".format(self.AT_model_dir,weight_fns)
        self.weight_fnwp=os.path.join(self.AT_model_dir,weight_fns[0])


class Strategy_agent_base(Strategy_Config,Experiment_Config,DBI_init):
    def __init__(self, portfolio_name, strategy_name,experiment_name):
        Strategy_Config.__init__(self, portfolio_name, strategy_name)
        Experiment_Config.__init__(self, portfolio_name, strategy_name,experiment_name)
        DBI_init.__init__(self)
        self.i_hfq_tb =hfq_toolbox()
        self.iFH = ATFH(self.Strategy_dir, experiment_name)
        self.set_df_params(self.rlc)

    def set_df_params(self, rlc):
        ##set dataframe params
        # Account_Holding_Items=["TransIDI","Holding_Gu","Holding_Invest","Holding_HRatio","Holding_NPrice","Buy_Times"]
        self.Account_Holding_Items_titles= rlc.account_inform_holding_titles+["HoldingStartDateI"]
        self.Account_Holding_Items_default = [0, 0, 0.0, 1.0, 0.0, 0,00000000]
        self.Account_Holding_Items_types={**rlc.account_inform_holding_types,**{"HoldingStartDateI":int}}

        self.Account_Inform_Items_title=["AfterClosing_HFQRatio","AfterClosing_NPrice"]
        self.Account_Inform_Items_default =[1.0,0.0]
        self.Account_Inform_Items_types={"AfterClosing_HFQRatio":float,"AfterClosing_NPrice":float}
        # Account_Step_Items=["Buy_Invest","Buy_NPrice","Sell_Return","Sell_NPrice","Sell_Earn","Tinpai_huaizhang", "Action"]
        self.Account_Step_Items_titles = rlc.account_inform_step_titles + ["Action"]
        self.Account_Step_Items_titles.remove("Buy_NPrice")
        self.Account_Step_Items_titles.remove("Sell_NPrice")
        self.Account_Step_Items_default = [0.0, 0.0, 0.0, 0.0, -1]
        self.Account_Step_Items_types={**rlc.account_inform_step_types,**{"Action":str}}
        del self.Account_Step_Items_types["Buy_NPrice"]
        del self.Account_Step_Items_types["Sell_NPrice"]
        self.account_Titles = ["Stock"] + self.Account_Holding_Items_titles +self.Account_Inform_Items_title+ self.Account_Step_Items_titles
        self.account_types={**{"Stock":str},**self.Account_Holding_Items_types,**self.Account_Inform_Items_types,**self.Account_Step_Items_types}
        self.account_default = [""] + self.Account_Holding_Items_default + self.Account_Inform_Items_default + self.Account_Step_Items_default

        self.aresult_Titles = ["Stock", "Action", "Action_Result", "Buy_Gu", "Buy_Invest","Sell_Return"]
        self.aresult_types = {"Stock": str, "Action":str, "Action_Result":str, "Buy_Gu":int,"Buy_Invest":float,"Sell_Return":float}

        self.a2e_Titles = ["Stock", "Action", "Gu"]
        self.a2e_types={"Stock":str, "Action":str, "Gu":int}

        self.AccountDetail_titles= ["DateI","Cash_after_closing", "MarketValue_after_closing"]
        self.AccountDetail_types = {"DateI":int,"Cash_after_closing":float, "MarketValue_after_closing":float}

    def init_stock_list(self):
        fnwp_sl = self.iFH.get_SL_fnwp()
        iSL = StockList(self.rlc.SLName)
        flag,sl = iSL.get_sub_sl("Train",0)
        assert flag
        pd.DataFrame(sl, columns=["Stock"]).to_csv(fnwp_sl, index=False)
        return sl

    def load_stock_list(self):
        fnwp_sl = self.iFH.get_SL_fnwp()
        assert os.path.exists(fnwp_sl)
        return pd.read_csv(fnwp_sl)["Stock"].tolist()

    def Init_df_account(self, stock_list):
        fnwp_account = self.iFH.get_account_fnwp()
        df_account=pd.DataFrame(columns=self.account_Titles)
        for stock in stock_list:
            row=list(self.account_default)
            row[0]=stock
            df_account.loc[len(df_account)]=row
        df_account = df_account.astype(dtype=self.account_types)
        df_account.set_index(["Stock"],drop=True,inplace=True,verify_integrity=True)
        df_account.to_csv(fnwp_account)
        return df_account

    def load_df_account(self):
        fnwp_account=self.iFH.get_account_fnwp()
        assert os.path.exists(fnwp_account),fnwp_account
        df_account = pd.read_csv(fnwp_account)
        df_account = df_account.astype(dtype=self.account_types)
        df_account.set_index(["Stock"],drop=True,inplace=True,verify_integrity=True)
        return df_account

    def save_df_account(self, df_account, DateI):
        fnwp_account = self.iFH.get_account_fnwp()
        fnwp_account_backup = self.iFH.get_account_backup_fnwp(DateI)
        df_account.to_csv(fnwp_account)   #here need to save index, it is stock
        df_account.to_csv(fnwp_account_backup) #here need to save index, it is stock

    def save_next_day_action(self, DateI, l_a,stocklist,df_account):
        a2e_fnwp=self.iFH.get_a2e_fnwp(DateI)
        df_e2a=pd.DataFrame(columns=self.a2e_Titles)
        df_e2a=df_e2a.astype(self.a2e_types)
        df_e2a.set_index(["Stock"], drop=True, inplace=True,verify_integrity=True)
        for action, stock in zip(l_a,stocklist):
            if action ==0:
                df_e2a.at[stock, "Action"] = "Buy"
                try:
                    df_e2a.at[stock, "Gu"] = (self.min_invest/df_account.loc[stock,"AfterClosing_NPrice"])//100*100
                except Exception as e:
                    print (stock,df_account.loc[stock,"AfterClosing_NPrice"])
                    df_account.to_csv("/home/rdchujf/a.csv")
                    assert False
            elif action ==2:
                df_e2a.at[stock, "Action"] = "Sell"
                df_e2a.at[stock, "Gu"]= self.i_hfq_tb.get_update_volume_on_hfq_ratio_change\
                    (df_account.loc[stock,"Holding_HRatio"], df_account.loc[stock,"AfterClosing_HFQRatio"],
                     df_account.loc[stock]["Holding_Gu"])
            else:
                continue
        df_e2a.to_csv(a2e_fnwp)
        return df_e2a

    def load_df_a2eDone(self, DateI):
        fnwp_action2exeDone = self.iFH.get_a2eDone_fnwp(DateI)
        assert os.path.exists(fnwp_action2exeDone), "{0} does not exists".format(fnwp_action2exeDone)
        df_a2eDone=pd.read_csv(fnwp_action2exeDone)
        df_a2eDone=df_a2eDone.astype(self.a2e_types)
        df_a2eDone.set_index(["Stock"], drop=True, inplace=True,verify_integrity=True)
        return df_a2eDone

    def load_df_aresult(self, DateI):
        fnwp_action_result=self.iFH.get_aresult_fnwp(DateI)
        assert os.path.exists(fnwp_action_result), "{0} does not exists".format(fnwp_action_result)
        df_aresult = pd.read_csv(fnwp_action_result)
        df_aresult= df_aresult.astype(self.aresult_types)
        df_aresult= df_aresult.groupby(["Stock"]).agg(
            Action=pd.NamedAgg(column='Action', aggfunc=lambda x: [xi for xi in x][0]),
            Action_Result=pd.NamedAgg(column='Action_Result', aggfunc=lambda x: [xi for xi in x][0]),
            Buy_Gu=pd.NamedAgg(column='Buy_Gu', aggfunc='sum'),
            Buy_Invest=pd.NamedAgg(column='Buy_Invest', aggfunc='sum'),
            Sell_Return=pd.NamedAgg(column='Sell_Return', aggfunc='sum')
        )
        return df_aresult

    def Check_all_actionDone_has_result(self,df_a2eDone,df_aresult):
        aresult_SA= set(df_aresult.apply(lambda x: x.name + x["Action"], axis=1).tolist())
        a2e_SA = set(df_a2eDone.apply(lambda x: x.name + x["Action"], axis=1).tolist())
        assert aresult_SA==a2e_SA,"action result and action 2 exe does not match"

    def Init_df_account_detail(self):
        df_account_detail=pd.DataFrame(columns=self.AccountDetail_titles)
        df_account_detail = df_account_detail.astype(dtype=self.AccountDetail_types)
        df_account_detail.set_index(["DateI"], drop=True, inplace=True,verify_integrity=True)
        return df_account_detail

    def load_account_detail(self,DateI):
        fnwp_account_detail=self.iFH.get_account_detail_fnwp()
        assert os.path.exists(fnwp_account_detail)
        df_account_detail=pd.read_csv(fnwp_account_detail)
        df_account_detail = df_account_detail.astype(dtype=self.AccountDetail_types)
        df_account_detail.set_index(["DateI"], drop=True, inplace=True,verify_integrity=True)
        account_day_inform = df_account_detail.loc[DateI]
        assert not account_day_inform.empty, "the last record of {0} is not for {1} {2}".\
            format(fnwp_account_detail, DateI,account_day_inform)
        Cash_afterclosing,MarketValue_afterclosing = \
            account_day_inform["Cash_after_closing"],account_day_inform["MarketValue_after_closing"]
        print ("Loaded account detail before {0} Cash_afterclosing {1:.2f} Market value {2:.2f} total {3:.2f}".format
               (account_day_inform.name,Cash_afterclosing,MarketValue_afterclosing,Cash_afterclosing+MarketValue_afterclosing))
        return df_account_detail,Cash_afterclosing,MarketValue_afterclosing

    def update_save_account_detail(self,df_account_detail, DateI,Cash_afterclosing,MarketValue_afterclosing):
        df_account_detail.loc[DateI]=[Cash_afterclosing,MarketValue_afterclosing]
        fnwp_account_detail=self.iFH.get_account_detail_fnwp()
        fnwp_account_detail_backup = self.iFH.get_account_detail_backup_fnwp(DateI)
        df_account_detail.to_csv(fnwp_account_detail) #here need to save index, it is DateI
        df_account_detail.to_csv(fnwp_account_detail_backup) #here need to save index, it is DateI

    def get_AfterClosing_Nprice_HFQRatio(self,Stock,DateI):
        #Check Whether addon DBI HFQ_index update finalized, otherwise return False
        #assert DateI > self.iDBI_init.Raw_Normal_Lumpsum_EndDayI
        if DateI > self.Raw_Normal_Lumpsum_EndDayI:
            if not os.path.exists(self.get_DBI_Update_Log_HFQ_Index_fnwp(DateI)):
                return False,[],"{0} DBI addon update for HFQ and index have not done"
        else:
            if not os.path.exists(self.get_DBI_Lumpsum_Log_HFQ_Index_fnwp()):
                return False,[],"{0} DBI lumpsum update for HFQ and index have not done"

        fnwp=self.get_DBI_hfq_fnwp(Stock)
        if not os.path.exists(fnwp):
            return False, [], "DBI HFQ File Not Found**** {0}".format(fnwp)
        flag,df, mess=self.get_hfq_df(fnwp)
        if not flag:
            return False, [], mess
        a=df[df["date"]<=str(DateI)]

        try:
            Closing_Nprice=a.iloc[-1]["close_price"]/a.iloc[-1]["coefficient_fq"]
            HFQRatio = a.iloc[-1]["coefficient_fq"]
            assert Closing_Nprice!=0,"{0} {1} {2}".format(Stock,DateI,a)
            return True, [Closing_Nprice, HFQRatio], "Success"
        except:
            mess="{0} {1} does not exist in {2}  \n {3}".format(Stock, DateI,fnwp, a)
            print(mess)
            return False, [], mess

class Strategy_agent_Report:
    def prepare_report(self, DateI, logs, df_e2a, sl, report_fnwp):
        Cash_afterclosing, MarketValue_afterclosing, mumber_of_stock_could_buy, \
        l_log_bought, l_log_Earnsold, l_log_balancesold, l_log_Losssold, \
        l_log_fail_action, l_log_holding_with_no_action, \
        l_ADlog = logs
        #fnwp = self.iFH.get_report_fnwp(DateI)
        with open(report_fnwp, "w") as f:
            f.write("Date: {0} Cash After Closing: {1} Market Value: {2}\n Num of Stock could bought tomorrow: {3}\n".
                    format(DateI, Cash_afterclosing, MarketValue_afterclosing, mumber_of_stock_could_buy))
            f.write("Today Bought {0} stock:\n".format(len(l_log_bought)))
            f.write("    {0}\n".format(",".join(l_log_bought)))
            f.write("Today Sold with Earn {0} stock:\n".format(len(l_log_Earnsold)))
            f.write("    {0}\n".format(",".join(l_log_Earnsold)))
            f.write("Today Sold with Loss {0} stock:\n".format(len(l_log_Losssold)))
            f.write("    {0}\n".format(",".join(l_log_Losssold)))
            f.write("Today Sold with Balance {0} stock:\n".format(len(l_log_balancesold)))
            f.write("    {0}\n".format(",".join(l_log_balancesold)))
            f.write("Today {0} Action with Fail :\n".format(len(l_log_fail_action)))
            for log_fail in l_log_fail_action:
                f.write("    {0} {1}\n".format(log_fail[0], log_fail[1]))
            f.write("\n")  # For look better
            f.write("Today Error holding without sell action {0} :\n".format(len(l_log_holding_with_no_action)))
            for holding_with_no_action in l_log_holding_with_no_action:
                f.write("    {0} {1}\n".format(holding_with_no_action[0], holding_with_no_action[1]))
            f.write("\n")  # For look better
            f.write("Tommorow to Buy {0}\n".format(len(df_e2a[df_e2a["Action"] == "Buy"])))
            f.write("    {0}\n".format(",".join(df_e2a[df_e2a["Action"] == "Buy"].index.tolist())))
            f.write("Tommorow to Sell {0}\n".format(len(df_e2a[df_e2a["Action"] == "Sell"])))
            f.write("    {0}\n".format(",".join(df_e2a[df_e2a["Action"] == "Sell"].index.tolist())))

            assert l_ADlog[0] == DateI
            l_not_buy_due_limit = l_ADlog[1].split("_")
            if l_not_buy_due_limit == ['']:
                f.write("Tommorow not_buy_due_limit {0}\n".format(0))
                f.write("\n")  # For look better
            else:
                f.write("Tommorow not_buy_due_limit {0}\n".format(len(l_not_buy_due_limit)))
                for sidx in l_not_buy_due_limit:
                    f.write("    {0}\n".format(sl[int(sidx)]))
            l_multibuy = l_ADlog[4].split("_")
            if l_multibuy == ['']:
                f.write("Tommorow multibuy(not sell due to multibuy) {0}\n".format(0))
                f.write("\n")  # For look better
            else:
                f.write("Tommorow multibuy(not sell due to multibuy) {0}\n".format(len(l_multibuy)))
                for sidx in l_multibuy:
                    f.write("    {0}\n".format(sl[int(sidx)]))
        print("{0} Report stored at {1}\n".format(DateI, report_fnwp))
