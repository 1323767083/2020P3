import os,json, shutil,sys,random,setproctitle,time, pickle
from collections import OrderedDict
import pandas as pd
import numpy as np
from multiprocessing import Process,Event,Manager
from itertools import accumulate
import config as sc
from DBI_Base import DBI_init_with_TD,hfq_toolbox,StockList
import DBTP_Reader
from DBR_Reader import RawData
from State import AV_Handler_AV1
from Buy_Strategies import Buy_Strategies

AT_base_dir = "/home/rdchujf/n_workspace/AT"
''' strategy_config.csv
RL_system_name,RL_Model_ET,GPU_idx,GPU_mem,TPDB_Name,SL_Name
'''


'''Experiement_config
Model_idx,SL_Tag,SL_Idx,total_invest,min_invest,StartI,EndI,flag_Print_on_screen_or_file,strategy_fun
'''

class ATFH:
    def __init__(self,Strategy_dir, experiment_name):
        self.AT_account_dir = os.path.join(Strategy_dir, experiment_name)
        if not os.path.exists(self.AT_account_dir): os.mkdir(self.AT_account_dir)

    def _get_month_date_dir(self, base_dir, DateI):
        monthI=DateI//100
        dir=base_dir
        for subdir in [str(monthI), str(DateI)]:
            dir=os.path.join(dir, subdir)
            if not os.path.exists(dir): os.mkdir(dir)
        return dir
    def get_SL_fnwp(self,idx):
        return os.path.join(self.AT_account_dir,"SL{0}_in_order.csv".format(idx))
    def get_account_fnwp(self,idx):
        return os.path.join(self.AT_account_dir,"AT{0}_account.csv".format(idx))
    def get_account_backup_fnwp(self,DateI,idx):
        return os.path.join(self._get_month_date_dir(self.AT_account_dir,DateI),"AT{0}_account_afterday_backup.csv".format(idx))
    def get_aresult_fnwp(self,DateI,idx):
        return os.path.join(self._get_month_date_dir(self.AT_account_dir,DateI),"AT{0}_ActionResult.csv".format(idx))
    def get_a2eDone_fnwp(self,DateI,idx):
        return os.path.join(self._get_month_date_dir(self.AT_account_dir,DateI),"AT{0}_Action2exeDone.csv".format(idx))
    def get_a2e_fnwp(self,DateI,idx):
        return os.path.join(self._get_month_date_dir(self.AT_account_dir,DateI),"AT{0}_Action2exe.csv".format(idx))
    def get_account_detail_fnwp(self):
        return os.path.join(self.AT_account_dir,"AT_AccountDetail.csv")
    def get_account_detail_backup_fnwp(self,DateI):
        return os.path.join(self._get_month_date_dir(self.AT_account_dir,DateI), "AT_AccountDetail_backup.csv")
    def get_report_fnwp(self, DateI):
        return os.path.join(self._get_month_date_dir(self.AT_account_dir,DateI), "Report.txt")
    def get_machine_report_fnwp(self, DateI):
        return os.path.join(self._get_month_date_dir(self.AT_account_dir,DateI), "Report.pikle")

class Experiment_Config:
    Experiment_titles=["Model_idx","SL_Tag","SL_Idx","total_invest","min_invest","StartI","EndI","flag_Print_on_screen_or_file","strategy_fun"]
    def __init__(self, portfolio_name, strategy_name, experiment_name, DEcofig):
        self.Experiment_dir = os.path.join(AT_base_dir, portfolio_name, strategy_name, experiment_name)
        if not os.path.exists(self.Experiment_dir): os.mkdir(self.Experiment_dir)
        self.experiment_name = experiment_name
        config_fnwp = os.path.join(self.Experiment_dir, "experiment_config.csv")
        if len(DEcofig)==0 :
            self.dfec=pd.read_csv(config_fnwp)
        else:
            self.dfec=pd.DataFrame(columns=self.Experiment_titles)
            assert(DEcofig["EP_titles"]==self.Experiment_titles[:3])
            dfcontent=pd.DataFrame(DEcofig["EP_values"],columns=DEcofig["EP_titles"])
            for idx, row in dfcontent.iterrows():
                self.dfec.loc[len(self.dfec)]=[row["Model_idx"],row["SL_Tag"],row["SL_Idx"],
                                DEcofig["total_invest"],DEcofig["min_invest"],DEcofig["StartI"],DEcofig["EndI"],
                                               DEcofig["flag_Print_on_screen_or_file"],DEcofig["strategy_fun"]]
            self.dfec.to_csv(config_fnwp,index=False)
        #Model_idx, SL_Tag, SL_Idx, total_invest, min_invest, StartI, EndI, flag_Print_on_screen_or_file
        for title in ["Model_idx", "SL_Idx", "total_invest", "min_invest", "StartI", "EndI"]:
            self.dfec[title]=self.dfec[title].astype(int)


class Strategy_Config:
    def __init__(self,portfolio_name, strategy_name, l_GPUs=""):
        self.Strategy_dir=os.path.join(AT_base_dir,portfolio_name, strategy_name)
        config_fnwp=os.path.join(self.Strategy_dir,"strategy_config.csv")
        self.dfsc=pd.read_csv(config_fnwp)
        # RL_system_name,RL_Model_ET,GPU_idx,GPU_mem,TPDB_Name,SL_Name
        for title in ["RL_Model_ET", "GPU_idx", "GPU_mem"]:
            self.dfsc[title]=self.dfsc[title].astype(int)
        if len(l_GPUs)!=0:
            assert len(l_GPUs)==len(self.dfsc), "lengh suould match {0} {1}".format(l_GPUs,self.dfsc)
            for Model_idx in list(range(len(l_GPUs))):
                self.dfsc.at[Model_idx,"GPU_idx"]=l_GPUs[Model_idx]
            self.dfsc.to_csv(config_fnwp, index=False)

    def initialization_strategy(self):
        self.strategy_setup(flag_init=True)
    def load_strategy(self):
        self.strategy_setup(flag_init=False)
    def strategy_setup(self, flag_init):
        RL_models_basedir=os.path.join(self.Strategy_dir,"RL_models")
        RL_configs_basedir = os.path.join(self.Strategy_dir, "RL_configs")
        if flag_init:
            if not os.path.exists(RL_models_basedir): os.mkdir(RL_models_basedir)
            if not os.path.exists(RL_configs_basedir): os.mkdir(RL_configs_basedir)

        self.RL_weights_fnwps, self.rlcs =[], []
        for Model_idx, row in self.dfsc.iterrows():
            model_dir=os.path.join(RL_models_basedir,"model{0}".format(Model_idx))
            if not os.path.exists(model_dir): os.mkdir(model_dir)
            config_dir=os.path.join(RL_configs_basedir,"config{0}".format(Model_idx))
            if not os.path.exists(config_dir): os.mkdir(config_dir)
            RL_config_fnwp = os.path.join(config_dir, "config.json")
            src_RL_config_fnwp = os.path.join(sc.base_dir_RL_system, row["RL_system_name"],"config.json")
            if not os.path.exists(RL_config_fnwp):
                if flag_init:
                    assert os.path.exists(src_RL_config_fnwp), \
                        "RL {0} config local and original copy does not exist".format(row["RL_system_name"])
                    shutil.copy(src_RL_config_fnwp, config_dir)
                else:
                    assert False,"RL {0} config local copy does not exist".format(row["RL_system_name"])
            rlc = sc.gconfig()
            rlc.read_from_json(RL_config_fnwp, system_name=row["RL_system_name"])
            self.rlcs.append(rlc)

            weight_fns = [fn for fn in os.listdir(model_dir) if fn.endswith(".h5") and "T{0}".format(row["RL_Model_ET"]) in fn]
            if len(weight_fns) == 0:
                if flag_init:
                    src_weights_dir=os.path.join(sc.base_dir_RL_system, row["RL_system_name"], "model")
                    src_weight_fns = [fn for fn in os.listdir(src_weights_dir) if fn.endswith(".h5") and "T{0}".
                        format(row["RL_Model_ET"]) in fn]
                    assert len(src_weight_fns) == 1,"Model weight Local and origin copy does not exist ET={0}".format(row["RL_Model_ET"])
                    shutil.copy(os.path.join(src_weights_dir, src_weight_fns[0]), model_dir)
                    weight_fns = [fn for fn in os.listdir(model_dir) if fn.endswith(".h5") and "T{0}".format(row["RL_Model_ET"]) in fn]
                else:
                    assert False,"Model weight Local copy does not exsit ET={0}".format(row["RL_Model_ET"])
            assert len(weight_fns)==1,"{0} has more than one weight file {1}".format(model_dir,weight_fns)
            self.RL_weights_fnwps.append(os.path.join(model_dir,weight_fns[0]))


class Strategy_agent_base(DBI_init_with_TD):
    def __init__(self, portfolio_name, strategy_name,experiment_name,DEcofig):
        DBI_init_with_TD.__init__(self)
        self.isc=Strategy_Config(portfolio_name, strategy_name)
        self.isc.load_strategy()
        self.iec=Experiment_Config(portfolio_name, strategy_name,experiment_name,DEcofig)
        self.i_hfq_tb =hfq_toolbox()
        self.iFH = ATFH(self.isc.Strategy_dir, experiment_name)
        self.set_df_params(self.isc.rlcs)

    def set_df_params(self, rlcs):
        rlc=rlcs[0]
        ##set dataframe params
        # Account_Holding_Items=["TransIDI","Holding_Gu","Holding_Invest","Holding_HRatio","Holding_NPrice","Buy_Times"]
        self.Account_Holding_Items_titles= rlc.account_inform_holding_titles+["HoldingStartDateI"]
        self.Account_Holding_Items_default = [0, 0, 0.0, 1.0, 0.0, 0,00000000]
        self.Account_Holding_Items_types={**rlc.account_inform_holding_types,**{"HoldingStartDateI":int}}

        self.Account_Inform_Items_title=["AfterClosing_HFQRatio","AfterClosing_NPrice","Tradable_flag"]
        self.Account_Inform_Items_default =[1.0,0.0,False]
        self.Account_Inform_Items_types={"AfterClosing_HFQRatio":float,"AfterClosing_NPrice":float,"Tradable_flag":bool}
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

        self.aresult_Titles = ["Stock", "Action", "Action_Result", "Buy_Gu", "Buy_NPrice","Buy_Invest","Sell_Gu", "Sell_NPrice","Sell_Return"]
        self.aresult_types = {"Stock": str, "Action":str, "Action_Result":str, "Buy_Gu":int,"Buy_NPrice":float,"Buy_Invest":float,
                              "Sell_Gu":int, "Sell_NPrice":float,"Sell_Return":float}

        self.a2e_Titles = ["Stock", "Action", "Gu"]
        self.a2e_types={"Stock":str, "Action":str, "Gu":int}

        #self.AccountDetail_titles= ["DateI","Cash_after_closing", "MarketValue_after_closing"]
        #self.AccountDetail_types = {"DateI":int,"Cash_after_closing":float, "MarketValue_after_closing":float}

        self.AccountDetail_titles= ["DateI","Cash_after_closing"]
        self.AccountDetail_types = {"DateI": int, "Cash_after_closing": float}
        #for Model_idx in list(range(len(rlcs))):
        for emidx,row  in self.iec.dfec.iterrows():
            add_title="MarketValue_after_closing_M{0}".format(emidx)
            self.AccountDetail_titles.append(add_title)
            self.AccountDetail_types[add_title]=float
        #self.AccountDetail_types = {"DateI":int,"Cash_after_closing":float, "MarketValue_after_closing":float}

    '''
    def init_stock_list(self):
        l_sl=[]
        for Model_idx, row in self.isc.dfsc.iterrows():
            fnwp_sl = self.iFH.get_SL_fnwp(Model_idx)
            iSL = StockList(row["SL_Name"])
            flag,sl = iSL.get_sub_sl(row["SL_Tag"],row["SL_Idx"])
            assert flag
            pd.DataFrame(sl, columns=["Stock"]).to_csv(fnwp_sl, index=False)
            l_sl.append(sl)
        return l_sl
    '''
    def init_stock_list(self):
        l_sl=[]
        for emidx,row  in self.iec.dfec.iterrows():
            fnwp_sl = self.iFH.get_SL_fnwp(emidx)
            iSL = StockList(self.isc.dfsc.loc[row["Model_idx"]]["SL_Name"])
            flag,sl = iSL.get_sub_sl(row["SL_Tag"],row["SL_Idx"])
            assert flag
            pd.DataFrame(sl, columns=["Stock"]).to_csv(fnwp_sl, index=False)
            l_sl.append(sl)
        return l_sl


    def load_stock_list(self):
        l_sl=[]
        for emidx,row  in self.iec.dfec.iterrows():
            fnwp_sl = self.iFH.get_SL_fnwp(emidx)
            assert os.path.exists(fnwp_sl)
            l_sl.append(pd.read_csv(fnwp_sl)["Stock"].tolist())
        return l_sl

    def Init_df_account(self, l_sl):
        l_df_account=[]
        for emidx,_  in self.iec.dfec.iterrows():
            fnwp_account = self.iFH.get_account_fnwp(emidx)
            df_account=pd.DataFrame(columns=self.account_Titles)
            for stock in l_sl[emidx]:
                row=list(self.account_default)
                row[0]=stock
                df_account.loc[len(df_account)]=row
            df_account = df_account.astype(dtype=self.account_types)
            df_account.set_index(["Stock"],drop=True,inplace=True,verify_integrity=True)
            df_account.to_csv(fnwp_account,float_format='%.2f')
            l_df_account.append(df_account)
        return l_df_account

    def load_df_account(self):
        l_df_account=[]
        for emidx,_  in self.iec.dfec.iterrows():
            fnwp_account=self.iFH.get_account_fnwp(emidx)
            assert os.path.exists(fnwp_account),fnwp_account
            df_account = pd.read_csv(fnwp_account)
            df_account = df_account.astype(dtype=self.account_types)
            df_account.set_index(["Stock"],drop=True,inplace=True,verify_integrity=True)
            l_df_account.append(df_account)
        return l_df_account

    def save_df_account(self, l_df_account, DateI):
        for emidx, df_account in enumerate(l_df_account):
            fnwp_account = self.iFH.get_account_fnwp(emidx)
            fnwp_account_backup = self.iFH.get_account_backup_fnwp(DateI,emidx)
            df_account.to_csv(fnwp_account,float_format='%.2f')   #here need to save index, it is stock
            df_account.to_csv(fnwp_account_backup,float_format='%.2f') #here need to save index, it is stock

    def save_next_day_action(self, DateI, ll_a,l_sl,l_df_account):
        l_df_e2a=[]
        for emidx,row  in self.iec.dfec.iterrows():
            a2e_fnwp=self.iFH.get_a2e_fnwp(DateI,emidx)
            df_e2a=pd.DataFrame(columns=self.a2e_Titles)
            df_e2a=df_e2a.astype(self.a2e_types)
            df_e2a.set_index(["Stock"], drop=True, inplace=True,verify_integrity=True)
            df_account=l_df_account[emidx]
            min_invest=row["min_invest"]
            for action, stock in zip(ll_a[emidx],l_sl[emidx]):
                if not df_account.at[stock, "Tradable_flag"]:
                    print("these situation should already be removed before call save_next_day_action {0} {1}".format(stock, DateI))
                    continue
                if action ==0:
                    df_e2a.at[stock, "Action"] = "Buy"
                    df_e2a.at[stock, "Gu"] = (min_invest/df_account.loc[stock,"AfterClosing_NPrice"])//100*100
                elif action ==2:
                    df_e2a.at[stock, "Action"] = "Sell"
                    df_e2a.at[stock, "Gu"]= self.i_hfq_tb.get_update_volume_on_hfq_ratio_change\
                        (df_account.loc[stock,"Holding_HRatio"], df_account.loc[stock,"AfterClosing_HFQRatio"],
                         df_account.loc[stock]["Holding_Gu"])
                else:
                    continue
            df_e2a.to_csv(a2e_fnwp,float_format='%.2f')
            l_df_e2a.append(df_e2a)
        return l_df_e2a

    def load_df_a2eDone(self, DateI):
        l_df_e2aDone=[]
        for emidx,row  in self.iec.dfec.iterrows():
            fnwp_action2exeDone = self.iFH.get_a2eDone_fnwp(DateI,emidx)
            assert os.path.exists(fnwp_action2exeDone), "{0} does not exists".format(fnwp_action2exeDone)
            df_a2eDone=pd.read_csv(fnwp_action2exeDone)
            if len(df_a2eDone)!=0:
                df_a2eDone=df_a2eDone.astype(self.a2e_types)
                df_a2eDone.set_index(["Stock"], drop=True, inplace=True,verify_integrity=True)
            l_df_e2aDone.append(df_a2eDone)
        return l_df_e2aDone

    def load_df_aresult(self, DateI):
        #        self.aresult_Titles = ["Stock", "Action", "Action_Result", "Buy_Gu", "Buy_NPrice","Buy_Invest","Sell_Gu", "Sell_NPrice","Sell_Return"]
        l_df_aresult=[]
        for emidx,_  in self.iec.dfec.iterrows():
            fnwp_action_result=self.iFH.get_aresult_fnwp(DateI,emidx)
            assert os.path.exists(fnwp_action_result), "{0} does not exists".format(fnwp_action_result)
            df_aresult = pd.read_csv(fnwp_action_result)
            df_aresult= df_aresult.astype(self.aresult_types)
            if len(df_aresult)!=0:
                df_aresult= df_aresult.groupby(["Stock"]).agg(
                    Action=pd.NamedAgg(column='Action', aggfunc=lambda x: [xi for xi in x][0]),
                    Action_Result=pd.NamedAgg(column='Action_Result', aggfunc=lambda x: [xi for xi in x][0]),
                    Buy_Gu=pd.NamedAgg(column='Buy_Gu', aggfunc='sum'),
                    Buy_NPrice=pd.NamedAgg(column='Buy_NPrice', aggfunc='mean'),
                    Buy_Invest=pd.NamedAgg(column='Buy_Invest', aggfunc='sum'),
                    Sell_Gu=pd.NamedAgg(column='Sell_Gu', aggfunc='sum'),
                    Sell_NPrice=pd.NamedAgg(column='Sell_NPrice', aggfunc='mean'),
                    Sell_Return=pd.NamedAgg(column='Sell_Return', aggfunc='sum')
                )
            l_df_aresult.append(df_aresult)
        return l_df_aresult

    def Check_all_actionDone_has_result(self,df_a2eDones,df_aresults):
        for df_a2eDone,df_aresult in zip(df_a2eDones,df_aresults):
            assert len(df_a2eDone)==len(df_aresult)
            if len(df_a2eDone)!=0:
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
        Cash_afterclosing=account_day_inform["Cash_after_closing"]
        l_MarketValue_afterclosing=[account_day_inform["MarketValue_after_closing_M{0}".format(emidx)] for emidx,_  in self.iec.dfec.iterrows()]
        #Cash_afterclosing,MarketValue_afterclosing = \
        #    account_day_inform["Cash_after_closing"],account_day_inform["MarketValue_after_closing"]
        #print ("Loaded account detail before {0} Cash_afterclosing {1:.2f} Market value {2:.2f} total {3:.2f}".format
        #       (account_day_inform.name,Cash_afterclosing,MarketValue_afterclosing,Cash_afterclosing+MarketValue_afterclosing))
        return df_account_detail,Cash_afterclosing,l_MarketValue_afterclosing

    def update_save_account_detail(self,df_account_detail, DateI,Cash_afterclosing,l_MarketValue_afterclosing):
        df_account_detail.loc[DateI]=[Cash_afterclosing]+l_MarketValue_afterclosing
        fnwp_account_detail=self.iFH.get_account_detail_fnwp()
        fnwp_account_detail_backup = self.iFH.get_account_detail_backup_fnwp(DateI)
        df_account_detail.to_csv(fnwp_account_detail,float_format='%.2f') #here need to save index, it is DateI
        df_account_detail.to_csv(fnwp_account_detail_backup,float_format='%.2f') #here need to save index, it is DateI

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
        if len(a)==0:
            return True, [0,1.0, False], mess   # Todo this is correct not IPO stock period
        else:
            Closing_Nprice=a.iloc[-1]["close_price"]/a.iloc[-1]["coefficient_fq"]
            HFQRatio = a.iloc[-1]["coefficient_fq"]
            assert Closing_Nprice!=0,"{0} {1} {2}".format(Stock,DateI,a)
            return True, [Closing_Nprice, HFQRatio, True], "Success"

#class Strategy_agent_Report:
    def prepare_report(self, DateI, logs, l_df_e2a, l_sl):
        Cash_afterclosing, l_MarketValue_afterclosing, mumber_of_stock_could_buy, \
        ll_log_bought, ll_log_Earnsold, ll_log_balancesold, ll_log_Losssold, \
        ll_log_fail_action, ll_log_holding_with_no_action, \
        ll_ADlog = logs
        report_fnwp = self.iFH.get_report_fnwp(DateI)
        with open(report_fnwp, "w") as f:
            f.write("Date: {0} Cash After Closing: {1:.2f} ".format(DateI, Cash_afterclosing))
            for emidx, MarketValue_afterclosing in enumerate(l_MarketValue_afterclosing):
                f.write("Model{0} Market Value: {1:.2f} ".format(emidx,MarketValue_afterclosing))
            f.write("Total: {0:.2f}\n".format(Cash_afterclosing+sum(l_MarketValue_afterclosing)))
            f.write("Num of Stock could bought tomorrow: {0}\n".format(mumber_of_stock_could_buy))

            for print_format_str, ll_log in [["Today Bought {0} stock:\n",ll_log_bought],
                                             ["Today Sold with Earn {0} stock:\n",ll_log_Earnsold],
                                             ["Today Sold with Loss {0} stock:\n",ll_log_Losssold],
                                             ["Today Sold with Balance {0} stock:\n",ll_log_balancesold]]:
                f.write(print_format_str.format(sum([len(l_log) for l_log in ll_log])))
                for Model_idx, l_log in enumerate(ll_log):
                    if len(l_log)!=0:
                        f.write("\tModel{0} {1}\n".format(Model_idx,",".join(l_log)))

            for print_format_str, lll_log in [["Today {0} Action with Fail :\n",ll_log_fail_action],
                                             ["Today holding without sell action {0} :\n",ll_log_holding_with_no_action]]:
                f.write(print_format_str.format(sum([len(ll_log) for ll_log in lll_log])))
                for Model_idx, ll_log in enumerate(lll_log):
                    if len(ll_log)!=0:
                        for l_log in ll_log:
                            f.write("\tModel{0} {1}\n".format(Model_idx,",".join(l_log)))


            for action_str in ["Buy","Sell"]:
                f.write("Tommorow to {0} {1}\n".format(action_str,sum([len(df_e2a[df_e2a["Action"] == action_str]) for df_e2a in l_df_e2a])))
                for Model_idx,df_e2a in enumerate(l_df_e2a):
                    a=df_e2a[df_e2a["Action"] == action_str].index.tolist()
                    if len(a)!=0:
                        f.write("\tModel{0} {1}: {2}\n".format(Model_idx,action_str,",".join(a)))

            assert ll_ADlog[0][0] == DateI

            for print_format_str, ADlog_ll_idx in [["Tommorow not_buy_due_limit {0}\n",1],[ "Tommorow multibuy(not sell due to multibuy) {0}\n",4]]:
                ll_result=[]
                for emidx in list(range(len(ll_ADlog))):
                    ll_result.append(ll_ADlog[emidx][ADlog_ll_idx].split("_"))
                count=sum([len(l_result[0]) for l_result in ll_result])
                f.write(print_format_str.format(count))
                for emidx in list(range(len(ll_result))):
                    if len(ll_result[emidx][0])!=0:
                        #print (ll_result[Model_idx],file=sys.__stdout__)
                        f.write("\tModel{0}: {1}\n".format(emidx,",".join([l_sl[emidx][int(sidx)] for sidx in ll_result[emidx]])))

        print("{0} Report stored at {1}\n".format(DateI, report_fnwp))


class Trader_GPU(Process):
    def __init__(self, portfolio, strategy,Model_idx,E_Stop_GPU, L_Agent2GPU,LL_GPU2Agent):
        Process.__init__(self)
        self.Model_idx,self.E_Stop_GPU,self.L_Agent2GPU,self.LL_GPU2Agent=\
            Model_idx,E_Stop_GPU, L_Agent2GPU,LL_GPU2Agent
        self.iStrategy=Strategy_Config(portfolio, strategy)
        self.iStrategy.load_strategy()
        self.process_name=self.__class__.__name__+str(self.Model_idx)

    def run(self):
        setproctitle.setproctitle(self.process_name)
        import tensorflow as tf
        from nets import Explore_Brain,init_virtual_GPU
        tf.random.set_seed(2)
        random.seed(2)
        np.random.seed(2)
        virtual_GPU = init_virtual_GPU(self.iStrategy.dfsc.loc[self.Model_idx]["GPU_mem"])
        with tf.device(virtual_GPU):
            i_eb = locals()[self.iStrategy.rlcs[self.Model_idx].CLN_brain_explore](self.iStrategy.rlcs[self.Model_idx])
            i_eb.load_weight(self.iStrategy.RL_weights_fnwps[self.Model_idx])
            print("Loaded model from {0} ".format(self.iStrategy.RL_weights_fnwps[self.Model_idx]))
            current_DateI=0
            AP_buffer=[]
            while not self.E_Stop_GPU.is_set():
                if len(self.L_Agent2GPU)!=0:
                    process_idx,stacked_state=self.L_Agent2GPU.pop()
                    l_AP_OB, l_AP_OS = i_eb.V3_get_AP_AT(stacked_state, calledby="")
                    self.LL_GPU2Agent[process_idx].append([l_AP_OB, l_AP_OS])

                    #l_a_OB, l_a_OS = i_eb.V3_choose_action_CC(stacked_state, calledby="")
                    #self.LL_GPU2Agent[process_idx].append([l_a_OB, l_a_OS])

                else:
                    time.sleep(0.1)

class strategy_sim(DBI_init_with_TD):
    def __init__(self, Strategy_dir, experiment_name,NumEModel,a2e_types,aresult_Titles):
        DBI_init_with_TD.__init__(self)
        self.iFH = ATFH(Strategy_dir, experiment_name)
        self.NumEModel,self.a2e_types,self.aresult_Titles=NumEModel,a2e_types,aresult_Titles
        self.i_RawData=RawData()
    def sim_load_df_a2e(self, DateI):
        l_df_a2e=[]
        for emidx in list(range(self.NumEModel)):
            fnwp_action2exe = self.iFH.get_a2e_fnwp(DateI,emidx)
            assert os.path.exists(fnwp_action2exe), "{0} does not exists".format(fnwp_action2exe)
            df_a2e = pd.read_csv(fnwp_action2exe)
            df_a2e = df_a2e.astype(self.a2e_types)
            df_a2e.set_index(["Stock"], drop=True, inplace=True,verify_integrity=True)
            print("Loaded a2e from ", fnwp_action2exe)
            l_df_a2e.append(df_a2e)
        return l_df_a2e

    def sim(self, YesterdayI, DateI):
        l_df_a2e = self.sim_load_df_a2e(YesterdayI)
        for Model_idx, df_a2e in enumerate(l_df_a2e):
            df_a2e.to_csv(self.iFH.get_a2eDone_fnwp(DateI,Model_idx),float_format='%.2f')

            df_aresult = pd.DataFrame(columns=self.aresult_Titles)
            # roughly buy 0.0003
            # roughly sell 0.0013
            for idx, row in df_a2e.iterrows():
                #print (row)
                stock, gu = row.name, row["Gu"]  # stock is index in Seris it is name
                flag, dfhqr, message = self.get_hfq_df(self.get_DBI_hfq_fnwp(stock))
                assert flag,message
                ahqr = dfhqr[dfhqr["date"] == str(DateI)]
                if not ahqr.empty:
                    flag, dfqz, message = self.i_RawData.get_qz_df_inteface(row.name, DateI)
                    assert flag
                    for low, high in [[93000, 93500], [93500, 94000], [94000, 94500], [94500, 95000], [95000, 95500],
                                      [95500, 96000]]:
                        aqz = dfqz[(dfqz["Time"] >= low) & (dfqz["Time"] < high)]
                        if not aqz.empty: break
                    else:
                        assert False, "QZ first half hour does not have trade record {0} {1} {2}".format(stock, gu, DateI)
                    num_trans = min(np.random.choice([1, 2, 3], p=[1 / 3, 1 / 3, 1 / 3]), len(aqz))
                    NPrices = aqz["Price"].to_list()
                    random.shuffle(NPrices)

                    gu_avg = gu // num_trans
                    l_Trans_Gu = [gu_avg if idx < num_trans - 1 else gu - (num_trans - 1) * gu_avg for idx in
                                  list(range(num_trans))]
                    l_Trans_Price = NPrices[:num_trans]
                    if row["Action"] == "Buy":
                        for trans_Gu, trans_Price in zip(l_Trans_Gu, l_Trans_Price):
                            #        self.aresult_Titles = ["Stock", "Action", "Action_Result", "Buy_Gu", "Buy_NPrice","Buy_Invest","Sell_Gu", "Sell_NPrice","Sell_Return"]
                            df_aresult.loc[len(df_aresult)] = [stock, "Buy", "Success", trans_Gu,trans_Price,trans_Gu * trans_Price * 1.0003, 0, 0.0, 0.0]
                    elif row["Action"] == "Sell":
                        for trans_Gu, trans_Price in zip(l_Trans_Gu, l_Trans_Price):
                            df_aresult.loc[len(df_aresult)] = [stock, "Sell", "Success", 0, 0.0, 0.0,trans_Gu,trans_Price,trans_Gu * trans_Price * (1 - 0.0013)]
                    else:
                        assert False, "Action only can by Buy or Sell not {0}".format(row["Action"])
                else:
                    df_aresult.loc[len(df_aresult)] = [stock, row["Action"], "Tinpai", 0,0.0,0.0,0,0.0,0.0]
            df_aresult.to_csv(self.iFH.get_aresult_fnwp(DateI,Model_idx), index=False,float_format='%.2f')
        return

