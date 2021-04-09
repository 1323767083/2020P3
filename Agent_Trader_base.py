import os,json, shutil,sys,random,setproctitle,time
from collections import OrderedDict
import pandas as pd
import numpy as np
from multiprocessing import Process,Event,Manager
import config as sc
from DBI_Base import DBI_init_with_TD,hfq_toolbox,StockList
import DBTP_Reader
from DBR_Reader import RawData
from State import AV_Handler_AV1
from Buy_Strategies import Buy_Strategies

AT_base_dir = "/home/rdchujf/n_workspace/AT"
#Strategy_config
{
    "strategy_fun": "Buy_Strategy_multi_time_Direct_sell",
    "RL_system_name": "SSS",
    "RL_Model_ET": "250",
    "GPU_mem":2600,
    "SL_Name": "",
    "SL_Tag": "",
    "SL_Idx": "",
    "TPDB_Name": ""
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

    def _get_month_date_dir(self, base_dir, DateI):
        monthI=DateI//100
        dir=base_dir
        for subdir in [str(monthI), str(DateI)]:
            dir=os.path.join(dir, subdir)
            if not os.path.exists(dir): os.mkdir(dir)
        return dir
    def get_SL_fnwp(self):
        return os.path.join(self.AT_account_dir,"SL_in_order.csv")
    def get_account_fnwp(self):
        return os.path.join(self.AT_account_dir,"AT_account.csv")
    def get_account_backup_fnwp(self,DateI):
        return os.path.join(self._get_month_date_dir(self.AT_account_dir,DateI),"AT_account_afterday_backup.csv")
    def get_aresult_fnwp(self,DateI):
        return os.path.join(self._get_month_date_dir(self.AT_account_dir,DateI),"AT_StepResult.csv")
    def get_a2eDone_fnwp(self,DateI):
        return os.path.join(self._get_month_date_dir(self.AT_account_dir,DateI),"AT_Action2exeDone.csv")
    def get_a2e_fnwp(self,DateI):
        return os.path.join(self._get_month_date_dir(self.AT_account_dir,DateI),"AT_Action2exe.csv")
    def get_account_detail_fnwp(self):
        return os.path.join(self.AT_account_dir,"AT_AccountDetail.csv")
    def get_account_detail_backup_fnwp(self,DateI):
        return os.path.join(self._get_month_date_dir(self.AT_account_dir,DateI), "AT_AccountDetail_backup.csv")
    def get_report_fnwp(self, DateI):
        return os.path.join(self._get_month_date_dir(self.AT_account_dir,DateI), "Report.txt")

class Experiment_Config:
    total_invest=float("NaN")
    min_invest=float("NaN")
    StartI=float("NaN")
    EndI=float("NaN")
    flag_Print_on_screen_or_file=False
    def __init__(self, portfolio_name, strategy_name, experiment_name, experiment_config_params):
        self.Experiment_dir = os.path.join(AT_base_dir, portfolio_name, strategy_name, experiment_name)
        self.experiment_name = experiment_name
        config_fnwp = os.path.join(self.Experiment_dir, "config.json")
        if len(experiment_config_params)==0:
            param = json.load(open(config_fnwp, "r"), object_pairs_hook=OrderedDict)
            for item in list(param.keys()):
                if not item.startswith("======="):
                    self.__dict__[item] = param[item]
        else:
            self.total_invest, self.min_invest, self.StartI, self.EndI, self.flag_Print_on_screen_or_file=experiment_config_params
            if not os.path.exists(self.Experiment_dir): os.mkdir(self.Experiment_dir)
            a=OrderedDict()
            for title in ["total_invest","min_invest","StartI","EndI","flag_Print_on_screen_or_file"]:
                a[title]=getattr(self,title)
            json.dump(a,open(config_fnwp, "w"))


class Strategy_Config:
    strategy_fun=""
    RL_system_name=""
    RL_Model_ET=float("NaN")
    GPU_mem=float("NaN")
    SL_Name=""
    SL_Tag=""
    SL_Idx=float("NaN")
    TPDB_Name=""

    def __init__(self,portfolio_name, strategy_name):
        #load strategy config
        self.Strategy_dir=os.path.join(AT_base_dir,portfolio_name, strategy_name)
        self.strategy_name = strategy_name
        self.portfolio_name=portfolio_name
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

class Strategy_agent_base(Strategy_Config,Experiment_Config,DBI_init_with_TD):
    def __init__(self, portfolio_name, strategy_name,experiment_name,experiment_config_params):
        Strategy_Config.__init__(self, portfolio_name, strategy_name)
        Experiment_Config.__init__(self, portfolio_name, strategy_name,experiment_name,experiment_config_params)
        DBI_init_with_TD.__init__(self)
        self.i_hfq_tb =hfq_toolbox()
        self.iFH = ATFH(self.Strategy_dir, experiment_name)
        self.set_df_params(self.rlc)

    def set_df_params(self, rlc):
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

        self.AccountDetail_titles= ["DateI","Cash_after_closing", "MarketValue_after_closing"]
        self.AccountDetail_types = {"DateI":int,"Cash_after_closing":float, "MarketValue_after_closing":float}

    def init_stock_list(self):

        fnwp_sl = self.iFH.get_SL_fnwp()
        if self.SL_Name == "":
            iSL = StockList(self.rlc.SLName)
            flag,sl = iSL.get_sub_sl("Train",0)
        else:
            iSL = StockList(self.SL_Name)
            flag,sl = iSL.get_sub_sl(self.SL_Tag,self.SL_Idx)
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
        df_account.to_csv(fnwp_account,float_format='%.2f')
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
        df_account.to_csv(fnwp_account,float_format='%.2f')   #here need to save index, it is stock
        df_account.to_csv(fnwp_account_backup,float_format='%.2f') #here need to save index, it is stock

    def save_next_day_action(self, DateI, l_a,stocklist,df_account):
        a2e_fnwp=self.iFH.get_a2e_fnwp(DateI)
        df_e2a=pd.DataFrame(columns=self.a2e_Titles)
        df_e2a=df_e2a.astype(self.a2e_types)
        df_e2a.set_index(["Stock"], drop=True, inplace=True,verify_integrity=True)
        for action, stock in zip(l_a,stocklist):
            if not df_account.at[stock, "Tradable_flag"]:
                continue
            if action ==0:
                df_e2a.at[stock, "Action"] = "Buy"
                df_e2a.at[stock, "Gu"] = (self.min_invest/df_account.loc[stock,"AfterClosing_NPrice"])//100*100
            elif action ==2:
                df_e2a.at[stock, "Action"] = "Sell"
                df_e2a.at[stock, "Gu"]= self.i_hfq_tb.get_update_volume_on_hfq_ratio_change\
                    (df_account.loc[stock,"Holding_HRatio"], df_account.loc[stock,"AfterClosing_HFQRatio"],
                     df_account.loc[stock]["Holding_Gu"])
            else:
                continue
        df_e2a.to_csv(a2e_fnwp,float_format='%.2f')
        return df_e2a

    def load_df_a2eDone(self, DateI):
        fnwp_action2exeDone = self.iFH.get_a2eDone_fnwp(DateI)
        assert os.path.exists(fnwp_action2exeDone), "{0} does not exists".format(fnwp_action2exeDone)
        df_a2eDone=pd.read_csv(fnwp_action2exeDone)
        if len(df_a2eDone)!=0:
            df_a2eDone=df_a2eDone.astype(self.a2e_types)
            df_a2eDone.set_index(["Stock"], drop=True, inplace=True,verify_integrity=True)
        return df_a2eDone

    def load_df_aresult(self, DateI):
        #        self.aresult_Titles = ["Stock", "Action", "Action_Result", "Buy_Gu", "Buy_NPrice","Buy_Invest","Sell_Gu", "Sell_NPrice","Sell_Return"]
        fnwp_action_result=self.iFH.get_aresult_fnwp(DateI)
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
        return df_aresult

    def Check_all_actionDone_has_result(self,df_a2eDone,df_aresult):
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
        Cash_afterclosing,MarketValue_afterclosing = \
            account_day_inform["Cash_after_closing"],account_day_inform["MarketValue_after_closing"]
        #print ("Loaded account detail before {0} Cash_afterclosing {1:.2f} Market value {2:.2f} total {3:.2f}".format
        #       (account_day_inform.name,Cash_afterclosing,MarketValue_afterclosing,Cash_afterclosing+MarketValue_afterclosing))
        return df_account_detail,Cash_afterclosing,MarketValue_afterclosing

    def update_save_account_detail(self,df_account_detail, DateI,Cash_afterclosing,MarketValue_afterclosing):
        df_account_detail.loc[DateI]=[Cash_afterclosing,MarketValue_afterclosing]
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

class Trader_GPU(Process):
    def __init__(self, iStrategy,E_Stop_GPU, L_Agent2GPU,LL_GPU2Agent):
        Process.__init__(self)
        self.iStrategy,self.E_Stop_GPU,self.L_Agent2GPU,self.LL_GPU2Agent=\
            iStrategy,E_Stop_GPU, L_Agent2GPU,LL_GPU2Agent
        self.process_name=self.__class__.__name__

    def run(self):
        setproctitle.setproctitle(self.process_name)
        import tensorflow as tf
        from nets import Explore_Brain,init_virtual_GPU
        tf.random.set_seed(2)
        random.seed(2)
        np.random.seed(2)
        virtual_GPU = init_virtual_GPU(self.iStrategy.GPU_mem)
        with tf.device(virtual_GPU):
            i_eb = locals()[self.iStrategy.rlc.CLN_brain_explore](self.iStrategy.rlc)
            i_eb.load_weight(os.path.join(self.iStrategy.weight_fnwp))
            print("Loaded model from {0} ".format(self.iStrategy.weight_fnwp))
            while not self.E_Stop_GPU.is_set():
                if len(self.L_Agent2GPU)!=0:
                    process_idx,stacked_state=self.L_Agent2GPU.pop()
                    #result = self.i_wb.choose_action(stacted_state, "Explore")
                    l_a_OB, l_a_OS = i_eb.V3_choose_action_CC(stacked_state, calledby="")
                    self.LL_GPU2Agent[process_idx].append([l_a_OB, l_a_OS])
                else:
                    time.sleep(0.1)

class strategy_sim(DBI_init_with_TD):
    def __init__(self, iFH,a2e_types,aresult_Titles):
        DBI_init_with_TD.__init__(self)
        self.iFH,self.a2e_types,self.aresult_Titles=iFH,a2e_types,aresult_Titles
        self.i_RawData=RawData()
    def sim_load_df_a2e(self, DateI):
        fnwp_action2exe = self.iFH.get_a2e_fnwp(DateI)
        assert os.path.exists(fnwp_action2exe), "{0} does not exists".format(fnwp_action2exe)
        df_a2e = pd.read_csv(fnwp_action2exe)
        df_a2e = df_a2e.astype(self.a2e_types)
        df_a2e.set_index(["Stock"], drop=True, inplace=True,verify_integrity=True)
        print("Loaded a2e from ", fnwp_action2exe)
        return df_a2e

    def sim(self, YesterdayI, DateI):
        df_a2e = self.sim_load_df_a2e(YesterdayI)
        df_a2e.to_csv(self.iFH.get_a2eDone_fnwp(DateI),float_format='%.2f')
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
        df_aresult.to_csv(self.iFH.get_aresult_fnwp(DateI), index=False,float_format='%.2f')
        return

class Strategy_agent_Report:
    def prepare_report(self, DateI, logs, df_e2a, sl, report_fnwp):
        Cash_afterclosing, MarketValue_afterclosing, mumber_of_stock_could_buy, \
        l_log_bought, l_log_Earnsold, l_log_balancesold, l_log_Losssold, \
        l_log_fail_action, l_log_holding_with_no_action, \
        l_ADlog = logs
        #fnwp = self.iFH.get_report_fnwp(DateI)
        with open(report_fnwp, "w") as f:
            f.write("Date: {0} Cash After Closing: {1:.2f} Market Value: {2:.2f}\n Num of Stock could bought tomorrow: {3}\n".
                    format(DateI, Cash_afterclosing, MarketValue_afterclosing, mumber_of_stock_could_buy))
            f.write("Total: {0:.2f}".format(Cash_afterclosing+MarketValue_afterclosing))
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

