import os,json, shutil
from collections import OrderedDict
import config as sc
import pandas as pd
import numpy as np
import DBI_Base
import Eval_CC
import DBTP_Reader
from nets import Explore_Brain
from State import AV_Handler_AV1
from Buy_Strategies import Buy_Strategies
from nets import Explore_Brain
{
"selected_strategys":[1,3],
}

{
    "strategy_name": "ZZZ",
    "total_invest": 250000,
    "strategy_fun": "Buy_Strategy_multi_time_Direct_sell",
    "min_invest": 50000,
    "RL_system_name": "SSS",
    "RL_Model_ET": "250"
}

class ATFH:
    def __init__(self,Strategy_dir):
        self.AT_account_dir = os.path.join(Strategy_dir, "Account")
        if not os.path.exists(self.AT_account_dir): os.mkdir(self.AT_account_dir)
        self.AT_log_dir=os.path.join(Strategy_dir, "Log")
        if not os.path.exists(self.AT_log_dir): os.mkdir(self.AT_log_dir)
        self.AT_model_dir=os.path.join(Strategy_dir,"model")
        if not os.path.exists(self.AT_model_dir): os.mkdir(self.AT_model_dir)
        self.RL_config_dir=os.path.join(Strategy_dir,"RL_config")
        if not os.path.exists(self.RL_config_dir): os.mkdir(self.RL_config_dir)
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

class Strategy_Config:
    AT_base_dir="/home/rdchujf/n_workspace/AT"
    #set default value for param
    total_invest=float("NaN")
    strategy_fun=""
    min_invest=float("NaN")
    RL_system_name=""
    RL_Model_ET=float("NaN")

    def __init__(self,portfolio_name, strategy_name):
        #load strategy config
        self.Strategy_dir=os.path.join(self.AT_base_dir,portfolio_name, strategy_name)
        self.strategy_name = strategy_name
        config_fnwp=os.path.join(self.Strategy_dir,"config.json")
        param = json.load(open(config_fnwp, "r"), object_pairs_hook=OrderedDict)
        for item in list(param.keys()):
            if not item.startswith("======="):
                self.__dict__[item] = param[item]

        self.iFH = ATFH(self.Strategy_dir)

        #load selkected RL config
        RL_config_fnwp=os.path.join(self.iFH.RL_config_dir, "config.json")
        if not os.path.exists(RL_config_fnwp):
            src_RL_config_fnwp=os.path.join(sc.base_dir_RL_system, self.RL_system_name, "config.json")
            assert os.path.exists(src_RL_config_fnwp),\
                "RL {0} config local copy does not exsit and original copy can not found".format(self.RL_system_name)
            shutil.copy(src_RL_config_fnwp,self.iFH.RL_config_dir)
        self.rlc = sc.gconfig()
        self.rlc.read_from_json(RL_config_fnwp,system_name=self.RL_system_name)
        self.set_df_params(self.rlc)

    def set_df_params(self, rlc):
        ##set dataframe params
        # Account_Holding_Items=["TransIDI","Holding_Gu","Holding_Invest","Holding_HRatio","Holding_NPrice","Buy_Times"]
        self.Account_Holding_Items_titles= rlc.account_inform_holding_titles+["HoldingStartDateI"]
        self.Account_Holding_Items_default = [0, 0, 0.0, 1.0, 0.0, 0.0,00000000]
        self.Account_Holding_Items_types={**rlc.account_inform_holding_types,**{"HoldingStartDateI":int}}
        # Account_Step_Items=["Buy_Invest","Buy_NPrice","Sell_Return","Sell_NPrice","Sell_Earn","Tinpai_huaizhang", "Action"]
        self.Account_Step_Items_titles = rlc.account_inform_step_titles + ["Action"]
        self.Account_Step_Items_default = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1]
        self.Account_Step_Items_types={**rlc.account_inform_step_types,**{"Action":str}}
        self.account_Titles = ["Stock"] + self.Account_Holding_Items_titles + self.Account_Step_Items_titles
        self.account_types={**{"Stock":str},**self.Account_Holding_Items_types,**self.Account_Step_Items_types}
        self.account_default = [""] + self.Account_Holding_Items_default + self.Account_Step_Items_default

        self.aresult_Titles = ["Stock", "Action", "Action_Result", "Buy_HRatio", "Buy_Gu", "Buy_Invest",
                                    "Buy_NPrice", "Sell_Return", "Sell_NPrice"]
        self.aresult_types = {"Stock": str, "Action":str, "Action_Result":str, "Buy_HRatio":float, "Buy_Gu":int,
                           "Buy_Invest":float, "Buy_NPrice":float, "Sell_Return":float, "Sell_NPrice":float}
        self.a2e_Titles = ["Stock", "Action", "Gu"]
        self.a2e_types={"Stock":str, "Action":str, "Gu":int}

        self.AccountDetail_titles= ["DateI","Cash_after_closing", "MarketValue_after_closing"]
        self.AccountDetail_types = {"DateI":int,"Cash_after_closing":float, "MarketValue_after_closing":float}

class Strategy_agent(Strategy_Config):
    def __init__(self, portfolio_name, strategy_name):
        Strategy_Config.__init__(self, portfolio_name, strategy_name)
        self.i_hfq_tb =DBI_Base.hfq_toolbox()
        self.i_get_data= DBTP_Reader.DBTP_Reader(self.rlc.data_name)   #no logic, only raw read data
        self.i_cav = globals()[self.rlc.CLN_AV_Handler](self.rlc)
        self.i_buystrategy = Buy_Strategies(self.rlc)

    def load_stock_list(self):
        fnwp_sl=self.iFH.get_SL_fnwp()
        if not os.path.exists(fnwp_sl):
            iSL = DBI_Base.StockList(self.rlc.SLName)
            sl = iSL.Sanity_Check_SL(iSL.Get_Total_SL(), flag_remove_adj=False, flag_remove_price=True)
            pd.DataFrame(sl, columns=["Stock"]).to_csv(fnwp_sl, index=False)
        else:
            sl=pd.read_csv(fnwp_sl)["Stock"].tolist()
        print("Loaded stock list from ", fnwp_sl)
        return sl

    def load_predict_model(self):
        weight_fns=[fn for fn in os.listdir(self.iFH.AT_model_dir) if fn.endswith(".h5") and "T{0}".
            format(self.RL_Model_ET) in fn ]
        if len(weight_fns)!=1:
            src_weight_fns=[fn for fn in os.listdir(self.rlc.brain_model_dir) if fn.endswith(".h5") and "T{0}".
                format(self.RL_Model_ET) in fn ]
            assert len(src_weight_fns)==1, \
                "Model weight Local copy does not exsit and original copy can not found ET={0}".format(self.RL_Model_ET)
            shutil.copy(os.path.join(self.rlc.brain_model_dir,src_weight_fns[0]),self.iFH.AT_model_dir)
        i_eb = globals()[self.rlc.CLN_brain_explore](self.rlc)
        i_eb.load_weight(os.path.join(self.iFH.AT_model_dir,weight_fns[0]))
        print("Loaded model from {0} {1}".format(self.iFH.AT_model_dir,weight_fns[0]))
        return i_eb

    def load_df_account(self):
        fnwp_account=self.iFH.get_account_fnwp()
        assert os.path.exists(fnwp_account),fnwp_account
        df_account = pd.read_csv(fnwp_account)
        df_account = df_account.astype(dtype=self.account_types)
        df_account.set_index(["Stock"],drop=True,inplace=True)
        print("Loaded account from ",fnwp_account )
        return df_account

    def load_df_a2eDone(self, DateI):
        fnwp_action2exeDone = self.iFH.get_a2eDone_fnwp(DateI)
        assert os.path.exists(fnwp_action2exeDone), "{0} does not exists".format(fnwp_action2exeDone)
        df_a2eDone=pd.read_csv(fnwp_action2exeDone)
        df_a2eDone=df_a2eDone.astype(self.a2e_types)
        df_a2eDone.set_index(["Stock"], drop=True, inplace=True)
        print("Loaded action to exec from ", fnwp_action2exeDone)
        return df_a2eDone

    def load_df_aresult(self, DateI):
        fnwp_action_result=self.iFH.get_aresult_fnwp(DateI)
        assert os.path.exists(fnwp_action_result), "{0} does not exists".format(fnwp_action_result)
        df_aresult = pd.read_csv(fnwp_action_result)
        df_aresult= df_aresult.astype(self.aresult_types)
        df_aresult.set_index(["Stock"], drop=True, inplace=True)
        print("Loaded action result from ", fnwp_action_result)
        return df_aresult

    def Check_all_actionDone_has_result(self,df_a2eDone,df_aresult):
        step_SA= set(df_aresult.apply(lambda x: x.name + x["Action"], axis=1).tolist())
        a2e_SA = set(df_a2eDone.apply(lambda x: x.name + x["Action"], axis=1).tolist())
        assert step_SA==a2e_SA,"action result and action 2 exe does not match"
        print ("Sanity check all action to execute has result")

    def load_account_detail(self,DateI):
        fnwp_account_detail=self.iFH.get_account_detail_fnwp()
        if not os.path.exists(fnwp_account_detail):
            df_account_detail=pd.DataFrame(columns=self.AccountDetail_titles)
            df_account_detail.loc[len(df_account_detail)]=[DateI,self.total_invest,0.0]
        else:
            df_account_detail=pd.read_csv(fnwp_account_detail)
        df_account_detail = df_account_detail.astype(dtype=self.AccountDetail_types)
        df_account_detail.set_index(["DateI"], drop=True, inplace=True)
        account_day_inform = df_account_detail.iloc[-1]
        assert account_day_inform.name == DateI, "the last record of {0} is not for {1} {2}".\
            format(fnwp_account_detail, DateI,account_day_inform)
        Cash_afterclosing,MarketValue_afterclosing = \
            account_day_inform["Cash_after_closing"],account_day_inform["MarketValue_after_closing"]
        print ("Loaded account detail before {0} Cash_afterclosing {1} Market value {2} from {3}".format
               (account_day_inform.name,Cash_afterclosing,MarketValue_afterclosing, fnwp_account_detail))
        return df_account_detail,Cash_afterclosing,MarketValue_afterclosing

    def save_df_account(self, df_account, DateI):
        fnwp_holding = self.iFH.get_account_fnwp()
        fnwp_holding_backup = self.iFH.get_account_backup_fnwp(DateI)
        df_account.to_csv(fnwp_holding)   #here need to save index, it is stock
        df_account.to_csv(fnwp_holding_backup) #here need to save index, it is stock
        print ("Saved account to {0} and backup in {1}".format(fnwp_holding,fnwp_holding_backup))

    def update_save_account_detail(self,df_account_detail, DateI,Cash_afterclosing,MarketValue_afterclosing):
        self.df_account_detail.loc[DateI]=[Cash_afterclosing,MarketValue_afterclosing]
        fnwp_account_detail=self.iFH.get_account_detail_fnwp()
        fnwp_account_detail_backup = self.iFH.get_account_detail_backup_fnwp(DateI)
        df_account_detail.to_csv(fnwp_account_detail) #here need to save index, it is DateI
        df_account_detail.to_csv(fnwp_account_detail_backup) #here need to save index, it is DateI
        print("Saved account detail {0} Cash after closing {1} Market value after closing {2} to {3} bakcup to {4}".
              format(DateI, Cash_afterclosing, MarketValue_afterclosing, fnwp_account_detail,fnwp_account_detail_backup))

    def get_next_day_action(self,DateI,mumber_of_stock_could_buy,l_holding, L_Eval_Profit_low_flag):
        assert self.rlc.CLN_AV_Handler=="AV_Handler_AV1"
        raw_av=np.array(self.i_cav.Fresh_Raw_AV()).reshape(1, -1)  #as raw_av all is 0, get_OB_AV in predict will get 0
        l_av=[]
        l_lv=[]
        l_sv=[]
        for stock in self.df_account.index:
            lv,sv,ref=self.i_get_data.read_1day_TP_Data(stock, DateI)
            l_lv.append(lv)
            l_sv.append(sv)
            l_av.append(raw_av)  # not final sate s_ all av should be 0
        stacked_state = [np.concatenate(l_lv, axis=0), np.concatenate(l_sv, axis=0), np.concatenate(l_av, axis=0)]
        print ("Get lv sv from {0} and Generate av by {1}".format(self.rlc.data_name,self.rlc.CLN_AV_Handler))

        l_a_OB, l_a_OS = self.i_eb.V3_choose_action_CC(stacked_state,calledby="")
        l_a, l_ADlog = getattr(self.i_buystrategy, self.strategy_fun)(DateI, mumber_of_stock_could_buy, l_a_OB, l_a_OS,
                                                             l_holding, L_Eval_Profit_low_flag)
        print ("Get action2exe by predict model {0} ET{1}and buy strategy {2}".
               format(self.RL_system_name, self.RL_Model_ET,self.strategy_fun))
        return l_a,l_ADlog

    def save_next_day_action(self, DateI, l_a,stocklist,df_account):
        a2e_fnwp=self.iFH.get_a2e_fnwp(DateI)
        df_e2a=pd.DataFrame(columns=self.a2e_Titles)

        df_e2a=df_e2a.astype(self.a2e_types)
        df_e2a.set_index(["Stock"], drop=True, inplace=True)
        for action, stock in zip(l_a,stocklist):
            if action ==0:
                df_e2a.at[stock, "Action"] = "Buy"
                df_e2a.at[stock, "Gu"] = 0
            elif action ==2:
                df_e2a.at[stock, "Action"] = "Sell"
                df_e2a.at[stock, "Gu"]= df_account.loc[stock]["Holding_Gu"]
        df_e2a.to_csv(a2e_fnwp)
        print ("Save action2exe to {0}".format(a2e_fnwp))
        return df_e2a

    def Clean_account_step_inform(self,df_account ):
        for title, default in zip (self.Account_Step_Items_titles,self.Account_Step_Items_default):
            df_account[title]=default

    def Clean_account_stock_holding_inform(self, df_account, stock):
        for title, default in zip (self.Account_Holding_Items_titles,self.Account_Holding_Items_default):
            df_account.loc[stock][title]=default

    def update_holding_with_action_result(self,df_account,df_aresult, stock, DateI):
        if df_aresult.loc[stock]["Action"]=="Buy":
            if df_aresult.loc[stock]["ActionResult"]!="Success":
                ErrorMessage="Fail to buy due to {0}".format(df_aresult.loc[stock]["ActionResult"])
                print (stock,"",ErrorMessage)
                return ErrorMessage
            else:
                df_account.at[stock,"Buy_Invest"] = df_aresult.loc[stock]["Buy_Invest"]
                df_account.at[stock,"Buy_NPrice"] = df_aresult.loc[stock]["Buy_NPrice"]

                current_holding_gu = self.i_hfq_tb.get_update_volume_on_hfq_ratio_change(
                    old_hfq_ratio=df_account.loc[stock]["Holding_HRatio"],
                    new_hfq_ratio=df_aresult.loc[stock]["Buy_HRatio"],
                    old_volume=df_account.loc[stock]["Holding_Gu"])
                df_account.at[stock,"Holding_Gu"]      = current_holding_gu + df_aresult.loc[stock]["Buy_Gu"]
                df_account.at[stock,"Holding_Invest"] += df_aresult.loc[stock]["Buy_Invest"]
                df_account.at[stock,"Buy_Times"]      += 1
                df_account.at[stock,"Holding_HRatio"]  = df_aresult.loc[stock]["Buy_HRatio"]
                df_account.at[stock,"Holding_NPrice"]  = df_aresult.loc[stock]["Buy_NPrice"]
                df_account.at[stock,"Action"]          = df_aresult.loc[stock]["Action"]
                df_account.at[stock,"HoldingStartDateI"] = DateI
                return "SuccessBought"
        elif df_aresult.loc[stock]["Action"]=="Sell":
            if df_aresult.loc[stock]["ActionResult"]!=("Success"):
                ErrorMessage="Fail to sell due to {0} which bought at {1}".\
                    format(stock,df_aresult.loc[stock]["ActionResult"], df_account.loc[stock]["HoldingStartDateI"])
                print (stock,"",ErrorMessage)
                return ErrorMessage
            else:
                df_account.at[stock,"TransIDI"] += 1 #Todo double checkthis move from reset to successful seel
                df_account.at[stock,"Sell_Return"] = df_aresult.loc[stock]["Sell_Return"]
                df_account.at[stock,"Sell_Earn"]   = df_aresult.loc[stock]["Sell_Return"]-\
                                                     df_account.loc[stock]["Holding_Invest"]
                df_account.at[stock,"Sell_NPrice"] = df_aresult.loc[stock]["Sell_NPrice"]
                df_account.at[stock,"Action"] =      df_aresult.loc[stock]["Action"]
                self.Clean_account_stock_holding_inform(df_account, stock)
                return "SuccessSold"+"Earn" if df_account.at[stock, "Sell_Earn"]>0 else "Balance" \
                    if df_account.at[stock, "Sell_Earn"]==0 else "Loss"
        else:
            assert False, "Action only can be Buy or Sell {0} {1}".format(df_aresult,stock)

    def prepare_report(self, DateI,logs, df_e2a):
        Cash_afterclosing,MarketValue_afterclosing,mumber_of_stock_could_buy,\
            l_log_bought, l_log_Earnsold, l_log_balancesold, l_log_Losssold,\
            l_log_fail_action, l_log_holding_with_no_action,\
            l_ADlog=logs
        fnwp=self.iFH.get_report_fnwp(DateI)
        with open(fnwp,"w") as f:
            f.write("Date: {0} Cash After Closing: {1} Market Value: {2}\n Num of Stock could bought tomorrow: {3}\n".
                    format(DateI, Cash_afterclosing, MarketValue_afterclosing ,mumber_of_stock_could_buy))
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
                f.write("    {0} {1}\n".format(log_fail[0],log_fail[1]))
            f.write("\n") # For look better
            f.write("Today Error holding without sell action {0} :\n".format(len(l_log_holding_with_no_action)))
            for holding_with_no_action in l_log_holding_with_no_action:
                f.write("    {0} {1}\n".format(holding_with_no_action[0],holding_with_no_action[1]))
            f.write("\n")  # For look better
            f.write("Tommorow to Buy {0}\n".format(len(df_e2a[df_e2a["Action"]=="Buy"])))
            f.write("    {0}\n".format(",".join(df_e2a[df_e2a["Action"]=="Buy"].index.tolist())))
            f.write("Tommorow to Sell {0}\n".format(len(df_e2a[self.df_e2a["Action"]=="Sell"])))
            f.write("    {0}\n".format(",".join(df_e2a[df_e2a["Action"]=="Sell"].index.tolist())))

            assert l_ADlog[0]==DateI
            l_not_buy_due_limit=l_ADlog[1].split("_")
            if l_not_buy_due_limit==['']:
                f.write("Tommorow not_buy_due_limit {0}\n".format(0))
                f.write("\n")  # For look better
            else:
                f.write("Tommorow not_buy_due_limit {0}\n".format(len(l_not_buy_due_limit)))
                for sidx in l_not_buy_due_limit:
                    f.write("    {0}\n".format(self.sl[int(sidx)]))
            l_multibuy=l_ADlog[4].split("_")
            if l_multibuy==['']:
                f.write("Tommorow multibuy(not sell due to multibuy) {0}\n".format(0))
                f.write("\n")  # For look better
            else:
                f.write("Tommorow multibuy(not sell due to multibuy) {0}\n".format(len(l_multibuy)))
                for sidx in l_multibuy:
                    f.write("    {0}\n".format(self.sl[int(sidx)]))
        print ("{0} Report stored at {1}\n".format(DateI, fnwp))

    def Init_df_account(self, stock_list):
        fnwp_account = self.iFH.get_account_fnwp()
        df_account=pd.DataFrame(columns=self.account_Titles)
        for stock in stock_list:
            row=list(self.account_default)
            row[0]=stock
            df_account.loc[len(df_account)]=row
        df_account = df_account.astype(dtype=self.account_types)
        df_account.set_index(["Stock"],drop=True,inplace=True)
        df_account.to_csv(fnwp_account)
        print ("Init the account at {0}".format(fnwp_account))
        return df_account

    def start_strategy(self,DateI):
        self.sl=self.load_stock_list()
        self.i_eb=self.load_predict_model()
        self.df_account = self.Init_df_account(self.sl)
        self.df_account_detail, self.Cash_afterclosing, self.MarketValue_afterclosing = self.load_account_detail(DateI)
        mumber_of_stock_could_buy = int(self.Cash_afterclosing//self.min_invest)

        l_holding=[True if Holding_Gu!=0 else False for Holding_Gu in self.df_account["Holding_Gu"].tolist()]
        assert self.strategy_fun=="Buy_Strategy_multi_time_Direct_sell"
        L_Eval_Profit_low_flag=[False for _ in self.sl]  # not used in Buy_Strategy_multi_time_Direct_sell

        self.save_df_account(self.df_account, DateI)
        self.update_save_account_detail(self.df_account_detail, DateI, self.Cash_afterclosing,
                                        self.MarketValue_afterclosing)

        l_a,l_ADlog=self.get_next_day_action(DateI, mumber_of_stock_could_buy, l_holding, L_Eval_Profit_low_flag)

        self.df_e2a=self.save_next_day_action(DateI, l_a, self.sl, self.df_account)

        logs=[self.Cash_afterclosing,self.MarketValue_afterclosing,mumber_of_stock_could_buy,[], [], [], [], [], [], l_ADlog]
        self.prepare_report(DateI,logs, self.df_e2a)

    def run_strategy(self, DateI):
        self.sl = self.load_stock_list()
        self.i_eb = self.load_predict_model()
        self.df_account = self.load_df_account()
        self.df_a2eDone = self.load_df_a2eDone(DateI)
        self.df_aresult = self.load_df_aresult(DateI)

        self.Check_all_actionDone_has_result(self.df_a2eDone,self.df_aresult)

        self.df_account_detail, self.Cash_afterclosing, self.MarketValue_afterclosing= self.load_account_detail(DateI)

        ##update holding
        l_log_bought,l_log_Earnsold,l_log_balancesold,l_log_Losssold,l_log_fail_action,l_log_holding_with_no_action=[],[],[],[],[],[]
        for stock in self.sl:
            if stock in self.df_aresult.index:
                message =self.update_holding_with_action_result(self.df_account, self.df_aresult,stock,DateI)
                if not message.startswith("Success"):
                    l_log_fail_action.append([stock,message])
                else:
                    if message.endswith("Bought"):
                        l_log_bought.append(stock)
                    else:
                        if message.endswith("Earn"):
                            l_log_Earnsold.append(stock)
                        elif message.endswith("Loss"):
                            l_log_Losssold.append(stock)
                        else:
                            l_log_balancesold.append(stock)
            else:
                if self.df_account.loc[stock]["Holding_Gu"]!=0:
                    l_log_holding_with_no_action.append([stock, "{0} start holding with no action".
                                                        format(self.df_account.loc[stock]["HoldingStartDateI"])])

        print("Updated account holding information")

        self.Clean_account_step_inform(self.df_account)
        print ("Cleared account step information")

        #是不是 就用 "Holding_Invest" 来做 market value？ #todo change EVal_CC 's way?
        self.Cash_afterclosing +=self.df_aresult["Sell_Return"].sum()-self.df_aresult["Buy_Invest"].sum()
        self.MarketValue_afterclosing=self.df_account["Holding_Invest"].sum()
        mumber_of_stock_could_buy = int(self.Cash_afterclosing//self.min_invest)

        l_holding=[True if Holding_Gu!=0 else False for Holding_Gu in self.df_account["Holding_Gu"].tolist()]
        assert self.strategy_fun=="Buy_Strategy_multi_time_Direct_sell"
        L_Eval_Profit_low_flag=[False for _ in self.sl]  # not used in Buy_Strategy_multi_time_Direct_sell

        self.save_df_account(self.df_account, DateI)
        self.update_save_account_detail(self.df_account_detail, DateI, self.Cash_afterclosing,
                                        self.MarketValue_afterclosing)

        l_a,l_ADlog=self.get_next_day_action(DateI, mumber_of_stock_could_buy, l_holding, L_Eval_Profit_low_flag)

        self.df_e2a=self.save_next_day_action(DateI, l_a, self.sl, self.df_account)

        logs=[self.Cash_afterclosing,self.MarketValue_afterclosing,mumber_of_stock_could_buy,
              l_log_bought, l_log_Earnsold, l_log_balancesold, l_log_Losssold,
              l_log_fail_action, l_log_holding_with_no_action,
              l_ADlog]
        self.prepare_report(DateI,logs, self.df_e2a)