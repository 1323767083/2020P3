from Agent_Trader_base import *
class Strategy_agent(Strategy_agent_base,Process):
    def __init__(self, portfolio_name, strategy_name,experiment_name,experiment_config_params,process_idx,L_Agent2GPU,L_GPU2Agent, E_Stop):
        Strategy_agent_base.__init__(self,portfolio_name, strategy_name,experiment_name,experiment_config_params)
        Process.__init__(self)
        self.process_idx,self.L_Agent2GPU,self.L_GPU2Agent, self.E_Stop=process_idx,L_Agent2GPU,L_GPU2Agent,E_Stop
        self.i_get_data= DBTP_Reader.DBTP_Reader(self.rlc.data_name if self.TPDB_Name == "" else self.TPDB_Name)   #no logic, only raw read data
        self.i_cav = globals()[self.rlc.CLN_AV_Handler](self.rlc)
        self.i_buystrategy = Buy_Strategies(self.rlc)
        self.i_sim=strategy_sim(self.iFH,self.a2e_types,self.aresult_Titles)
    def get_next_day_action(self,DateI,df_account, mumber_of_stock_could_buy,l_holding, L_Eval_Profit_low_flag):
        assert self.rlc.CLN_AV_Handler=="AV_Handler_AV1"
        raw_av=np.array(self.i_cav.Fresh_Raw_AV()).reshape(1, -1)  #as raw_av all is 0, get_OB_AV in predict will get 0
        l_av=[]
        l_lv=[]
        l_sv=[]
        for stock in df_account.index:
            lv,sv,ref=self.i_get_data.read_1day_TP_Data(stock, DateI)
            l_lv.append(lv)
            l_sv.append(sv)
            l_av.append(raw_av)  # not final sate s_ all av should be 0
        stacked_state = [np.concatenate(l_lv, axis=0), np.concatenate(l_sv, axis=0), np.concatenate(l_av, axis=0)]

        self.L_Agent2GPU.append([self.process_idx, stacked_state])
        while len(self.L_GPU2Agent)==0:
            time.sleep(0.1)
        l_a_OB, l_a_OS=self.L_GPU2Agent.pop()

        #for idx, row in df_account.iterrows():
        for idx in list(range(len(df_account))):
            if not df_account.iloc[idx]["Tradable_flag"]: #this is handle not IPO stock
                l_a_OB[idx]=1
                l_a_OS[idx]=3
        l_a, l_ADlog = getattr(self.i_buystrategy, self.strategy_fun)(DateI, mumber_of_stock_could_buy, l_a_OB, l_a_OS,
                                                             l_holding, L_Eval_Profit_low_flag)
        return l_a,l_ADlog

    def update_holding_with_action_result(self,df_account,df_aresult, stock, DateI):
        if df_aresult.loc[stock]["Action"]=="Buy":
            if df_aresult.loc[stock]["Action_Result"]=="Tinpai":
                ErrorMessage="Fail to buy due to {0}".format(df_aresult.loc[stock]["Action_Result"])
                print (stock,"",ErrorMessage)
                return ErrorMessage
            else:
                df_account.at[stock,"Buy_Invest"] = df_aresult.loc[stock]["Buy_Invest"]
                current_holding_gu = self.i_hfq_tb.get_update_volume_on_hfq_ratio_change(
                    old_hfq_ratio=df_account.loc[stock]["Holding_HRatio"],
                    new_hfq_ratio=df_account.loc[stock]["AfterClosing_HFQRatio"],
                    old_volume=df_account.loc[stock]["Holding_Gu"])
                df_account.at[stock,"Holding_Gu"]      = current_holding_gu + df_aresult.loc[stock]["Buy_Gu"]
                df_account.at[stock,"Holding_Invest"] += df_aresult.loc[stock]["Buy_Invest"]
                df_account.at[stock,"Buy_Times"]      += 1
                df_account.at[stock,"Holding_HRatio"]  = df_account.loc[stock]["AfterClosing_HFQRatio"]
                df_account.at[stock,"Action"]          = df_aresult.loc[stock]["Action"]
                df_account.at[stock,"HoldingStartDateI"] = DateI
                return "SuccessBought"
        elif df_aresult.loc[stock]["Action"]=="Sell":
            if df_aresult.loc[stock]["Action_Result"]=="Tinpai":
                ErrorMessage="Fail to sell due to {0} which bought at {1}".\
                    format(stock,df_aresult.loc[stock]["Action_Result"], df_account.loc[stock]["HoldingStartDateI"])
                print (stock,"",ErrorMessage)
                return ErrorMessage
            else:
                df_account.at[stock,"TransIDI"] += 1 #Todo double checkthis move from reset to successful seel
                df_account.at[stock,"Sell_Return"] = df_aresult.loc[stock]["Sell_Return"]
                df_account.at[stock,"Sell_Earn"]   = df_aresult.loc[stock]["Sell_Return"]-\
                                                     df_account.loc[stock]["Holding_Invest"]
                df_account.at[stock,"Action"] =      df_aresult.loc[stock]["Action"]
                self.Clean_account_stock_holding_inform(df_account, stock)
                mess="SuccessSoldEarn" if df_account.at[stock, "Sell_Earn"]>0 else "SuccessSoldBalance" \
                    if df_account.at[stock, "Sell_Earn"]==0 else "SuccessSoldLoss"
                return "{0}_{1:.2f}".format(mess,df_account.at[stock, "Sell_Earn"])
        else:
            assert False, "Action only can be Buy or Sell {0} {1}".format(df_aresult,stock)

    def Clean_account_step_inform(self,df_account ,stock):
        for title, default in zip (self.Account_Step_Items_titles,self.Account_Step_Items_default):
            df_account.at[stock,title]=default

    def Clean_account_stock_holding_inform(self, df_account, stock):
        for title, default in zip (self.Account_Holding_Items_titles,self.Account_Holding_Items_default):
            df_account.at[stock,title]=default

    def remove_old_experiment_data(self):
        for sub_item in os.listdir(self.iFH.AT_account_dir):
            if sub_item in ["config.json","Output.txt","Error.txt"]: continue
            sub_itemwp=os.path.join(self.iFH.AT_account_dir,sub_item)
            if os.path.isdir(sub_itemwp):
                dir2remove=os.path.join(self.iFH.AT_account_dir, sub_itemwp)
                shutil.rmtree(dir2remove)
                print ("removed directory {0}".format(dir2remove))
            elif os.path.isfile(sub_itemwp):
                os.remove(sub_itemwp)
                print("removed file {0}".format(sub_itemwp))
            else:
                assert False, "{0} is neither a file or a directory".format(sub_itemwp)

    def start_strategy(self,DateI):
        self.remove_old_experiment_data()
        sl=self.init_stock_list()
        df_account = self.Init_df_account(sl)
        df_account_detail =self.Init_df_account_detail()
        Cash_afterclosing=self.total_invest
        MarketValue_afterclosing=0
        cash_to_use=Cash_afterclosing if Cash_afterclosing<self.total_invest/2 else min((Cash_afterclosing+ MarketValue_afterclosing)/2,Cash_afterclosing)
        mumber_of_stock_could_buy = int(cash_to_use // self.min_invest)

        #mumber_of_stock_could_buy = int(Cash_afterclosing//self.min_invest)
        for stock in sl:
            flag, datas, mess=self.get_AfterClosing_Nprice_HFQRatio(stock,DateI)
            assert flag, "the HFQ informat for {0} {1} does not exsts. error message:{1}".format(stock, DateI, mess)
            df_account.at[stock, "AfterClosing_NPrice"] =datas[0]
            df_account.at[stock, "AfterClosing_HFQRatio"]=datas[1]
            df_account.at[stock, "Tradable_flag"] =datas[2]
        l_holding=[True if Holding_Gu!=0 else False for Holding_Gu in df_account["Holding_Gu"].tolist()]
        assert self.strategy_fun=="Buy_Strategy_multi_time_Direct_sell"
        L_Eval_Profit_low_flag=[False for _ in sl]  # not used in Buy_Strategy_multi_time_Direct_sell

        self.save_df_account(df_account, DateI)
        self.update_save_account_detail(df_account_detail, DateI, Cash_afterclosing,MarketValue_afterclosing)

        l_a,l_ADlog=self.get_next_day_action(DateI, df_account, mumber_of_stock_could_buy, l_holding, L_Eval_Profit_low_flag)

        df_e2a=self.save_next_day_action(DateI, l_a, sl, df_account)

        logs=[Cash_afterclosing,MarketValue_afterclosing,mumber_of_stock_could_buy,[], [], [], [], [], [], l_ADlog]
        self.prepare_report(DateI,logs, df_e2a,sl)
        machine_report_fnwp=self.iFH.get_machine_report_fnwp(DateI)
        pickle.dump(logs+[l_a], open(machine_report_fnwp, 'wb'))

    def run_strategy(self, YesterDayI,DateI):
        sl = self.load_stock_list()
        df_account = self.load_df_account()
        df_a2eDone = self.load_df_a2eDone(DateI)
        df_aresult = self.load_df_aresult(DateI)

        self.Check_all_actionDone_has_result(df_a2eDone,df_aresult)

        df_account_detail, Cash_afterclosing, MarketValue_afterclosing= self.load_account_detail(YesterDayI)

        ##update holding
        l_log_bought,l_log_Earnsold,l_log_balancesold,l_log_Losssold,l_log_fail_action,l_log_holding_with_no_action=[],[],[],[],[],[]
        for stock in sl:
            flag, datas, mess=self.get_AfterClosing_Nprice_HFQRatio(stock,DateI)
            assert flag, "the HFQ informat for {0} {1} does not exsts. error message:{1}".format(stock, DateI, mess)
            assert datas[0]!=0 or not datas[2],"{0} {1} {2}".format(stock, DateI, datas)
            df_account.at[stock, "AfterClosing_NPrice"]  = datas[0]
            df_account.at[stock, "AfterClosing_HFQRatio"]= datas[1]
            df_account.at[stock, "Tradable_flag"] = datas[2]
            if stock in df_aresult.index:
                mess =self.update_holding_with_action_result(df_account, df_aresult,stock,DateI)
                if not mess.startswith("Success"):
                    l_log_fail_action.append([stock,mess])
                else:
                    if mess.endswith("Bought"):
                        l_log_bought.append(stock)
                    else:
                        mi=mess.split("_")
                        if mi[0].endswith("Earn"):
                            l_log_Earnsold.append(stock+":"+mi[1])
                        elif mi[0].endswith("Loss"):
                            l_log_Losssold.append(stock+":"+mi[1])
                        else:
                            l_log_balancesold.append(stock+":"+mi[1])
            else:
                if df_account.loc[stock]["Holding_Gu"]!=0:
                    l_log_holding_with_no_action.append([stock, "{0} start holding with no action".
                                                        format(df_account.loc[stock]["HoldingStartDateI"])])

            self.Clean_account_step_inform(df_account,stock)

        #是不是 就用 "Holding_Invest" 来做 market value？
        Cash_afterclosing +=df_aresult["Sell_Return"].sum()-df_aresult["Buy_Invest"].sum()
        MarketValue_afterclosing=df_account["Holding_Invest"].sum()
        #mumber_of_stock_could_buy = int(Cash_afterclosing//self.min_invest)
        cash_to_use=Cash_afterclosing if Cash_afterclosing<self.total_invest/2 else min((Cash_afterclosing+ MarketValue_afterclosing)/2,Cash_afterclosing)
        mumber_of_stock_could_buy = int(cash_to_use // self.min_invest)
        self.save_df_account(df_account, DateI)
        self.update_save_account_detail(df_account_detail, DateI, Cash_afterclosing,MarketValue_afterclosing)
        print ("After_Closing, Cash {0:.2f} Market Value {1:.2f}, total {2:.2f}".format(Cash_afterclosing,
                                            MarketValue_afterclosing,Cash_afterclosing+MarketValue_afterclosing))
        l_holding=[True if Holding_Gu!=0 else False for Holding_Gu in df_account["Holding_Gu"].tolist()]
        assert self.strategy_fun=="Buy_Strategy_multi_time_Direct_sell"
        L_Eval_Profit_low_flag=[False for _ in sl]  # not used in Buy_Strategy_multi_time_Direct_sell

        l_a,l_ADlog=self.get_next_day_action(DateI, df_account,mumber_of_stock_could_buy, l_holding, L_Eval_Profit_low_flag)

        df_e2a=self.save_next_day_action(DateI, l_a, sl, df_account)

        logs=[Cash_afterclosing,MarketValue_afterclosing,mumber_of_stock_could_buy,
              l_log_bought, l_log_Earnsold, l_log_balancesold, l_log_Losssold,
              l_log_fail_action, l_log_holding_with_no_action,
              l_ADlog]
        self.prepare_report(DateI,logs, df_e2a,sl)
        machine_report_fnwp=self.iFH.get_machine_report_fnwp(DateI)
        pickle.dump(logs+[l_a], open(machine_report_fnwp, 'wb'))


    def Sell_All(self,DateI):
        sl = self.load_stock_list()
        df_account = self.load_df_account()
        l_a= [2 if df_account.loc[stock]["Holding_Gu"] != 0 else  3  for stock in sl]
        df_e2a = self.save_next_day_action(DateI, l_a, sl, df_account)

    def run(self):
        from contextlib import redirect_stdout, redirect_stderr
        AStart_idx, AStartI = self.get_closest_TD(self.StartI, True)
        AEnd_idx, AEndI = self.get_closest_TD(self.EndI, False)
        assert AStartI <= AEndI
        if self.flag_Print_on_screen_or_file:
            newstdout = sys.__stdout__
            newstderr = sys.__stderr__
            stdoutfnwp,stderrfnwp="",""
        else:
            stdoutfnwp=os.path.join(self.Experiment_dir,"Output.txt")
            stderrfnwp=os.path.join(self.Experiment_dir,"Error.txt")
            print ("Output will be direct to {0}".format(stdoutfnwp))
            print ("Error will be direct to {0}".format(stderrfnwp))
            newstdout = open(stdoutfnwp, "w")
            newstderr = open(stderrfnwp, "w")
        with redirect_stdout(newstdout), redirect_stderr(newstderr):
            self.start_strategy(AStartI)
            print ("Init strategy at ", AStartI)
            YesterdayI = AStartI
            period = self.nptd[AStart_idx + 1:AEnd_idx+1]
            for DateI in period:
                print("Run strategy at ", DateI)
                if DateI==AEndI:
                    self.Sell_All(YesterdayI)
                self.i_sim.sim(YesterdayI, DateI)
                self.run_strategy(YesterdayI,DateI)
                YesterdayI = DateI
                newstdout.flush()
                newstderr.flush()
            self.E_Stop.set()

def One_batch_experiment(portfolio,config_name):
    df = pd.read_csv(os.path.join("/home/rdchujf/n_workspace/AT", portfolio, "{0}.csv".format(config_name)))
    iManager = Manager()
    L_Agent2GPU = iManager.list()
    LL_GPU2Agent = [iManager.list() for _ in range(len(df))]
    E_Stop_GPU=iManager.Event()

    Strategies=list(set(df["strategy"].to_list()))
    assert len(Strategies)==1
    GPUIs=list(set(df["GPUI"].to_list()))
    assert len(GPUIs) == 1

    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUIs[0])
    iTrader_GPU = Trader_GPU(Strategy_Config(portfolio,Strategies[0]),E_Stop_GPU, L_Agent2GPU, LL_GPU2Agent)
    iTrader_GPU.daemon = True
    iTrader_GPU.start()

    iProcesses = []
    Stop_Events = []
    for process_idx, row in df.iterrows():
        E_Stop = iManager.Event()
        experiment_config_params=row["total_invest"], row["min_invest"], row["StartI"], row["EndI"], row["flag_Print_on_screen_or_file"]
        iProcess = Strategy_agent(portfolio, row["strategy"], row["experiment"], experiment_config_params,
                                  process_idx,L_Agent2GPU, LL_GPU2Agent[process_idx],E_Stop)
        iProcess.daemon = True
        iProcess.start()
        iProcesses.append(iProcess)
        Stop_Events.append(E_Stop)

    while not all([E.is_set() for E in Stop_Events]):
        time.sleep(10)
    for iP in iProcesses:
        iP.join()
    E_Stop_GPU.set()
    iTrader_GPU.join()

def main(argv):
    if sys.argv[1]=="Batch":
        portfolio = sys.argv[2]
        for config_name in sys.argv[3:]:
            One_batch_experiment(portfolio, config_name)
    elif len(sys.argv)==5:
        portfolio   = sys.argv[1]
        strategy    = sys.argv[2]
        experiment  = sys.argv[3]
        GPU_idx     = sys.argv[4]
        assert GPU_idx in ["0","1"]
        iStrategy = Strategy_Config(portfolio, strategy)

        iManager = Manager()
        E_Stop = iManager.Event()
        iManager = Manager()
        L_E_Weight_Updated_Agent = [iManager.Event() for _ in range(1)]
        L_Agent2GPU = iManager.list()
        LL_GPU2Agent = [iManager.list() for _ in range(1)]
        E_Stop_GPU = iManager.Event()

        os.environ["CUDA_VISIBLE_DEVICES"] =GPU_idx
        iTrader_GPU = Trader_GPU(iStrategy, E_Stop_GPU, L_Agent2GPU, LL_GPU2Agent)
        iTrader_GPU.daemon = True
        iTrader_GPU.start()

        iProcess=Strategy_agent(portfolio, strategy, experiment,[], 0, L_Agent2GPU, LL_GPU2Agent[0], E_Stop)
        iProcess.daemon = True
        iProcess.start()

        while not E_Stop.is_set():
            time.sleep(60)
        iProcess.join()
        E_Stop_GPU.set()
        iTrader_GPU.join()

    elif len(sys.argv)==3:
        portfolio = sys.argv[1]
        config_name = sys.argv[2]
        One_batch_experiment(portfolio, config_name)
    else:
        print ("python Agent_Trader.py portfolio strategy experiment GPUI")
        print ("python Agent_Trader.py portfolio Config_name")
        print("python Agent_Trader.py Batch portfolio Config_name1,Config_name2....")

if __name__ == '__main__':
    main(sys.argv)
