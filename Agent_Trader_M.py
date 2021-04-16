from Agent_Trader_MBase import *
from itertools import accumulate
class Experiment(Strategy_agent_base,Process):
    #self.isb=Strategy_agent_base(portfolio_name, strategy_name,experiment_name,experiment_config_params)
    def __init__(self, portfolio_name, strategy_name,experiment_name,experiment_config_params,process_idx,LL_Agent2GPU,LL_GPU2Agent, E_Stop):
        Process.__init__(self)
        Strategy_agent_base.__init__(self,portfolio_name, strategy_name,experiment_name,experiment_config_params)
        self.process_idx,self.LL_Agent2GPU,self.LL_GPU2Agent, self.E_Stop=process_idx,LL_Agent2GPU,LL_GPU2Agent,E_Stop
        self.l_i_get_data= [DBTP_Reader.DBTP_Reader(self.isc.rlcs[idx].data_name if row["TPDB_Name"] == "" else row["TPDB_Name"])
                            for idx,row in self.isc.dfsc.iterrows()]   #no logic, only raw read data
        assert all([rlc.CLN_AV_Handler=="AV_Handler_AV1" for rlc in self.isc.rlcs])
        self.i_cav = globals()["AV_Handler_AV1"](self.isc.rlcs[0])
        assert all([row["strategy_fun"] == "Buy_Strategy_multi_time_Direct_sell" for _, row in self.isc.dfsc.iterrows()])
        self.i_buystrategy = Buy_Strategies(self.isc.rlcs[0])
        self.i_sim=strategy_sim(self.isc.Strategy_dir, experiment_name,self.isc.NumModel,self.a2e_types,self.aresult_Titles)
    def get_next_day_action(self,DateI,l_df_account, mumber_of_stock_could_buy,ll_holding, ll_Eval_Profit_low_flag):
        ll_a,ll_ADlog=[],[]
        for Model_idx in list(range(self.isc.NumModel)):
            raw_av=np.array(self.i_cav.Fresh_Raw_AV()).reshape(1, -1)  #as raw_av all is 0, get_OB_AV in predict will get 0
            l_av,l_lv,l_sv=[],[],[]
            for stock in l_df_account[Model_idx].index:
                lv,sv,ref=self.l_i_get_data[Model_idx].read_1day_TP_Data(stock, DateI)
                l_lv.append(lv)
                l_sv.append(sv)
                l_av.append(raw_av)  # not final sate s_ all av should be 0
            stacked_state = [np.concatenate(l_lv, axis=0), np.concatenate(l_sv, axis=0), np.concatenate(l_av, axis=0)]

            self.LL_Agent2GPU[Model_idx].append([self.process_idx, stacked_state])
            while len(self.LL_GPU2Agent[Model_idx])==0:
                time.sleep(0.1)
            l_a_OB, l_a_OS=self.LL_GPU2Agent[Model_idx].pop()
            for idx in list(range(len(l_df_account[Model_idx]))): #this is handle not IPO stock
                if not l_df_account[Model_idx].iloc[idx]["Tradable_flag"]:
                    l_a_OB[idx]=1
                    l_a_OS[idx]=3
            l_a, l_ADlog = getattr(self.i_buystrategy, self.isc.dfsc.loc[Model_idx]["strategy_fun"])(
                DateI, mumber_of_stock_could_buy, l_a_OB, l_a_OS,ll_holding[Model_idx], ll_Eval_Profit_low_flag[Model_idx])
            ll_a.append(l_a)
            ll_ADlog.append(l_ADlog)
        return ll_a,ll_ADlog

    def update_holding_with_action_result(self,df_account,df_aresult, stock, DateI):
        if not (stock in df_aresult.index):
            assert False, "Unexpected situation {0} doest not have {1}".format(df_aresult.index,stock)
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

    def start_experiement(self,DateI):
        self.remove_old_experiment_data()
        l_sl=self.init_stock_list()
        l_df_account = self.Init_df_account(l_sl)
        df_account_detail =self.Init_df_account_detail()
        Cash_afterclosing=self.iec.total_invest
        l_MarketValue_afterclosing=[0 for _ in list(range(self.isc.NumModel))]
        cash_to_use=Cash_afterclosing/2
        mumber_of_stock_could_buy = int(cash_to_use // self.iec.min_invest)

        for Model_idx,sl in enumerate(l_sl):
            for stock in sl:
                flag, datas, mess=self.get_AfterClosing_Nprice_HFQRatio(stock,DateI)
                assert flag, "the HFQ informat for {0} {1} does not exsts. error message:{1}".format(stock, DateI, mess)
                l_df_account[Model_idx].at[stock, "AfterClosing_NPrice"] =datas[0]
                l_df_account[Model_idx].at[stock, "AfterClosing_HFQRatio"]=datas[1]
                l_df_account[Model_idx].at[stock, "Tradable_flag"] =datas[2]

        ll_holding=[[True if Holding_Gu!=0 else False for Holding_Gu in df_account["Holding_Gu"].tolist()]for df_account in l_df_account]
        assert all ([row["strategy_fun"]=="Buy_Strategy_multi_time_Direct_sell" for Model_idx,row in self.isc.dfsc.iterrows() ])
        ll_Eval_Profit_low_flag=[[False for _ in sl] for sl in l_sl]  # not used in Buy_Strategy_multi_time_Direct_sell

        self.save_df_account(l_df_account, DateI)
        self.update_save_account_detail(df_account_detail, DateI, Cash_afterclosing,l_MarketValue_afterclosing)

        ll_a,ll_ADlog=self.get_next_day_action(DateI, l_df_account, mumber_of_stock_could_buy, ll_holding, ll_Eval_Profit_low_flag)
        adj_ll_a = self.Filter_ll_a(mumber_of_stock_could_buy, ll_a)
        l_df_e2a=self.save_next_day_action(DateI, adj_ll_a, l_sl, l_df_account)
        fake_ll=[[] for _ in list(range(self.isc.NumModel))]
        logs=[Cash_afterclosing,l_MarketValue_afterclosing,mumber_of_stock_could_buy,fake_ll, fake_ll, fake_ll, fake_ll, fake_ll, fake_ll, ll_ADlog]
        self.prepare_report(DateI,logs, l_df_e2a,l_sl)
        machine_report_fnwp=self.iFH.get_machine_report_fnwp(DateI)
        pickle.dump(logs+[ll_a,adj_ll_a], open(machine_report_fnwp, 'wb'))

    def run_experiement(self, YesterDayI,DateI):
        l_sl = self.load_stock_list()
        l_df_account = self.load_df_account()
        l_df_a2eDone = self.load_df_a2eDone(DateI)
        l_df_aresult = self.load_df_aresult(DateI)

        self.Check_all_actionDone_has_result(l_df_a2eDone,l_df_aresult)

        df_account_detail, Cash_afterclosing, l_MarketValue_afterclosing= self.load_account_detail(YesterDayI)

        ##update holding
        ll_log_bought,ll_log_Earnsold,ll_log_balancesold,ll_log_Losssold,ll_log_fail_action,ll_log_holding_with_no_action\
            =[],[],[],[],[],[]
        for Model_idx,sl in enumerate(l_sl):
            l_log_bought, l_log_Earnsold, l_log_balancesold, l_log_Losssold, l_log_fail_action, l_log_holding_with_no_action\
                = [], [], [], [], [], []
            for stock in sl:
                flag, datas, mess=self.get_AfterClosing_Nprice_HFQRatio(stock,DateI)
                assert flag, "the HFQ informat for {0} {1} does not exsts. error message:{1}".format(stock, DateI, mess)
                assert datas[0]!=0 or not datas[2],"{0} {1} {2}".format(stock, DateI, datas)
                l_df_account[Model_idx].at[stock, "AfterClosing_NPrice"]  = datas[0]
                l_df_account[Model_idx].at[stock, "AfterClosing_HFQRatio"]= datas[1]
                l_df_account[Model_idx].at[stock, "Tradable_flag"] = datas[2]
                if stock in l_df_aresult[Model_idx].index:
                    mess =self.update_holding_with_action_result(l_df_account[Model_idx], l_df_aresult[Model_idx],stock,DateI)
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
                    if l_df_account[Model_idx].loc[stock]["Holding_Gu"]!=0:
                        l_log_holding_with_no_action.append([stock, "{0} start holding with no action".
                                                            format(l_df_account[Model_idx].loc[stock]["HoldingStartDateI"])])

                self.Clean_account_step_inform(l_df_account[Model_idx],stock)
            ll_log_bought.append(l_log_bought)
            ll_log_Earnsold.append(l_log_Earnsold)
            ll_log_balancesold.append(l_log_balancesold)
            ll_log_Losssold.append(l_log_Losssold)
            ll_log_fail_action.append(l_log_fail_action)
            ll_log_holding_with_no_action.append(l_log_holding_with_no_action)
        #是不是 就用 "Holding_Invest" 来做 market value？
        Cash_afterclosing +=sum([df_aresult["Sell_Return"].sum()-df_aresult["Buy_Invest"].sum() for df_aresult in l_df_aresult])
        l_MarketValue_afterclosing=[df_account["Holding_Invest"].sum() for df_account in l_df_account]
        #mumber_of_stock_could_buy = int(Cash_afterclosing//self.min_invest)
        cash_to_use=Cash_afterclosing if Cash_afterclosing<self.iec.total_invest/2 else min((Cash_afterclosing+ sum(l_MarketValue_afterclosing))/2,Cash_afterclosing)
        mumber_of_stock_could_buy = int(cash_to_use // self.iec.min_invest)
        self.save_df_account(l_df_account, DateI)
        self.update_save_account_detail(df_account_detail, DateI, Cash_afterclosing,l_MarketValue_afterclosing)
        print ("After_Closing, Cash:{0:.2f} ".format(Cash_afterclosing), end=" ")
        for idx,MarketValue_afterclosing in enumerate(l_MarketValue_afterclosing):
            print ("Model{0} Market Value:{1:.2f} ".format(idx, MarketValue_afterclosing), end=" ")
        print("total {0:.2f}".format(Cash_afterclosing+sum(l_MarketValue_afterclosing)))

        ll_holding=[[True if Holding_Gu!=0 else False for Holding_Gu in df_account["Holding_Gu"].tolist()] for df_account in l_df_account]
        assert all([row["strategy_fun"]=="Buy_Strategy_multi_time_Direct_sell" for _, row in self.isc.dfsc.iterrows()])
        ll_Eval_Profit_low_flag=[[False for _ in sl] for sl in l_sl]  # not used in Buy_Strategy_multi_time_Direct_sell

        ll_a,ll_ADlog=self.get_next_day_action(DateI, l_df_account,mumber_of_stock_could_buy, ll_holding, ll_Eval_Profit_low_flag)
        adj_ll_a=self.Filter_ll_a(mumber_of_stock_could_buy,ll_a)
        l_df_e2a=self.save_next_day_action(DateI, adj_ll_a, l_sl, l_df_account)

        logs=[Cash_afterclosing,l_MarketValue_afterclosing,mumber_of_stock_could_buy,
              ll_log_bought, ll_log_Earnsold, ll_log_balancesold, ll_log_Losssold,
              ll_log_fail_action, ll_log_holding_with_no_action,
              ll_ADlog]
        self.prepare_report(DateI,logs, l_df_e2a,l_sl)
        machine_report_fnwp=self.iFH.get_machine_report_fnwp(DateI)
        pickle.dump(logs+[ll_a,adj_ll_a], open(machine_report_fnwp, 'wb'))

    def Sell_All(self,DateI):
        l_sl = self.load_stock_list()
        l_df_account = self.load_df_account()
        ll_a= [[2 if l_df_account[Model_idx].loc[stock]["Holding_Gu"] != 0 else  3  for stock in l_sl[Model_idx]] for Model_idx in list(range(len(l_sl)))]
        df_e2a = self.save_next_day_action(DateI, ll_a, l_sl, l_df_account)

    def Filter_ll_a(self,mumber_of_stock_could_buy, ll_a):
        InOnes = []
        for l_a in ll_a:
            InOnes = InOnes + l_a
        pidxs = [idx for idx, item in enumerate(InOnes) if item == 0]
        total_Cbuy = len(pidxs)

        if mumber_of_stock_could_buy >= total_Cbuy:
            return ll_a
        else:
            Num_remove_from_buy = total_Cbuy - mumber_of_stock_could_buy
            random.shuffle(pidxs)
            for selected_pidx in pidxs[:Num_remove_from_buy]:
                assert InOnes[selected_pidx] == 0
                InOnes[selected_pidx] = 1

            adj_ll_a = []
            pstartidx = 0
            for l_a in ll_a:
                adj_ll_a.append(InOnes[pstartidx:pstartidx + len(l_a)])
                pstartidx += len(l_a)
            return adj_ll_a

    def run(self):
        from contextlib import redirect_stdout, redirect_stderr
        AStart_idx, AStartI = self.get_closest_TD(self.iec.StartI, True)
        AEnd_idx, AEndI = self.get_closest_TD(self.iec.EndI, False)
        assert AStartI <= AEndI
        if self.iec.flag_Print_on_screen_or_file:
            newstdout = sys.__stdout__
            newstderr = sys.__stderr__
            stdoutfnwp,stderrfnwp="",""
        else:
            stdoutfnwp=os.path.join(self.iec.Experiment_dir,"Output.txt")
            stderrfnwp=os.path.join(self.iec.Experiment_dir,"Error.txt")
            print ("Output will be direct to {0}".format(stdoutfnwp))
            print ("Error will be direct to {0}".format(stderrfnwp))
            newstdout = open(stdoutfnwp, "w")
            newstderr = open(stderrfnwp, "w")
        with redirect_stdout(newstdout), redirect_stderr(newstderr):
            self.start_experiement(AStartI)
            print ("Init strategy at ", AStartI)
            YesterdayI = AStartI
            period = self.nptd[AStart_idx + 1:AEnd_idx+1]
            for DateI in period:
                print("Run strategy at ", DateI)
                if DateI==AEndI:
                    self.Sell_All(YesterdayI)
                self.i_sim.sim(YesterdayI, DateI)
                self.run_experiement(YesterdayI,DateI)
                YesterdayI = DateI
                newstdout.flush()
                newstderr.flush()
            self.E_Stop.set()

def One_batch_experiment(portfolio,dfbatch):

    Strategies=list(set(dfbatch["strategy"].to_list()))
    assert len(Strategies)==1
    iStrategy = Strategy_Config(portfolio, Strategies[0])
    iStrategy.initialization_strategy()

    iManager = Manager()
    ll_Agent2GPU = [iManager.list() for _ in list(range(iStrategy.NumModel))]
    lll_GPU2Agent = [[iManager.list() for _ in range(len(dfbatch))] for _ in list(range(iStrategy.NumModel))]
    l_E_Stop_GPUs= [iManager.Event() for _ in list(range(iStrategy.NumModel))]

    iTrader_GPUs=[]
    for Modle_idx,row in iStrategy.dfsc.iterrows():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(row["GPU_idx"])
        #iTrader_GPU = Trader_GPU(Strategy_Config(portfolio,Strategies[0]),l_E_Stop_GPUs[Modle_idx], ll_Agent2GPU[Modle_idx], lll_GPU2Agent[Modle_idx])
        iTrader_GPU = Trader_GPU(portfolio,Strategies[0], Modle_idx,l_E_Stop_GPUs[Modle_idx],
                                 ll_Agent2GPU[Modle_idx], lll_GPU2Agent[Modle_idx])
        iTrader_GPU.daemon = True
        iTrader_GPU.start()
        iTrader_GPUs.append(iTrader_GPU)

    L_Experiement_Stop_Events = [iManager.Event() for _ in list(range(len(dfbatch)))]
    iProcesses = []
    for experiement_idx, row in dfbatch.iterrows():

        experiment_config_params=row["total_invest"], row["min_invest"], row["StartI"], row["EndI"], row["flag_Print_on_screen_or_file"]
        iProcess = Experiment(portfolio, row["strategy"], row["experiment"], experiment_config_params,
                            experiement_idx,ll_Agent2GPU,
                            [lll_GPU2Agent[Modle_idx][experiement_idx] for Modle_idx in list(range(iStrategy.NumModel))],
                            L_Experiement_Stop_Events[experiement_idx])
        iProcess.daemon = True
        iProcess.start()
        iProcesses.append(iProcess)


    while not all([E.is_set() for E in L_Experiement_Stop_Events]):
        time.sleep(10)
    for iP in iProcesses:
        iP.join()
    for E_Stop_GPU in l_E_Stop_GPUs:
        E_Stop_GPU.set()
    for iTrader_GPU in iTrader_GPUs:
        iTrader_GPU.join()

def main(argv):
    if sys.argv[1]=="Batch":
        portfolio = sys.argv[2]
        for batch_config_name in sys.argv[3:]:
            dfbatch = pd.read_csv(os.path.join("/home/rdchujf/n_workspace/AT", portfolio, "{0}.csv".format(batch_config_name)))
            One_batch_experiment(portfolio, dfbatch)
    elif len(sys.argv)==3:
        portfolio,batch_config_name = sys.argv[1],sys.argv[2]
        dfbatch = pd.read_csv(
            os.path.join("/home/rdchujf/n_workspace/AT", portfolio, "{0}.csv".format(batch_config_name)))
        One_batch_experiment(portfolio, dfbatch)
    elif len(sys.argv)==5:
        portfolio, strategy  ,experiment,GPU_idx= sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
        assert GPU_idx in ["0","1"]
        iec=Experiment_Config(portfolio,strategy,experiment,[])
        dfbatch=pd.DataFrame([strategy,experiment,GPU_idx,iec.total_invest,iec.min_invest,iec.StartI,iec.EndI,iec.flag_Print_on_screen_or_file],
            columns=["strategy","experiment","GPUI","total_invest","min_invest","StartI","EndI","flag_Print_on_screen_or_file"])
        One_batch_experiment(portfolio, dfbatch)
    else:
        print ("python Agent_Trader.py portfolio strategy experiment GPUI")
        print ("python Agent_Trader.py portfolio Config_name")
        print ("python Agent_Trader.py Batch portfolio Config_name1,Config_name2....")

if __name__ == '__main__':
    main(sys.argv)
