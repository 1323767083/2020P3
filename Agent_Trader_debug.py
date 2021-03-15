from Agent_Trader import *
from multiprocessing import Process,Manager
import time

class debug_strategy(Strategy_agent,Process):
    def __init__(self, portfolio_name, strategy_name, experiment_name, E_Stop):
        Process.__init__(self)
        Strategy_agent.__init__(self, portfolio_name, strategy_name,experiment_name)
        self.E_Stop=E_Stop
        #debug
        self.i_RawData=RawData()

        #self.StartI, self.EndI=StartI, EndI
    def debug_load_df_a2e(self, DateI):
        fnwp_action2exe = self.iFH.get_a2e_fnwp(DateI)
        assert os.path.exists(fnwp_action2exe), "{0} does not exists".format(fnwp_action2exe)
        df_a2e = pd.read_csv(fnwp_action2exe)
        df_a2e = df_a2e.astype(self.a2e_types)
        df_a2e.set_index(["Stock"], drop=True, inplace=True,verify_integrity=True)
        print("Loaded a2e from ", fnwp_action2exe)
        return df_a2e

    def debug_sim(self, YesterdayI, DateI):
        df_a2e = self.debug_load_df_a2e(YesterdayI)
        df_a2e.to_csv(self.iFH.get_a2eDone_fnwp(DateI))
        df_aresult = pd.DataFrame(columns=self.aresult_Titles)
        # roughly buy 0.0003
        # roughly sell 0.0013
        for idx, row in df_a2e.iterrows():
            #print (row)
            stock, gu = row.name, row["Gu"]  # stock is index in Seris it is name
            flag, dfhqr, message = self.get_hfq_df(self.get_DBI_hfq_fnwp(stock))
            assert flag
            a = dfhqr[dfhqr["date"] == str(DateI)]
            if not a.empty:
                flag, dfqz, message = self.i_RawData.get_qz_df_inteface(row.name, DateI)
                assert flag
                for low, high in [[93000, 93500], [93500, 94000], [94000, 94500], [94500, 95000], [95000, 95500],
                                  [95500, 96000]]:
                    a = dfqz[(dfqz["Time"] >= 93000) & (dfqz["Time"] < 93500)]
                    if not a.empty: break
                num_trans = min(np.random.choice([1, 2, 3], p=[1 / 3, 1 / 3, 1 / 3]), len(a))
                NPrices = a["Price"].to_list()
                random.shuffle(NPrices)

                gu_avg = gu // num_trans
                l_Trans_Gu = [gu_avg if idx < num_trans - 1 else gu - (num_trans - 1) * gu_avg for idx in
                              list(range(num_trans))]
                l_Trans_Price = NPrices[:num_trans]
                if row["Action"] == "Buy":
                    for trans_Gu, trans_Price in zip(l_Trans_Gu, l_Trans_Price):
                        df_aresult.loc[len(df_aresult)] = [stock, "Buy", "Success", trans_Gu,trans_Gu * trans_Price * 1.0003, 0.0]
                elif row["Action"] == "Sell":
                    for trans_Gu, trans_Price in zip(l_Trans_Gu, l_Trans_Price):
                        df_aresult.loc[len(df_aresult)] = [stock, "Sell", "Success", 0, 0.0,trans_Gu * trans_Price * (1 - 0.0013)]
                else:
                    assert False, "Action only can by Buy or Sell not {0}".format(row["Action"])
            else:
                df_aresult.loc[len(df_aresult)] = [stock, row["Action"], "Tinpai", 0, 0.0, 0.0]
        df_aresult.to_csv(self.iFH.get_aresult_fnwp(DateI), index=False)
        return

    def run(self):
        from contextlib import redirect_stdout, redirect_stderr
        import tensorflow as tf
        from nets import init_virtual_GPU
        from nets import Explore_Brain
        AStart_idx, AStartI = self.get_closest_TD(self.StartI, True)
        AEnd_idx, AEndI = self.get_closest_TD(self.EndI, False)
        assert AStartI <= AEndI
        flag_Print_on_screen_or_file=False
        if flag_Print_on_screen_or_file:
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
            print ("Here")
            virtual_GPU =init_virtual_GPU(self.GPU_mem)

            with tf.device(virtual_GPU):
                i_eb = locals()[self.rlc.CLN_brain_explore](self.rlc)
                i_eb.load_weight(os.path.join(self.weight_fnwp))
                print("Loaded model from {0} ".format(self.weight_fnwp))

                self.start_strategy(i_eb,AStartI)
                print ("Init strategy at ", AStartI)
                YesterdayI = AStartI
                period = self.nptd[AStart_idx + 1:AEnd_idx+1]
                for DateI in period:
                    print("Run strategy at ", DateI)
                    if DateI==AEndI:
                        self.Sell_All(YesterdayI)
                    self.debug_sim(YesterdayI, DateI)
                    self.run_strategy(i_eb,YesterdayI,DateI)
                    YesterdayI = DateI
                    newstdout.flush()
                    newstderr.flush()
                self.E_Stop.set()
def One_batch_experiment(portfolio,config_name):
    df = pd.read_csv(os.path.join("/home/rdchujf/n_workspace/AT", portfolio, "{0}.csv".format(config_name)))
    iManager = Manager()
    iProcesses = []
    Stop_Events = []
    for _, row in df.iterrows():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        E_Stop = iManager.Event()
        iProcess = debug_strategy(portfolio, row["strategy"], row["experiment"], E_Stop)
        iProcess.daemon = True

        iProcess.start()
        iProcesses.append(iProcess)
        Stop_Events.append(E_Stop)
    while not all([E.is_set() for E in Stop_Events]):
        time.sleep(10)
    for iP in iProcesses:
        iP.join()


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
        os.environ["CUDA_VISIBLE_DEVICES"] =GPU_idx

        iManager = Manager()
        E_Stop = iManager.Event()

        iProcess=debug_strategy(portfolio, strategy, experiment,E_Stop)
        iProcess.daemon = True
        iProcess.start()

        while not E_Stop.is_set():
            time.sleep(60)
        iProcess.join()
    elif len(sys.argv)==3:
        portfolio = sys.argv[1]
        config_name = sys.argv[2]
        One_batch_experiment(portfolio, config_name)
    else:
        print ("python Agent_Trader_debug.py portfolio strategy experiment GPUI")
        print ("python Agent_Trader_debug.py portfolio Config_name")
        print("python Agent_Trader_debug.py Batch portfolio Config_name1,Config_name2....")


if __name__ == '__main__':
    #debug_strategy("Portfolio_try1","Strategy_1", "experience1").debug_main(20201101, 20201110)
    main(sys.argv)
    '''
    portfolio   = sys.argv[1]
    strategy    = sys.argv[2]
    experiment  = sys.argv[3]
    StartI      = eval(sys.argv[4])
    EndI        = eval(sys.argv[5])
    debug_strategy(portfolio, strategy, experiment).debug_main(StartI, EndI)
    '''