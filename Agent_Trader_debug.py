from Agent_Trader import *
from multiprocessing import Process,Manager

def One_batch_experiment(portfolio,config_name):
    df = pd.read_csv(os.path.join("/home/rdchujf/n_workspace/AT", portfolio, "{0}.csv".format(config_name)))
    iManager = Manager()
    #L_E_Weight_Updated_Agent = [iManager.Event() for _ in range(len(df))]
    L_Agent2GPU = iManager.list()
    LL_GPU2Agent = [iManager.list() for _ in range(len(df))]
    E_Stop_GPU=iManager.Event()
    Strategies=list(set(df["strategy"].to_list()))
    assert len(Strategies)==1
    iStrategy=Strategy_Config(portfolio,Strategies[0])
    GPUIs=list(set(df["GPUI"].to_list()))
    assert len(GPUIs) == 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUIs[0])
    #iTrader_GPU=Trader_GPU(iStrategy, L_E_Weight_Updated_Agent, E_Stop_GPU, L_Agent2GPU,LL_GPU2Agent)
    iTrader_GPU = Trader_GPU(iStrategy,E_Stop_GPU, L_Agent2GPU, LL_GPU2Agent)
    iTrader_GPU.daemon = True
    iTrader_GPU.start()

    iProcesses = []
    Stop_Events = []
    for process_idx, row in df.iterrows():
        E_Stop = iManager.Event()
        experiment_config_params=row["total_invest"], row["min_invest"], row["StartI"], row["EndI"], row["flag_Print_on_screen_or_file"]
        iProcess = Strategy_agent(portfolio, row["strategy"], row["experiment"], experiment_config_params,process_idx,L_Agent2GPU, LL_GPU2Agent[process_idx],E_Stop)
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
        iTrader_GPU = Trader_GPU(iStrategy, L_E_Weight_Updated_Agent, E_Stop_GPU, L_Agent2GPU, LL_GPU2Agent)
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
        print ("python Agent_Trader_debug.py portfolio strategy experiment GPUI")
        print ("python Agent_Trader_debug.py portfolio Config_name")
        print("python Agent_Trader_debug.py Batch portfolio Config_name1,Config_name2....")

if __name__ == '__main__':
    main(sys.argv)
