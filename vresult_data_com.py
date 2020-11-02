#are base related
import os,re,collections
import pandas as pd
import numpy as np
import config as sc

def get_data_start_end(lgc, process_name):
    eval_process_idx = int(re.findall(r"{0}_(\d)".format(lgc.eval_process_seed), process_name)[0])
    _, StartI, EndI=lgc.l_eval_SL_param[eval_process_idx]
    return str(StartI), str(EndI)

def get_addon_setting(system_name,process_name):
    def Month_list(Start_YM, End_YM):
        LMonth=[]
        if  Start_YM>End_YM:
            return []
        StartYear= int(Start_YM[:4])
        StartMonth = int(Start_YM[-2:])
        EndYear=int(End_YM[:4])
        EndMonth=int(End_YM[-2:])
        if StartYear==EndYear:
            for M in range(StartMonth, EndMonth+1):
                LMonth.append("{0}{1:02d}".format(StartYear, M))
            return  LMonth
        for M in range(StartMonth, 13):
            LMonth.append("{0}{1:02d}".format(StartYear, M))
        if EndYear-StartYear>1:
            for Y in range(StartYear+1, EndYear):
                for M in range (1, 13):
                    LMonth.append("{0}{1:02d}".format(Y, M))
        for M in range (1, EndMonth+1):
            LMonth.append("{0}{1:02d}".format(EndYear, M))
        return LMonth


    param_fnwp = os.path.join(sc.base_dir_RL_system, system_name, "config.json")
    if not os.path.exists(param_fnwp):
        raise ValueError("{0} does not exisit".format(param_fnwp))
    lgc = sc.gconfig()
    lgc.read_from_json(param_fnwp)

    data_start_s, data_end_s=get_data_start_end(lgc, process_name)

    src_dir = os.path.join(sc.base_dir_RL_system, system_name, process_name)
    Lstock = [fn for fn in os.listdir(src_dir) if len(fn) == 8]

    LEvalT = [int(re.findall(r'\w+T(\d+).h5', fn)[0]) for fn in os.listdir(lgc.brain_model_dir)
            if fn.startswith("train_model_AIO_")]
    LEvalT.sort()
    LEvalT.pop(-1)
    LEvalT.pop(0)
    decision = input("current EvalT from {0} to {1} specify end Eval(Y/N)?".format(LEvalT[0],LEvalT[-1]))
    if decision == "Y":
        endEvalT=LEvalT[-1]+1
        while not( endEvalT%lgc.num_train_to_save_model==0 and endEvalT<= LEvalT[-1]):
            endEvalT = int(input("current EvalT from {0} to {1} input specify End TEval?".format(LEvalT[0], LEvalT[-1])))
        else:
            endEvalT_idx=LEvalT.index(endEvalT)
            del LEvalT[endEvalT_idx:]
    LYM = Month_list(data_start_s[:6], data_end_s[:6])[1:-2]

    dir_analysis = os.path.join(sc.base_dir_RL_system, system_name, "analysis")
    if not os.path.exists(dir_analysis): os.makedirs(dir_analysis)

    return Lstock, LEvalT, LYM, lgc


class are_esi_reader:
    ssdi_seed = "log_s_s_d_i_T"
    are_fn_seed = "log_a_r_e_T"
    def __init__(self,system_name, process_name):
        self.process_name   =   process_name
        self.system_name    =   system_name
        self.src_dir = os.path.join(sc.base_dir_RL_system, self.system_name, self.process_name)
        self.dir_analysis=os.path.join(sc.base_dir_RL_system,self.system_name, "analysis")
        if not os.path.exists(self.dir_analysis): os.makedirs(self.dir_analysis)

    def _read_stock_ssdi(self, stock, evalT):
        stock_dir = os.path.join(self.src_dir, stock)
        fn = "{0}{1}.csv".format(self.ssdi_seed, evalT)
        fnwp=os.path.join(stock_dir,fn )
        if os.path.exists(fnwp):
            df = pd.read_csv(fnwp, header=0)
        else:
            print("can not find {0}".format(fnwp))
            df=pd.DataFrame()
        return df

    def _read_stock_are(self, stock, evalT):
        stock_dir = os.path.join(self.src_dir, stock)
        fn = "{0}{1}.csv".format(self.are_fn_seed, evalT)
        fnwp=os.path.join(stock_dir,fn )
        if os.path.exists(fnwp):
            try:
                df = pd.read_csv(fnwp, header=0)
            except Exception as e:
                assert False, " {0} {1}".format(fnwp,e)
            if len(df) == 0:
                # return pd.DataFrame(),{}
                return False,"File_is_empty"
            else:
                return True, df
        else:
            print("can not find {0}".format(fnwp))
            return False, "File_Not_Found"



