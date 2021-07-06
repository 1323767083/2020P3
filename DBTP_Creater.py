import pandas as pd
import os, sys, time
from DB_Base import DB_Base
from DBTP_Base import DBTP_Base
from DBI_Base import StockList,DBI_init_with_TD
from datetime import datetime
from multiprocessing import Process
from DBTP_CreaterV1 import DBTP_CreaterV1
from DBTP_CreaterV2 import DBTP_CreaterV2

class Process_Generate_DBTP(Process):
    def __init__(self, DBTP_Name, SL_Name,logdn,Stocks, StartI, EndI, process_id, flag_overwrite, flag_debug=False ):
        Process.__init__(self)
        self.DBTP_Name=DBTP_Name
        self.SL_Name=SL_Name
        self.Stocks=Stocks
        self.StartI= StartI
        self.EndI= EndI
        self.process_id=process_id
        self.flag_overwrite=flag_overwrite  #overwrite flag是 让 log file 从新开始写
        self.flag_debug=flag_debug  # debug flag 是 让所有输出都在屏幕上， 不写log
        self.stdoutfnwp=os.path.join(logdn,"Process{0}Output.txt".format(process_id))
        self.stderrfnwp = os.path.join(logdn, "Process{0}Error.txt".format(process_id))
        pd.DataFrame(self.Stocks,columns=["stock"]).to_csv(os.path.join(logdn,"Process{0}SL.csv".format(process_id)), index=False)

    def run(self):
        print ("Printout has been redirected to {0}".format(self.stdoutfnwp))
        from contextlib import redirect_stdout,redirect_stderr
        if self.flag_debug:
            newstdout = sys.__stdout__
            newstderr = sys.__stderr__
        else:
            newstdout = open(self.stdoutfnwp, "w" if self.flag_overwrite else "a")
            newstderr = open(self.stderrfnwp, "w" if self.flag_overwrite else "a")
        with redirect_stdout(newstdout),redirect_stderr(newstderr):
            tag=DBTP_Base(self.DBTP_Name).CLN_DBTPCreater[-2:]
            assert tag in ["V1","V2"]
            getattr(self,f"Create{tag}")()

    def CreateV1(self):
        self.iDBTP_Creater = DBTP_CreaterV1(self.DBTP_Name)
        total_num = len(self.Stocks)
        for idx, Stock in enumerate(self.Stocks):
            print("********************************************************************")
            print(f"start process {idx} at {datetime.now().time()}", file=sys.__stdout__)
            print("Start Generate {0} {1} {2} @ {3}".format(Stock, self.StartI, self.EndI, datetime.now().time()))
            flag, mess = self.iDBTP_Creater.DBTP_generator(Stock, self.StartI, self.EndI)
            print("End with {0}".format(mess))
            print(f"End Process {0} finish {1:.2f}".format(self.process_id, (idx + 1) / total_num), file=sys.__stdout__)
            #newstdout.flush()

    def CreateV2(self):
        self.iDBTP_Creater = DBTP_CreaterV2(self.DBTP_Name,self.Stocks,self.SL_Name)
        print("********************************************************************")
        print(f"start process at {datetime.now().time()}", file=sys.__stdout__)
        print("Start Generate {0} {1} @{2} ".format(self.StartI, self.EndI, datetime.now().time()))
        flag, mess = self.iDBTP_Creater.DBTP_generator(self.StartI, self.EndI)
        print("End with {0}".format(mess))
        print(f"End Process", file=sys.__stdout__)


def DBTP_creator(DBTP_Name,SL_Name,SL_tag, SL_idx,StartI,EndI,NumP,flag_overwrite):
    #assert StartI < EndI and StartI // 1000000 == 20 and EndI // 1000000 == 20
    assert StartI <=EndI
    logdn=DB_Base().Dir_TPDB_Update_Log
    for sub_dir in [DBTP_Name,"{0}_{1}_{2}".format(SL_Name,SL_tag,SL_idx),"{0}-{1}".format(StartI, EndI)]:
        logdn=os.path.join(logdn,sub_dir)
        if not os.path.exists(logdn):os.mkdir(logdn)

    flag,sl= StockList(SL_Name).get_sub_sl(SL_tag, SL_idx)
    tag = DBTP_Base(DBTP_Name).CLN_DBTPCreater[-2:]
    assert tag in ["V1","V2"]
    PIs = []
    if tag=="V1":
        sub_len = len(sl) // NumP
        sub_beneficial = len(sl) % NumP
        for i in list(range(NumP)):
            len_to_get = sub_len + 1 if i < sub_beneficial else sub_len
            PI = Process_Generate_DBTP(DBTP_Name, SL_Name,logdn, sl[:len_to_get], StartI, EndI, i, flag_overwrite)
            PI.daemon = True
            PI.start()
            PIs.append(PI)
            sl = sl[len_to_get:]
    else:
        td=DBI_init_with_TD()
        AStart_idx, AStartI=td.get_closest_TD(StartI, True)
        AEnd_idx, AEndI = td.get_closest_TD(EndI, False)
        if AStartI<=AEndI:
            period=td.nptd[AStart_idx:AEnd_idx+1]
        else:
            assert False, "No trading day between {0} {1}".format(StartI, EndI)
        sub_len = len(period) // NumP
        sub_beneficial = len(period) % NumP
        for i in list(range(NumP)):
            len_to_get = sub_len + 1 if i < sub_beneficial else sub_len
            PI = Process_Generate_DBTP(DBTP_Name, SL_Name,logdn, sl, period[:len_to_get][0], period[:len_to_get][-1], i, flag_overwrite)
            PI.daemon = True
            PI.start()
            PIs.append(PI)
            period = period[len_to_get:]

    while any([PI.is_alive() for PI in PIs]):
        time.sleep(10)
    print ("Logs store in {0}".format(logdn))
    for PI in PIs:
        PI.join()
    fns = [fn for fn in os.listdir(logdn) if "Error" in fn]
    fns.sort()
    for fn in fns:
        print("{0} with size {1}".format(fn, os.path.getsize(os.path.join(logdn, fn))))

def DBTP_creator_on_SLperiod(DBTP_Name,SL_Name, NumP, flag_overwrite):
    iSL=StockList(SL_Name)
    for SL_tag,SL_idx, StartI, EndI in iSL.SLDef["DBTP_Generator"]:
        DBTP_creator(DBTP_Name,SL_Name,SL_tag, SL_idx,StartI,EndI,NumP,flag_overwrite)
