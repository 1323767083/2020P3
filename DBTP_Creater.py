from DB_Base import DB_Base
from DBTP_Base import DBTP_Base
from DBI_Base import StockList,hfq_toolbox
from sklearn.preprocessing import minmax_scale
#from data_common import hfq_toolbox
import numpy as np
import pandas as pd
import os, pickle,time, sys
from datetime import datetime
from multiprocessing import Process
from DBTP_Filters import DBTP_Filters
class Memory_For_DBTP_Creater:
    def __init__(self,npMemTitlesM,iDBIs,max_len):
        self.npMemTitlesM=npMemTitlesM
        self.iDBIs=iDBIs
        self.max_len=max_len
        self.Reset()

    def Reset(self):
        self.memory = [[] for _ in range(len(self.npMemTitlesM))]
        self.daysI = []

    def get_np_item(self, TitleM, ShapeM):
        idx_found = np.where(self.npMemTitlesM == TitleM)
        assert len(idx_found[0]) == 1
        data = self.memory[idx_found[0][0]]
        assert self.Is_Ready(),"The get_data_item only be called either fnwp is exsist or the mx len reached in record"
        if len(ShapeM) == 1:
            assert len(data[0]) == ShapeM[0]
        elif len(ShapeM) == 2:
            assert len(data[0]) == ShapeM[0]
            assert len(data[0][0]) == ShapeM[1]
        else:
            assert False, "Not Support shape length more than 2 {0}".format(ShapeM)
        return np.array(data)

    def Is_Ready(self):
        return len(self.memory[0]) == self.max_len

    def Is_Last_Day(self, DayI):
        return self.daysI[-1] == DayI

    def Add(self, Stock, DayI):
        iDBIDatas = []
        for iDBI in self.iDBIs:
            flag,mess=iDBI.Generate_DBI_day( Stock, DayI)
            if not flag:
                self.Reset()
                return flag,mess
            Lresult = iDBI.load_DBI(Stock, DayI)
            iDBIDatas.extend(Lresult)
        assert len(iDBIDatas) == len(self.npMemTitlesM),"len(iDBIDatas)={0} len(self.np_TTL_DBITitlesM)={1}".\
            format(len(iDBIDatas),len(self.npMemTitlesM))

        flag_pop=self.Is_Ready()
        for idx, iDBIData in enumerate(iDBIDatas):
            if flag_pop:
                self.memory[idx].pop(0)
            self.memory[idx].append(iDBIData)
        if flag_pop:
            self.daysI.pop(0)
        self.daysI.append(DayI)
        return True, "Success"


class DBTP_Creater(DBTP_Base):
    #def __init__(self, DBTP_Name, max_len=20):
    def __init__(self, DBTP_Name):
        DBTP_Base.__init__(self,DBTP_Name)
        self.iHFQTool=hfq_toolbox()
        self.buff=Memory_For_DBTP_Creater(self.npMemTitlesM, self.iDBIs,self.NumDBIDays)
        self.iFilters= DBTP_Filters(self.FT_TypesD_Flat)

    def create_DBTP_Raw_Data_From_DBI(self):
        assert self.buff.Is_Ready(), "The get_DBTP_data only be called either fnwp is exsist or the mx len reached in record"
        result_datas=[[] for _ in range(self.FT_num)]
        for FT_idx in list(range(self.FT_num)):
            flag_FT = True
            result_data_item = np.array([])
            for Select_TitleM, Selected_ShapeM in zip(self.FT_TitlesM[FT_idx], self.FT_ShapesM[FT_idx]):
                item = self.buff.get_np_item(Select_TitleM, Selected_ShapeM)
                if flag_FT:
                    result_data_item = item
                    flag_FT = False
                else:
                    result_data_item = np.concatenate((result_data_item, item), axis=-1)
            result_datas[FT_idx] = result_data_item
        return result_datas

    def Generate_DBTP_Data(self, Stock, DayI):
        fnwp = self.get_DBTP_data_fnwp(Stock, DayI)
        if os.path.exists(fnwp):
            # return pickle.load(open(fnwp,"rb"))
            print("DBTP from DBI {0} {1} already exists ".format(Stock, DayI))
            return False

        result_datas = self.create_DBTP_Raw_Data_From_DBI()
        for Fidx, label in [[0, "LV"], [1, "SV"],[2,"AV"]]:
            #flag_init_HFQ_Ratio = False
            HFQ_Ratio=[]
            for Filter in self.Filters[Fidx]:
                component = Filter.split("_")
                assert component[0] == label
                if component[-1] in ["NPrice", "HFD"]:
                    if len(HFQ_Ratio)==0:
                        HFQ_Ratio = self.buff.get_np_item("HFQ_Ratio", [1])
                    result_datas, HFQ_Ratio = getattr(self.iFilters, Filter)(result_datas, HFQ_Ratio)
                elif component[-1] in ["Volume"]:
                    result_datas = getattr(self.iFilters, Filter)(result_datas)
                elif component[-1] in ["Sanity"]:
                    assert self.Filters[Fidx][-1] == Filter, "Sanity Filter should be the last one {0}".format(
                        self.Filters[Fidx])
                    result_datas = getattr(self.iFilters, Filter)(result_datas)
                elif "Together" in component[-1]:
                    if len(HFQ_Ratio)==0:
                        HFQ_Ratio = self.buff.get_np_item("HFQ_Ratio", [1])
                    result_datas, HFQ_Ratio,dict_idxs = getattr(self.iFilters, Filter)(result_datas, HFQ_Ratio)
                    if len(dict_idxs)!=0: # means title sorted
                        LV_sorted_fnwp=self.get_Output_DBTP_FT_TitleType_Detail_fnwp(label)
                        if not os.path.exists(LV_sorted_fnwp):
                            df =pd.DataFrame(columns=["Label","Title","Type"])
                            for key,values in dict_idxs.items():
                                for idx in values:
                                    df.loc[len(df)]=[label,self.FT_TitlesD_Flat[Fidx][idx],self.FT_TypesD_Flat[Fidx][idx]]
                            df.to_csv(LV_sorted_fnwp,index=False)
                else:
                    assert False, "only support filter end with NPrice, HFD, Volume, Together,Sanity, not {0}".format(Filter)
        print("DBTP from DBI {0} {1} success generated".format(Stock, DayI))
        pickle.dump(result_datas, open(fnwp, "wb"))
        return True

    def DBTP_generator(self, Stock, StartI, EndI):
        AStart_idx, AStartI=self.get_closest_TD(StartI, True)
        AEnd_idx, AEndI = self.get_closest_TD(EndI, False)
        if AStartI<=AEndI:
            assert AStart_idx>self.NumDBIDays-1
            period=self.nptd[AStart_idx-(self.NumDBIDays-1):AEnd_idx+1]
        else:
            print("{0},{1},{2}".format(Stock, StartI, "No trading day between {0} {1}".format(StartI, EndI)), file=sys.stderr)
            return False, "No trading day between {0} {1}".format(StartI, EndI)
        '''
        if AStartI<=AEndI:
            assert AStart_idx>19
            period=self.nptd[AStart_idx-19:AEnd_idx+1]
        else:
            print("{0},{1},{2}".format(Stock, StartI, "No trading day between {0} {1}".format(StartI, EndI)), file=sys.stderr)
            return False, "No trading day between {0} {1}".format(StartI, EndI)
        '''
        logfnwp = self.get_DBTP_data_log_fnwp(Stock)
        for DayI in period:
            flag, mess=self.buff.Add(Stock,DayI)
            if not flag:
                self.log_append_keep_new([[flag,DayI,mess]], logfnwp, ["Result", "Date", "Message"])
                #self.buff.Reset()   #already called in Add dicontinues in self.memory , so should reset meomory
                print("{0},{1},{2}".format(Stock,DayI,mess ), file=sys.stderr)
                continue
            if not self.buff.Is_Ready():
                if DayI>=StartI:
                    self.log_append_keep_new([[False, DayI, "Not Enough Record"]], logfnwp, ["Result", "Date", "Message"])
                    print("{0},{1},{2}".format(Stock, DayI, "Not Enough Record"), file=sys.stderr)
                continue
            assert self.buff.Is_Last_Day(DayI)

            #self.get_DBTP_data(Stock, DayI)
            flag_return=self.Generate_DBTP_Data(Stock, DayI)
            self.log_append_keep_new([[True, DayI, "Generate" if flag_return else "Already Exists"]], logfnwp, ["Result", "Date", "Message"])
        self.buff.Reset()
        return True, "Success"


class Process_Generate_DBTP(Process):
    def __init__(self, DBTP_Name, logdn,Stocks, StartI, EndI, process_id, flag_overwrite, flag_debug=False ):
        Process.__init__(self)
        self.iDBTP_Creater=DBTP_Creater(DBTP_Name)
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
            print("#####################################################################")
            print("start process at {0}".format(datetime.now().time()))
            total_num=len(self.Stocks)
            for idx,Stock in enumerate(self.Stocks):
                print ("********************************************************************")
                print ("Start Generate {0} {1} {2} @ {3}".format(Stock, self.StartI, self.EndI,datetime.now().time() ))
                flag, mess=self.iDBTP_Creater.DBTP_generator(Stock, self.StartI, self.EndI)
                print ("End with {0}".format(mess))
                print ("Process {0} finish {1:.2f}".format(self.process_id, (idx+1)/total_num), file=sys.__stdout__)
                newstdout.flush()


def DBTP_creator(DBTP_Name,SL_Name,SL_tag, SL_idx,StartI,EndI,NumP,flag_overwrite):
    #assert StartI < EndI and StartI // 1000000 == 20 and EndI // 1000000 == 20
    assert StartI <=EndI
    logdn=DB_Base().Dir_TPDB_Update_Log
    for sub_dir in [DBTP_Name,"{0}_{1}_{2}".format(SL_Name,SL_tag,SL_idx),"{0}-{1}".format(StartI, EndI)]:
        logdn=os.path.join(logdn,sub_dir)
        if not os.path.exists(logdn):os.mkdir(logdn)
    fnwp = StockList(SL_Name).Sub_fnwp(SL_tag, SL_idx)
    if not os.path.exists(fnwp):
        print("File Not exist {0}".format(fnwp))
        return
    sl = pd.read_csv(fnwp, header=0, names=["stock"])["stock"].tolist()
    sub_len = len(sl) // NumP
    sub_beneficial = len(sl) % NumP

    PIs = []
    for i in list(range(NumP)):
        len_to_get = sub_len + 1 if i < sub_beneficial else sub_len
        # PI=Process_Generate_DBTP(DBTP_Name, SL_Name,sl[:len_to_get+1], StartI, EndI,i)
        PI = Process_Generate_DBTP(DBTP_Name, logdn, sl[:len_to_get], StartI, EndI, i, flag_overwrite)
        PI.daemon = True
        PI.start()
        PIs.append(PI)
        # sl=sl[len_to_get+1:]
        sl = sl[len_to_get:]
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
