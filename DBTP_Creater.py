from DBTP_Base import DBTP_Base
from DBI_Base import DBI_init,StockList,hfq_toolbox
from sklearn.preprocessing import minmax_scale
#from data_common import hfq_toolbox
import numpy as np
import pandas as pd
import os, pickle,time, sys
from datetime import datetime
from multiprocessing import Process

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
            fnwp = iDBI.get_DBI_data_fnwp(Stock, DayI)
            Lresult = pickle.load(open(fnwp, "rb"))
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
    def __init__(self, DBTP_Name, max_len=20):
        DBTP_Base.__init__(self,DBTP_Name)
        self.iHFQTool=hfq_toolbox()
        self.buff=Memory_For_DBTP_Creater(self.npMemTitlesM, self.iDBIs,max_len)

    def Adjust_on_NPrice(self,result_datas):
        HFQ_Ratio = self.buff.get_np_item("HFQ_Ratio", [1])

        FT_idx=0
        SelectedIdxs=[idx for idx, typeD in enumerate(self.FT_TypesD_Flat[FT_idx]) if typeD=="NPrice"]
        SelectedData=result_datas[FT_idx][:,SelectedIdxs]
        adj_HFQ_Ratio=[]
        for _ in range(len(SelectedIdxs)):
            adj_HFQ_Ratio.extend(HFQ_Ratio.tolist())
        SelectedDataHprice=self.iHFQTool.get_hfqprice_from_Nprice(SelectedData.reshape((-1,)),np.array(adj_HFQ_Ratio).reshape((-1,)))
        SelectedDataHpriceNorm = minmax_scale(SelectedDataHprice.reshape((-1,1)), feature_range=(0, 1), axis=0).ravel()
        adjSelectedData=SelectedDataHpriceNorm.reshape((-1,len(SelectedIdxs)))
        for adj_idx, source_idx in enumerate (SelectedIdxs):
            result_datas[FT_idx][:,source_idx] =adjSelectedData[:,adj_idx]

        FT_idx = 1
        HFQ_Ratio20_25 = []
        for ia in HFQ_Ratio:
            HFQ_Ratio20_25.extend([ia[0] for _ in range(result_datas[FT_idx].shape[1])])
        npHFQ_Ratio20_25=np.array(HFQ_Ratio20_25)
        assert len(self.FT_TypesD_Flat[FT_idx]) == result_datas[FT_idx].shape[2], "len(self.FT_TypesD_Flat[1])= {0} result_datas[1].shape= {1}".format(
            self.FT_TypesD_Flat[FT_idx],  result_datas[FT_idx].shape)

        SelectedIdxs = [idx for idx, typeD in enumerate(self.FT_TypesD_Flat[FT_idx]) if typeD == "NPrice"]
        assert len(SelectedIdxs)==1, "only support sv has one column Nprice type SelectedIdxs= {0} ".format(SelectedIdxs)
        SelectedData = result_datas[FT_idx][:,:,SelectedIdxs]
        SelectedDataHprice = self.iHFQTool.get_hfqprice_from_Nprice(SelectedData.reshape((-1,)),npHFQ_Ratio20_25)
        SelectedDataHpriceNorm = minmax_scale(SelectedDataHprice.reshape((-1, 1)), feature_range=(0, 1), axis=0).ravel()

        adjSelectedData=SelectedDataHpriceNorm.reshape((result_datas[FT_idx].shape[0],result_datas[FT_idx].shape[1]))
        for idx in list(range(result_datas[FT_idx].shape[0])):
            result_datas[FT_idx][idx,:,SelectedIdxs[0]]=adjSelectedData[idx]
        return result_datas

    def Adjust_on_Volume(self,result_datas):
        FT_idx=0
        SelectedIdxs = [idx for idx, typeD in enumerate(self.FT_TypesD_Flat[FT_idx]) if typeD == "Volume"]
        assert len(SelectedIdxs)==0,"Volume type in lv  has not tested"
        FT_idx = 1
        assert len(self.FT_TypesD_Flat[FT_idx]) == result_datas[FT_idx].shape[2], "len(self.FT_TypesD_Flat[1])= {0} result_datas[1].shape= {1}".format(
            self.FT_TypesD_Flat[FT_idx],  result_datas[FT_idx].shape)

        SelectedIdxs = [idx for idx, typeD in enumerate(self.FT_TypesD_Flat[FT_idx]) if typeD == "Volume"]
        assert len(SelectedIdxs)==1, "sv only support has one column Volume type"
        SelectedData = result_datas[FT_idx][:, :, SelectedIdxs]
        #SelectedDataVolume = minmax_scale(SelectedData.reshape((-1, 1)), feature_range=(0, 1), axis=0).ravel()
        SelectedDataVolume = minmax_scale(SelectedData.astype(float).reshape((-1, 1)), feature_range=(0, 1), axis=0).ravel()


        adjSelectedDataVolume = SelectedDataVolume.reshape((result_datas[FT_idx].shape[0], result_datas[FT_idx].shape[1]))
        for idx in list(range(result_datas[FT_idx].shape[0])):
            result_datas[FT_idx][idx, :, SelectedIdxs[0]] = adjSelectedDataVolume[idx]
        return result_datas


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
            #return pickle.load(open(fnwp,"rb"))
            print("DBTP from DBI {0} {1} already exists ".format(Stock, DayI))
            return False

        print("DBTP from DBI {0} {1} success generated".format(Stock, DayI))
        result_datas=self.create_DBTP_Raw_Data_From_DBI()
        result_datas=self.Adjust_on_NPrice(result_datas)
        result_datas = self.Adjust_on_Volume(result_datas)
        pickle.dump(result_datas, open(fnwp, "wb"))
        #return result_datas
        return True


    def DBTP_generator(self, Stock, StartI, EndI):
        AStart_idx, AStartI=self.get_closest_TD(StartI, True)
        AEnd_idx, AEndI = self.get_closest_TD(EndI, False)
        if AStartI<=AEndI:
            assert AStart_idx>19
            period=self.nptd[AStart_idx-19:AEnd_idx+1]
        else:
            print("{0},{1},{2}".format(Stock, StartI, "No trading day between {0} {1}".format(StartI, EndI)), file=sys.stderr)
            return False, "No trading day between {0} {1}".format(StartI, EndI)
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
    def __init__(self, DBTP_Name, SL_Name,Stocks, StartI, EndI, process_id ):
        Process.__init__(self)
        self.iDBTP_Creater=DBTP_Creater(DBTP_Name)
        self.Stocks=Stocks
        self.StartI= StartI
        self.EndI= EndI
        self.process_id=process_id
        logdn=self.iDBTP_Creater.Dir_IDB
        for sub_dir in ["Stock_List",SL_Name, "CreateLog"]:
            logdn=os.path.join(logdn,sub_dir)
            if not os.path.exists(logdn): os.mkdir(logdn)

        self.stdoutfnwp=os.path.join(logdn,"Process{0}Output.txt".format(process_id))
        self.stderrfnwp = os.path.join(logdn, "Process{0}Error.txt".format(process_id))
        pd.DataFrame(self.Stocks,columns=["stock"]).to_csv(os.path.join(logdn,"Process{0}SL.csv".format(process_id)), index=False)

    def run(self):
        print ("Printout has been redirected to {0}".format(self.stdoutfnwp))
        from contextlib import redirect_stdout,redirect_stderr
        newstdout = open(self.stdoutfnwp, "a")
        newstderr = open(self.stderrfnwp, "a")
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
def DBTP_main(DBTP_Name,SL_Name, NumP=4):
    iSL=StockList(SL_Name)
    for tag, idx, StartI, EndI in iSL.SLDef["DBTP_Generator"]:
        assert StartI<EndI and StartI//1000000==20 and EndI//1000000==20
        fnwp=iSL.Sub_fnwp(tag, idx)
        if not os.path.exists(fnwp):
            print ("File Not exist {0}".format(fnwp))
            return
        sl=pd.read_csv(fnwp,header=0, names=["stock"])["stock"].tolist()
        sub_len=len(sl)//NumP
        sub_beneficial=len(sl)%NumP
        PIs=[]
        for i in list(range(NumP)):
            len_to_get=sub_len+1 if i< sub_beneficial else sub_len
            #PI=Process_Generate_DBTP(DBTP_Name, SL_Name,sl[:len_to_get+1], StartI, EndI,i)
            PI = Process_Generate_DBTP(DBTP_Name, SL_Name, sl[:len_to_get], StartI, EndI, i)
            PI.daemon = True
            PI.start()
            PIs.append(PI)
            #sl=sl[len_to_get+1:]
            sl=sl[len_to_get:]
        while  any([PI.is_alive() for PI in PIs]):
            time.sleep(10)
        for PI in PIs:
            PI.join()

