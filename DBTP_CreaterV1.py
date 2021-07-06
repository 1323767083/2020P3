from DBI_Base import hfq_toolbox
from DBTP_Base import DBTP_Base
import numpy as np
import pandas as pd
import os, pickle,sys
from DBTP_Filters import DBTP_Filters
from DBTP_Creater_Comm import Memory_For_DBTP_Creater

class DBTP_CreaterV1(DBTP_Base):
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
