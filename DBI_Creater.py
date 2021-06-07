import os,pickle
from copy import deepcopy

import numpy as np
from DBI_Base import DBI_Base
from DBI_Generator_LVs import *
from DBI_Generator_SVs import *
from DBI_Generator_Refs import *


class DBI_Creater(DBI_Base):
    def __init__(self,DBI_name):
        DBI_Base.__init__(self,DBI_name)
        self.DBITitlesM = []
        self.DBIShapesM = []
        self.DBITypesD = []
        self.DBITitlesD = []
        TitlesM = self.TypeDefinition["Elements"]
        self.DBITitlesM.extend(TitlesM)
        for TitleM in TitlesM:
            iE=globals()[TitleM ]()
            self.DBIShapesM.append(iE.Get_ShapesM())
            self.DBITypesD.append(iE.Get_TypesD())
            self.DBITitlesD.append(iE.Get_TitlesD())

        #config Tinpai Data
        self.get_Tinpai_item=getattr(self,self.TypeDefinition["TinpaiFun"])

        #optimize performance
        self.current_stock=""
        self.hfq_flag,self.hfq_df, self.hfq_mess="","",""

    def IsTinpai(self, Stock, DayI):
        fnwp = self.get_DBI_hfq_fnwp(Stock)
        if not os.path.exists(fnwp):
            return False, "", "HFQ File Not Found****{0}".format(fnwp)
        flag, df, mess = self.get_hfq_df(fnwp)
        if not flag:
            return False, "", mess
        return True, df[df["date"] == str(DayI)].empty, "Success"

    #def get_Tinpai_item(self, DBITypeD,DayI):
    def TinpaiZero(self, DBITypeD, DayI):
        assert DBITypeD in ["NPrice", "Percent", "Volume", "Ratio", "DateI","Flag_Tradable","NPrice_Not_Normal"],\
            "{0} {1}".format( DBITypeD,DayI)
        if DBITypeD in ["NPrice"]:
            return 0.0
        elif DBITypeD in ["Volume"]:
            return 0
        elif DBITypeD in ["Percent", "Ratio"]:
            return 0.0
        elif DBITypeD in ["NPrice_Not_Normal"]:
            return 0.0
        elif DBITypeD in ["DateI"]:
            return DayI
        elif DBITypeD in ["Flag_Tradable"]:
            return False

    def TinpaiNAN(self, DBITypeD, DayI):
        assert DBITypeD in ["NPrice", "Percent", "Volume", "Ratio", "DateI","NPrice_Not_Normal","Flag_Tradable","NPrice_Not_Normal"],\
            "{0} {1}".format( DBITypeD,DayI)
        if DBITypeD in ["NPrice"]:
            return np.NaN
        elif DBITypeD in ["Volume"]:
            return 0
        elif DBITypeD in ["Percent", "Ratio"]:
            return 0
        elif DBITypeD in ["NPrice_Not_Normal"]:
            return np.NaN
        elif DBITypeD in ["DateI"]:
            return DayI
        elif DBITypeD in ["Flag_Tradable"]:
            return False

    ##DBI data generator
    def Generate_Oneday_Tinpai(self,Stock, DayI):
        DBIdata=[]
        for DBIShapeM, DBITypesD in zip(self.DBIShapesM,self.DBITypesD):
            if len(DBIShapeM)==1:
                DBIdata.append([self.get_Tinpai_item(DBITypeD,DayI) for DBITypeD in DBITypesD])
            elif len(DBIShapeM)==2:
                DBIdata.append([[self.get_Tinpai_item(DBITypeD,DayI) for DBITypeD in DBITypesD] for _ in range(DBIShapeM[0])])
            else:
                assert False, "Not Support shape length more than 2 {0}".format(DBIShapeM)
        self.dump_DBI(Stock, DayI, DBIdata)
        return True

    def Generate_Oneday(self,  df_qz, df_hfq, dayI, stock, param):
        result_L=[]
        for Element in param["Elements"]:
            iE=globals()[Element]()
            inputsL=[]
            for input_item in iE.Get_Input_Params():
                if input_item=="DateI":
                    inputsL.append(dayI)
                elif input_item=="QZ_DF":
                    inputsL.append(df_qz)
                elif input_item=="HFQ_DF":
                    inputsL.append(df_hfq)
                else:
                    raise ValueError("{0} not in the supported input param type".format(input_item))
            result_L.append(iE.Gen(inputsL))
        self.dump_DBI(stock, dayI,result_L)
        return True

    def Generate_DBI_day(self, Stock, DayI):
        logfnwp = self.get_DBI_log_fnwp(Stock)

        if self.Is_DBI_Oneday_exists(Stock, DayI):
            self.log_append_keep_new([[True, DayI, "Success" + "Already Exists"]],logfnwp, ["Result", "Date", "Message"])
            print("DBI {0} {1} {2}".format(Stock, DayI, "Success" + "Already Exists"))
            return True, "Success"

        if DayI>self.Raw_Normal_Lumpsum_EndDayI:
            update_logfnwp = self.get_DBI_Update_Log_HFQ_Index_fnwp(DayI)
            if not os.path.exists(update_logfnwp):
                Error_Mess="Need Update Index and HFQ and decompress QZ for {0} first as addon".format(DayI)
                self.log_append_keep_new([[False, DayI, Error_Mess]], logfnwp, ["Result", "Date", "Message"])
                print("DBI {0} {1} {2}".format(Stock, DayI, "Fail Generate " + Error_Mess))
                return False, Error_Mess


        if self.current_stock!=Stock:
            DBI_HFQ_fnwp=self.get_DBI_hfq_fnwp(Stock)
            if not os.path.exists(DBI_HFQ_fnwp):
                Error_Mess= "DBI HFQ File Not Found****{0}".format(DBI_HFQ_fnwp)
                self.log_append_keep_new([[False, DayI,Error_Mess]], logfnwp, ["Result", "Date", "Message"])
                print("DBI {0} {1} {2}".format(Stock, DayI, "Fail Generate "+ Error_Mess))
                return False, Error_Mess
            hfq_flag,hfq_df, hfq_mess=self.get_hfq_df(DBI_HFQ_fnwp)
            if not hfq_flag:
                self.log_append_keep_new([[False, DayI,hfq_mess]], logfnwp, ["Result", "Date", "Message"])
                print("DBI {0} {1} {2}".format(Stock, DayI, "Fail Generate " + hfq_mess))
                return False, hfq_mess
            self.current_stock = Stock
            self.hfq_flag, self.hfq_df, self.hfq_mess=hfq_flag,hfq_df, hfq_mess
        else:
            hfq_flag, hfq_df, hfq_mess=self.hfq_flag, self.hfq_df, self.hfq_mess

        if hfq_df[hfq_df["date"] == str(DayI)].empty: # Tinpai
            Flag_Return=self.Generate_Oneday_Tinpai(Stock, DayI)
            self.log_append_keep_new([[True, DayI, "Tinpai" + "Generate" if Flag_Return else  "Already Exists"]], logfnwp, ["Result", "Date", "Message"])
            return True, "Tinpai"
        qz_flag, qz_df, qz_mess = self.IRD.get_qz_df_inteface( Stock, DayI)
        if not qz_flag:
            self.log_append_keep_new([[False,DayI,qz_mess]], logfnwp, ["Result", "Date", "Message"])
            return False,qz_mess
        self.Generate_Oneday(qz_df, hfq_df, DayI, Stock,self.TypeDefinition)
        self.log_append_keep_new([[True, DayI, "Success" + "Generate" ]], logfnwp, ["Result", "Date", "Message"])
        print("DBI {0} {1} {2}".format(Stock, DayI,"Success" + "Generate" ))
        return True, "Success"



# 7z a -r 20200703.7z 2020-07-03/