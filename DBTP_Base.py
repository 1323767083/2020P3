from DBI_Base import DBI_init
from DBR_Reader import Raw_HFQ_Index
from DBI_Creater import *
import os,json, random
from _collections import OrderedDict
import pandas as pd
import numpy as np
{
    "DataFromDBI":
    {
        "LV":{
            "VTest":["Price_VS_Mount", "Sell_Dan", "Buy_Dan","Exchange_Ratios"]
        },
        "SV":{
            "VTest":["Norm_Average_Price_And_Mount"]
        },
        "Reference":{
            "VTest": ["DateI","hfq_ratio", "Potential_9_Nprice"]
         }
    },
    "Param":
    {

    }
}

class DBTP_Base(DBI_init):
    def __init__(self, DBTP_Name):
        DBI_init.__init__(self)
        self.DBTP_Name=DBTP_Name
        self.Dir_DBTP_WP=os.path.join(self.Dir_TPDB,self.DBTP_Name)
        DBTP_fnwp = os.path.join(self.Dir_DBTP_WP, "DBTP_Definition.json")
        for fndn in [self.Dir_DBTP_WP,DBTP_fnwp]:
            assert os.path.exists(fndn),"DBI name folder with its Type_Definition.json should be ready before run the program"
        self.Dir_DBTP_log = os.path.join(self.Dir_DBTP_WP, "log")
        for dn in [self.Dir_DBTP_log]:
            if not os.path.exists(dn): os.mkdir(dn)

        self.DBTP_Definition = json.load(open(DBTP_fnwp, "r"), object_pairs_hook=OrderedDict)
        temp_list = []
        for L1_item in self.DBTP_Definition["DataFromDBI"].keys():
            temp_list.extend(self.DBTP_Definition["DataFromDBI"][L1_item].keys())
        self.DBINames = list(set(temp_list))
        assert len(self.DBINames) >= 1

        self.iDBIs = [DBI_Creater(DBI_name) for DBI_name in self.DBINames]

        MemTitlesM = []
        self.MemShapesM = []
        self.MemTypesD = []
        self.MemTitlesD = []
        for iDBI in self.iDBIs:
            #TitlesM = iDBI.TypeDefinition.keys()
            MemTitlesM.extend(iDBI.DBITitlesM)
            self.MemShapesM.extend(iDBI.DBIShapesM)
            self.MemTypesD.extend(iDBI.DBITypesD)
            self.MemTitlesD.extend(iDBI.DBITitlesD)
        self.npMemTitlesM = np.array(MemTitlesM)


        self.FT_num=3
        self.FT_TitlesM=[[] for _ in range(self.FT_num)]
        self.FT_ShapesM=[[] for _ in range(self.FT_num)]
        self.FT_TypesD_Flat=[[] for _ in range(self.FT_num)]
        self.FT_TitlesD_Flat = [[] for _ in range(self.FT_num)]
        self.FT_Names=["LV", "SV", "Reference"]
        for FT_idx in list(range(self.FT_num)):
            for DBIName in self.DBTP_Definition["DataFromDBI"][self.FT_Names[FT_idx]].keys():
                self.FT_TitlesM[FT_idx].extend(self.DBTP_Definition["DataFromDBI"][self.FT_Names[FT_idx]][DBIName])
                for FT_title in self.DBTP_Definition["DataFromDBI"][self.FT_Names[FT_idx]][DBIName]:
                    found_idx=np.where(self.npMemTitlesM==FT_title)
                    assert len(found_idx[0])==1
                    self.FT_TypesD_Flat[FT_idx].extend(self.MemTypesD[found_idx[0][0]])
                    self.FT_TitlesD_Flat[FT_idx].extend(self.MemTitlesD[found_idx[0][0]])
                    self.FT_ShapesM[FT_idx].append(self.MemShapesM[found_idx[0][0]])


    def get_DBTP_data_fnwp(self, stock, dayI):
        dn=self.Dir_DBTP_WP
        for subdir in ["data",stock, str(dayI//100)]:
            dn = os.path.join(dn,subdir)
            if not os.path.exists(dn): os.mkdir (dn)
        return os.path.join(dn,"{0}.pickle".format(dayI))

    def get_DBTP_data_log_fnwp(self, stock):
        return os.path.join(self.Dir_DBTP_log,"{0}.csv".format(stock))


