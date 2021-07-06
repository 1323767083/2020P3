from DBI_Base import DBI_init_with_TD
from DBR_Reader import Raw_HFQ_Index
from DBI_Creater import *
import os,json, random
from _collections import OrderedDict
import pandas as pd
import numpy as np
{
    "CLN_DBTPCreater":"DBTP_CreaterV2",
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
        "LV": {
            "Filters_In_order":[]
        },
        "SV": {
            "Filters_In_order":[]
        },
        "Reference": {
            "Filters_In_order":[]
        }
    },
    "NumDBIDays": 20,
    "V2Param": {
        "BuyAtTimes": [94500],
        "SellAtTimes": [94500],
        "Profit_Scale": 100
    }
}

class DBTP_Base(DBI_init_with_TD):
    def __init__(self, DBTP_Name):
        DBI_init_with_TD.__init__(self)
        self.DBTP_Name=DBTP_Name
        self.Dir_DBTP_WP=os.path.join(self.Dir_TPDB,self.DBTP_Name)
        DBTP_fnwp = os.path.join(self.Dir_DBTP_WP, "DBTP_Definition.json")
        for fndn in [self.Dir_DBTP_WP,DBTP_fnwp]:
            assert os.path.exists(fndn),f"DBTP name folder with its Type_Definition.json should be ready before run the program {fndn}"
        self.Dir_DBTP_log = os.path.join(self.Dir_DBTP_WP, "log")
        for dn in [self.Dir_DBTP_log]:
            if not os.path.exists(dn): os.mkdir(dn)

        self.DBTP_Definition = json.load(open(DBTP_fnwp, "r"), object_pairs_hook=OrderedDict)
        if "CLN_DBTPCreater" not in self.DBTP_Definition.keys():
            self.CLN_DBTPCreater = "DBTP_Creater"
        else:
            self.CLN_DBTPCreater = "DBTP_CreaterV2"
        temp_list = []
        for L1_item in self.DBTP_Definition["DataFromDBI"].keys():
            temp_list.extend(self.DBTP_Definition["DataFromDBI"][L1_item].keys())
        self.DBINames = list(set(temp_list))
        assert len(self.DBINames) >= 1
        self.iDBIs = [DBI_Creater(DBI_name) for DBI_name in self.DBINames]
        assert len(set([iDBI.TypeDefinition["TinpaiFun"] for iDBI in self.iDBIs]))==1, "DBTP used DBI should have same TinpaiFun"

        if len(self.DBTP_Definition["Param"])==0:
            self.Filters=[["LV_NPrice","LV_Volume"],["SV_NPrice","SV_Volume"],[]]
        else:
            self.Filters=[self.DBTP_Definition["Param"]["LV"]["Filters_In_order"],
                          self.DBTP_Definition["Param"]["SV"]["Filters_In_order"],
                          self.DBTP_Definition["Param"]["Reference"]["Filters_In_order"]]

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

        if "NumDBIDays" not in self.DBTP_Definition.keys(): #legacy
            self.DBTP_Definition["NumDBIDays"]=20
        self.NumDBIDays=self.DBTP_Definition["NumDBIDays"]

        fnwp=self.get_Input_DBTP_FT_TitleType_Detail_fnwp()
        if not os.path.exists(fnwp):
            df = pd.DataFrame(columns=["FeatureGroupName", "Title", "Type"])
            for featureGroup_name, titles, Types, in zip(["LV", "SV", "Ref"], self.FT_TitlesD_Flat, self.FT_TypesD_Flat):
                assert len(titles) == len(Types)
                for title, Type in zip(titles, Types):
                    df.loc[len(df)] = [featureGroup_name, title, Type]
            df.to_csv(fnwp, index=False)
    def get_DBTP_data_fnwp(self, stock, dayI):
        dn=self.Dir_DBTP_WP
        for subdir in ["data",stock, str(dayI//100)]:
            dn = os.path.join(dn,subdir)
            if not os.path.exists(dn): os.mkdir (dn)
        return os.path.join(dn,"{0}.pickle".format(dayI))

    def get_DBTP_data_fnwpV2(self, dayI):
        dn=self.Dir_DBTP_WP
        for subdir in ["data",str(dayI//100)]:
            dn = os.path.join(dn,subdir)
            if not os.path.exists(dn): os.mkdir (dn)
        return os.path.join(dn,"{0}.pickle".format(dayI))


    def get_DBTP_data_log_fnwp(self, stock):
        return os.path.join(self.Dir_DBTP_log,"{0}.csv".format(stock))

    def get_DBTP_data_log_fnwpV2(self, DayI, Stock_list_name):
        return os.path.join(self.Dir_DBTP_log,f"{Stock_list_name}_{DayI}.csv")


    def get_Input_DBTP_FT_TitleType_Detail_fnwp(self):
        return os.path.join(self.Dir_DBTP_WP, "TitleTypeDetail_FT.csv")

    def get_Output_DBTP_FT_TitleType_Detail_fnwp(self,label):
        return os.path.join(self.Dir_DBTP_WP, "TitleTypeDetail_{0}_Sorted.csv".format(label))
