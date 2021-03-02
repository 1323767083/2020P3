from DBI_Generator_Base import DBI_Generator_Base
from copy import deepcopy
import numpy as np
def _Norm_Average_Nprice_And_Mount_Gen_Base(inputs,param):
    df_src=inputs[0]
    df = deepcopy(df_src)
    df["Time"] = df["Time"] / param["time_interval"]  # ten minutes
    df["Time"] = df["Time"].astype(int)
    df=df[df["Time"]>=param["result_should_contain"][0]]
    np_time = df["Time"].values
    np.unique(np_time)
    set_diff = set(param["result_should_contain"]) - set(np_time)

    if len(set_diff) > 0:
        for item in set_diff:
            df.loc[len(df)] = [item, 0, 0, 0, 0, "B", 0, 0.0, 0, 0.0, 0.0]
        df.sort_values(['Time'], ascending=True, inplace=True)

    df_result = df[["Time", "Volume", "Money"]].groupby(["Time"]).sum()
    assert len(df_result) == param["result_len"], "{0} _{1}".format(len(df_result), df_result)
    df_result["average_price"] = df_result["Money"] / df_result["Volume"]
    df_result = df_result.ffill().bfill()
    result = df_result[["average_price", "Volume"]].values.tolist()

    return result

class Norm_Average_Nprice_And_Mount_Whole_Day(DBI_Generator_Base):
    ShapesM = [25,2]
    Input_Params = ["QZ_DF"]
    TitilesD =["Average_NPrice_10M_Whole_Day", "Total_Volume_10M_Whole_Day"]
    TypesD = ["NPrice","Volume"]
    param_norm_average_price_and_mount = {
        "time_interval": 1000,  # 10 minutes
        "result_len": 25,
        "result_should_contain": [92, 93, 94, 95, 100, 101, 102, 103, 104, 105, 110, 111, 112,
                                  130, 131, 132, 133, 134, 135, 140, 141, 142, 143, 144, 145]
    }

    def Gen(self, inputs):
        result= _Norm_Average_Nprice_And_Mount_Gen_Base(inputs,self.param_norm_average_price_and_mount)
        self.Result_Check_Shape(result)
        return result

class Norm_Average_Nprice_And_Mount_Half_Day(DBI_Generator_Base):
    ShapeFirstDim=12
    ShapesM = [ShapeFirstDim,2]
    Input_Params = ["QZ_DF"]
    TitilesD =["Average_NPrice_10M_Half_Day", "Total_Volume_10M_Half_Day"]
    TypesD = ["NPrice","Volume"]
    param_norm_average_price_and_mount = {
        "time_interval": 1000,  # 10 minutes
        "result_len": 13,
        "result_should_contain": [92, 93, 94, 95, 100, 101, 102, 103, 104, 105, 110, 111, 112]
    }
    def Gen(self, inputs):
        param={}
        param["time_interval"] = self.param_norm_average_price_and_mount["time_interval"]
        param["result_len"] = self.ShapeFirstDim
        param["result_should_contain"] = self.param_norm_average_price_and_mount["result_should_contain"][-self.ShapeFirstDim:]
        result= _Norm_Average_Nprice_And_Mount_Gen_Base(inputs,param)
        self.Result_Check_Shape(result)
        return result


def _Norm_Average_Nprice_And_Mount_Gen_1M_Base(inputs,param):
    df_src=inputs[0]
    df = deepcopy(df_src)
    df["Time"] = df["Time"] / param["time_interval"]  # 1 minutes
    df["Time"] = df["Time"].astype(int)
    Sidx=(df["Time"]>=param["result_should_contain"][0])&(df["Time"]<param["result_should_contain"][1])

    before_start_volume=df.loc[Sidx]["Volume"].sum()
    before_start_money = df.loc[Sidx]["Money"].sum()
    before_start_price=before_start_money/before_start_volume if before_start_volume!=0 else 0
    df.drop([idx for idx, flag in enumerate(Sidx) if flag ], inplace =True)
    df.reset_index(inplace=True,drop=True)
    df.loc[len(df)] = [param["result_should_contain"][0], before_start_price, before_start_volume, 0, 0, "B", 0, 0.0, 0, 0.0, before_start_money]
    df=df[df["Time"]>=param["result_should_contain"][0]]
    np_time = df["Time"].values
    np.unique(np_time)
    set_diff = set(param["result_should_contain"]) - set(np_time)
    #print (set_diff)
    if len(set_diff) > 0:
        for item in set_diff:
            #print (len(df))
            df.loc[len(df)] = [item, 0.0, 0, 0, 0, "B", 0, 0.0, 0, 0.0, 0.0]
            #print(len(df))
    df.sort_values(['Time'], ascending=True, inplace=True)

    df_result = df[["Time", "Volume", "Money"]].groupby(["Time"]).sum()

    #df_result.to_csv("/home/rdchujf/a.csv")

    assert len(df_result) == param["result_len"], "{0} _{1}".format(len(df_result), df_result)
    df_result["average_price"] = df_result["Money"] / df_result["Volume"]
    df_result = df_result.ffill().bfill()
    result = df_result[["average_price", "Volume"]].values.tolist()

    return result


class Norm_Average_Nprice_And_Mount_Whole_Day_1M(DBI_Generator_Base):
    ShapesM = [241,2]
    Input_Params = ["QZ_DF"]
    TitilesD =["Average_NPrice_1M_Whole_Day", "Total_Volume_1M_Whole_Day"]
    TypesD = ["NPrice","Volume"]
    def __init__(self):
        fab_1M_list=[925]
        for i in [93, 94, 95, 100, 101, 102, 103, 104, 105, 110, 111, 112,130, 131, 132, 133, 134, 135, 140, 141, 142, 143, 144, 145]:
            fab_1M_list=fab_1M_list+[i*10+idx for idx in list(range(10))]
        self.param_norm_average_price_and_mount_1M = {
            "time_interval": 100,  # 1 minutes
            "result_len": 241,
            "result_should_contain": fab_1M_list
        }
    def Gen(self, inputs):
        result= _Norm_Average_Nprice_And_Mount_Gen_1M_Base(inputs,self.param_norm_average_price_and_mount_1M)
        self.Result_Check_Shape(result)
        return result

def _Norm_Average_Nprice_And_Mount_Gen_5M_Base(inputs,param):
    df_src=inputs[0]
    df = deepcopy(df_src)
    #df["Time"] = df["Time"] / param["time_interval"]  # 1 minutes
    #df["Time"] = df["Time"].astype(int)

    df["Time"]=df["Time"].apply(lambda x: x//1000*10+ (0 if x//100%10<5 else 5))
    Sidx=(df["Time"]>=param["result_should_contain"][0])&(df["Time"]<param["result_should_contain"][1])

    before_start_volume=df.loc[Sidx]["Volume"].sum()
    before_start_money = df.loc[Sidx]["Money"].sum()
    before_start_price=before_start_money/before_start_volume if before_start_volume!=0 else 0
    df.drop([idx for idx, flag in enumerate(Sidx) if flag ], inplace =True)
    df.reset_index(inplace=True,drop=True)
    df.loc[len(df)] = [param["result_should_contain"][0], before_start_price, before_start_volume, 0, 0, "B", 0, 0.0, 0, 0.0, before_start_money]
    df=df[df["Time"]>=param["result_should_contain"][0]]
    np_time = df["Time"].values
    np.unique(np_time)
    set_diff = set(param["result_should_contain"]) - set(np_time)
    #print (set_diff)
    if len(set_diff) > 0:
        for item in set_diff:
            #print (len(df))
            df.loc[len(df)] = [item, 0.0, 0, 0, 0, "B", 0, 0.0, 0, 0.0, 0.0]
            #print(len(df))
    df.sort_values(['Time'], ascending=True, inplace=True)

    df_result = df[["Time", "Volume", "Money"]].groupby(["Time"]).sum()

    assert len(df_result) == param["result_len"], "{0} _{1}".format(len(df_result), df_result)
    df_result["average_price"] = df_result["Money"] / df_result["Volume"]
    df_result = df_result.ffill().bfill()
    result = df_result[["average_price", "Volume"]].values.tolist()

    return result


class Norm_Average_Nprice_And_Mount_Whole_Day_5M(DBI_Generator_Base):
    ShapesM = [49,2]
    Input_Params = ["QZ_DF"]
    TitilesD =["Average_NPrice_5M_Whole_Day", "Total_Volume_5M_Whole_Day"]
    TypesD = ["NPrice","Volume"]
    def __init__(self):
        fab_5M_list=[925]
        for i in [93, 94, 95, 100, 101, 102, 103, 104, 105, 110, 111, 112,130, 131, 132, 133, 134, 135, 140, 141, 142, 143, 144, 145]:
            fab_5M_list=fab_5M_list+[i*10+idx*5 for idx in list(range(2))]
        self.param_norm_average_price_and_mount_5M = {
            "time_interval": 50,  # 1 minutes
            "result_len": 49,
            "result_should_contain": fab_5M_list
        }
    def Gen(self, inputs):
        result= _Norm_Average_Nprice_And_Mount_Gen_5M_Base(inputs,self.param_norm_average_price_and_mount_5M)
        self.Result_Check_Shape(result)
        return result
