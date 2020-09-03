import os,pickle
from copy import deepcopy
import numpy as np
from DBI_Base import DBI_Base


class DBI_Generator_Base:
    da_dan_threadhold = 1000000
    xiao_dan_threadhold = 100000
    param_price_vs_mount = {"l_percent": [0, 0.25, 0.5, 0.75, 1.0]}
    param_potential_price_time_interval_list = [[93000, 96000], [100000, 103000], [103000, 106000],
                                                [106000, 113000], [130000, 133000], [133000, 140000],
                                                [140000, 143000], [143000, 150000]]
    param_norm_average_price_and_mount = {
        "time_interval": 1000,  # 10 minutes
        "result_len": 25,
        "result_should_contain": [92, 93, 94, 95, 100, 101, 102, 103, 104, 105, 110, 111, 112,
                                  130, 131, 132, 133, 134, 135, 140, 141, 142, 143, 144, 145]
    }
    ShapesM,Input_Params,TitilesD,TypesD = [],[],[],[]
    def Result_Check_Shape(self, result_item):
        if len(self.ShapesM) == 1:
            assert len(result_item) == self.ShapesM[0],"{0} {1} {2}".format(len(result_item), self.ShapesM, result_item)
        else:
            to_check = result_item
            for shape_item in self.ShapesM:
                assert len(to_check) == shape_item, "{0} {1} {2}".format(len(to_check), shape_item, to_check)
                to_check = to_check[0]
    def Gen(self, Inputs):
        return []
    def Get_TitleM(self):
        return self.__class__.__name__
    def Get_ShapesM(self):
        return self.ShapesM
    def Get_Input_Params(self):
        return self.Input_Params
    def Get_TitlesD(self):
        return self.TitilesD
    def Get_TypesD(self):
        return self.TypesD

class Price_VS_Mount(DBI_Generator_Base):
    ShapesM = [5]
    Input_Params = ["QZ_DF"]
    TitilesD = ["NPrice_0_Percent", "NPrice_25_Percent", "NPrice_50_Percent","NPrice_75_Percent", "NPrice_100_Percent"]
    TypesD = ["NPrice", "NPrice", "NPrice", "NPrice", "NPrice"]

    def Gen(self, inputs):
        df=inputs[0]
        param = self.param_price_vs_mount
        df_acc = df[['Price', 'Volume']].groupby(["Price"]).sum().cumsum()
        total = df_acc.iloc[-1]
        result = [df_acc[df_acc["Volume"] >= int(total * percent)].index[0] for percent in param["l_percent"]]
        self.Result_Check_Shape(result)
        return result

class Buy_Dan(DBI_Generator_Base):
    ShapesM = [4]
    Input_Params = ["QZ_DF"]
    TitilesD =["Buy_Da_Dan_Median_NPrice", "Buy_Da_Dan_Average_NPrice", "Buy_Xiao_Dan_Percent", "Buy_Da_Dan_Percent"]
    TypesD = ["NPrice","NPrice","Percent","Percent"]
    def Gen(self, inputs):
        df = inputs[0]
        result = df[["BuyOrderID", "Volume", "Money"]].groupby("BuyOrderID").sum()
        result["average_price_dan"] = result["Money"] / result["Volume"]
        total_money = result["Money"].sum()
        r_da_dan = result[result["Money"] > self.da_dan_threadhold]
        if len(r_da_dan) > 0:
            buy_da_dan_median_price = r_da_dan["average_price_dan"].median()
            buy_da_dan_total_money = r_da_dan["Money"].sum()
            buy_da_dan_total_volume = r_da_dan["Volume"].sum()
            buy_da_dan_average_price = buy_da_dan_total_money / buy_da_dan_total_volume
            buy_da_dan_percent = buy_da_dan_total_money / total_money
        else:
            buy_da_dan_median_price = 0.0  # ?
            buy_da_dan_average_price = 0.0  # ?
            buy_da_dan_percent = 0.0
        r_xiao_dan = result[result["Money"] < self.xiao_dan_threadhold]
        if len(r_xiao_dan) > 0:
            buy_xiao_dan_total_money = r_xiao_dan["Money"].sum()
            buy_xiao_dan_percent = buy_xiao_dan_total_money / total_money
        else:
            buy_xiao_dan_percent = 0.0
        result = [buy_da_dan_median_price, buy_da_dan_average_price, buy_xiao_dan_percent, buy_da_dan_percent]
        self.Result_Check_Shape(result)
        return result
class Sell_Dan(DBI_Generator_Base):
    ShapesM = [4]
    Input_Params = ["QZ_DF"]
    TitilesD =["Sell_Da_Dan_Median_NPrice", "Sell_Da_Dan_Average_NPrice", "Sell_Xiao_Dan_Percent", "Sell_Da_Dan_Percent"]
    TypesD = ["NPrice","NPrice","Percent","Percent"]
    def Gen(self, inputs):
        df = inputs[0]
        result = df[["SaleOrderID", "Volume", "Money"]].groupby("SaleOrderID").sum()
        result["average_price_dan"] = result["Money"] / result["Volume"]  # ??
        total_money = result["Money"].sum()
        r_da_dan=result[result["Money"] > self.da_dan_threadhold]
        if len(r_da_dan)>0:
            sell_da_dan_median_price = r_da_dan["average_price_dan"].median()
            sell_da_dan_total_money = r_da_dan["Money"].sum()
            sell_da_dan_total_volume = r_da_dan ["Volume"].sum()
            sell_da_dan_percent = sell_da_dan_total_money / total_money
            sell_da_dan_average_price = sell_da_dan_total_money / sell_da_dan_total_volume
        else:
            sell_da_dan_median_price = 0.0  # ??
            sell_da_dan_average_price = 0.0  # ??
            sell_da_dan_percent = 0.0
        r_xiao_dan=result[result["Money"] < self.xiao_dan_threadhold]
        if len(r_xiao_dan)>0:
            sell_xiao_dan_total_money = r_xiao_dan["Money"].sum()
            sell_xiao_dan_percent = sell_xiao_dan_total_money / total_money
        else:
            sell_xiao_dan_percent = 0.0
        result = [sell_da_dan_median_price, sell_da_dan_average_price, sell_xiao_dan_percent, sell_da_dan_percent]
        self.Result_Check_Shape(result)
        return result

#class _Potential_Nprice_Base:
def _Gen_Potential_Nprice_Base(inputs, time_interval_list):
    df = inputs[0]
    for interval in time_interval_list:
        df_acc = df[(df["Time"] >= interval[0]) & (df["Time"] < interval[1])][["Price", "Volume"]].groupby(
            "Price").sum().cumsum()
        if len(df_acc) > 0:
            total_amount = df_acc["Volume"].iloc[-1]
            return [True, df_acc[df_acc["Volume"] >= 0.5 * total_amount].index[0]]
    else:
        return [False, 0.0]

class Potential_Nprice_930(DBI_Generator_Base):
    ShapesM = [2]
    Input_Params = ["QZ_DF"]
    TitilesD =["Flag_Tradable","Potential_NPrice_930"]
    TypesD = ["Flag_Tradable","NPrice_Not_Normal"]
    def Gen(self, inputs):
        result= _Gen_Potential_Nprice_Base(inputs, self.param_potential_price_time_interval_list)
        self.Result_Check_Shape(result)
        return result

class Potential_Nprice_1300(DBI_Generator_Base):
    ShapesM = [2]
    Input_Params = ["QZ_DF"]
    TitilesD =["Flag_Tradable","Potential_NPrice_1300"]
    TypesD = ["Flag_Tradable","NPrice_Not_Normal"]
    def Gen(self, inputs):
        result= _Gen_Potential_Nprice_Base(inputs, self.param_potential_price_time_interval_list[4:])
        self.Result_Check_Shape(result)
        return result

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
    def Gen(self, inputs):
        param={}
        param["time_interval"] = self.param_norm_average_price_and_mount["time_interval"]
        param["result_len"] = self.ShapeFirstDim
        param["result_should_contain"] = self.param_norm_average_price_and_mount["result_should_contain"][-self.ShapeFirstDim:]
        result= _Norm_Average_Nprice_And_Mount_Gen_Base(inputs,param)
        self.Result_Check_Shape(result)
        return result

class HFQ_Ratio(DBI_Generator_Base):
    ShapesM = [1]
    Input_Params = ["HFQ_DF","DateI"]
    TitilesD =["HFQ_Ratio"]
    TypesD = ["Ratio"]
    def Gen(self,inputs):
        df_hfq, date_I=inputs
        df_result=df_hfq[df_hfq["date"]==str(date_I)]
        assert len(df_result)==1,("{0} does not find in hfq df ".format(date_I))
        result= df_result["coefficient_fq"].values.tolist()
        self.Result_Check_Shape(result)
        return result

class Exchange_Ratios(DBI_Generator_Base):
    ShapesM = [2]
    Input_Params = ["HFQ_DF","DateI"]
    TitilesD =["Exchange_Ratio_Tradable_Part","Exchange_Ratio_Whole"]
    TypesD = ["Ratio", "Ratio"]
    def Gen(self,inputs):
        df_hfq, date_I=inputs
        df_result=df_hfq[df_hfq["date"]==str(date_I)]
        assert len(df_result)==1,"{0} does not find in hfq df ".format(date_I)
        result= df_result[["exchange_ratio_for_tradable_part","exchange_ratio_for_whole"]].values.tolist()[0]
        self.Result_Check_Shape(result)
        return result

class DateI(DBI_Generator_Base):
    ShapesM = [1]
    Input_Params = ["DateI"]
    TitilesD =["DateI"]
    TypesD = ["DateI"]
    def Gen(self,inputs):
        DateI =inputs[0]
        result=[DateI]
        self.Result_Check_Shape(result)
        return result


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

    def IsTinpai(self, Stock, DayI):
        fnwp = self.get_DBI_hfq_fnwp(Stock)
        if not os.path.exists(fnwp):
            return False, "", "HFQ File Not Found****{0}".format(fnwp)
        flag, df, mess = self.get_hfq_df(fnwp)
        if not flag:
            return False, "", mess
        return True, df[df["date"] == str(DayI)].empty, "Success"

    def get_Tinpai_item(self, DBITypeD,DayI):
        assert DBITypeD in ["NPrice", "Percent", "Volume", "Ratio", "DateI","NPrice_Not_Normal","Flag_Tradable","NPrice_Not_Normal"],\
            "{0} {1}".format( DBITypeD,DayI)
        if DBITypeD in ["NPrice", "Volume"]:
            return 0
        elif DBITypeD in ["Percent", "Ratio"]:
            return 0
        elif DBITypeD in ["NPrice_Not_Normal"]:
            return 0
        elif DBITypeD in ["DateI"]:
            return DayI
        elif DBITypeD in ["NPrice_Not_Normal"]:
            return False

    def get_oneday_tinpai(self,Stock, DayI):
        fnwp=self.get_DBI_data_fnwp(Stock, DayI)
        if os.path.exists(fnwp):
            DBIdata=pickle.load(open(fnwp,"rb"))
            return DBIdata
        DBIdata=[]
        for DBIShapeM, DBITypesD in zip(self.DBIShapesM,self.DBITypesD):
            if len(DBIShapeM)==1:
                DBIdata.append([self.get_Tinpai_item(DBITypeD,DayI) for DBITypeD in DBITypesD])
            elif len(DBIShapeM)==2:
                DBIdata.append([[self.get_Tinpai_item(DBITypeD,DayI) for DBITypeD in DBITypesD] for _ in range(DBIShapeM[0])])
            else:
                assert False, "Not Support shape length more than 2 {0}".format(DBIShapeM)
        pickle.dump(DBIdata, open(fnwp, 'wb'))
        return DBIdata
    ##DBI data generator
    def get_oneday(self,  df_qz, df_hfq, dayI, stock, param):
        fnwp=self.get_DBI_data_fnwp(stock, dayI)
        if os.path.exists(fnwp):
            result_L=pickle.load(open(fnwp,"rb"))
            return result_L
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
        pickle.dump(result_L, open(fnwp, 'wb'))
        return result_L

    def generate_DBI_day(self, Stock, DayI):
        if DayI>self.Raw_Normal_Lumpsum_EndDayI:
            logfnwp = self.get_DBI_Update_Log_HFQ_Index_fnwp(DayI)
            if not os.path.exists(logfnwp):
                Error_Mess="Need Update Index and HFQ and decompress QZ for {0} first".format(DayI)
                print (Error_Mess)
                return False, Error_Mess
        logfnwp = self.get_DBI_log_fnwp(Stock)

        DBI_HFQ_fnwp=self.get_DBI_hfq_fnwp(Stock)
        if not os.path.exists(DBI_HFQ_fnwp):
            Error_Mess= "DBI HFQ File Not Found****{0}".format(DBI_HFQ_fnwp)
            self.log_append_keep_new([[False, DayI,Error_Mess]], logfnwp, ["Result", "Date", "Message"])
            return False, Error_Mess
        hfq_flag,hfq_df, hfq_mess=self.get_hfq_df(DBI_HFQ_fnwp)
        if not hfq_flag:
            self.log_append_keep_new([[False, DayI,hfq_mess]], logfnwp, ["Result", "Date", "Message"])
            return False, hfq_mess
        print ("DBI generate {0} {1}".format(Stock,DayI))
        if hfq_df[hfq_df["date"] == str(DayI)].empty: # Tinpai
            self.get_oneday_tinpai(Stock, DayI)
            self.log_append_keep_new([[True, DayI, "Tinpai"]], logfnwp, ["Result", "Date", "Message"])
            return True, "Tinpai"
        qz_flag, qz_df, qz_mess = self.IRD.get_qz_df_inteface( Stock, DayI)
        if not qz_flag:
            self.log_append_keep_new([[False,DayI,qz_mess]], logfnwp, ["Result", "Date", "Message"])
            return False,qz_mess
        self.get_oneday(qz_df, hfq_df, DayI, Stock,self.TypeDefinition)
        self.log_append_keep_new([[True, DayI, "Success"]], logfnwp, ["Result", "Date", "Message"])
        return True, "Success"



# 7z a -r 20200703.7z 2020-07-03/