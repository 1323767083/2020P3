from DBI_Generator_Base import DBI_Generator_Base
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

    param_potential_price_time_interval_list = [[93000, 96000], [100000, 103000], [103000, 106000],
                                                [110000, 113000], [130000, 133000], [133000, 136000],
                                                [140000, 143000], [143000, 146000]]

    def Gen(self, inputs):
        result= _Gen_Potential_Nprice_Base(inputs, self.param_potential_price_time_interval_list)
        self.Result_Check_Shape(result)
        return result

class param_M5:
    param_potential_price_time_interval_list = [[93000, 96000], [100000, 103000], [103000, 106000],
                                                [110000, 113000], [130000, 133000], [133000, 136000],
                                                [140000, 143000], [143000, 146000]]
    Final_interval_list=[]
    step=500
    for start, end in param_potential_price_time_interval_list:
        a=[]
        for itemStart in list(range(start, end, step)):
            a.append([itemStart,itemStart+step])
        Final_interval_list.extend(a)

class Potential_Nprice_930M5(DBI_Generator_Base,param_M5):
    ShapesM = [2]
    Input_Params = ["QZ_DF"]
    TitilesD =["Flag_Tradable","Potential_NPrice_1300"]
    TypesD = ["Flag_Tradable","NPrice_Not_Normal"]


    def Gen(self, inputs):
        result= _Gen_Potential_Nprice_Base(inputs, self.Final_interval_list)
        self.Result_Check_Shape(result)
        return result


class Potential_Nprice_1300M5(DBI_Generator_Base,param_M5):
    ShapesM = [2]
    Input_Params = ["QZ_DF"]
    TitilesD =["Flag_Tradable","Potential_NPrice_1300"]
    TypesD = ["Flag_Tradable","NPrice_Not_Normal"]


    def Gen(self, inputs):
        result= _Gen_Potential_Nprice_Base(inputs, self.Final_interval_list)
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
        assert len(df_result)==1,"{0} does not find in hfq df  {1}".format(date_I,df_result)
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
