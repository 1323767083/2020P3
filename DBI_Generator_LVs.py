from DBI_Generator_Base import DBI_Generator_Base
class Price_VS_Mount(DBI_Generator_Base):
    param_price_vs_mount = {"l_percent": [0, 0.25, 0.5, 0.75, 1.0]}

    ShapesM = [5]
    Input_Params = ["QZ_DF"]
    TitilesD = ["NPrice_0_Percent", "NPrice_25_Percent", "NPrice_50_Percent", "NPrice_75_Percent", "NPrice_100_Percent"]
    TypesD = ["NPrice", "NPrice", "NPrice", "NPrice", "NPrice"]

    def Gen(self, inputs):
        df = inputs[0]
        param = self.param_price_vs_mount
        df_acc = df[['Price', 'Volume']].groupby(["Price"]).sum().cumsum()
        total = df_acc.iloc[-1]
        result = [df_acc[df_acc["Volume"] >= int(total * percent)].index[0] for percent in param["l_percent"]]
        self.Result_Check_Shape(result)
        return result


class Buy_Dan(DBI_Generator_Base):
    ShapesM = [4]
    Input_Params = ["QZ_DF"]
    TitilesD = ["Buy_Da_Dan_Median_NPrice", "Buy_Da_Dan_Average_NPrice", "Buy_Xiao_Dan_Percent", "Buy_Da_Dan_Percent"]
    TypesD = ["NPrice", "NPrice", "Percent", "Percent"]

    da_dan_threadhold = 1000000
    xiao_dan_threadhold = 100000

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
    TitilesD = ["Sell_Da_Dan_Median_NPrice", "Sell_Da_Dan_Average_NPrice", "Sell_Xiao_Dan_Percent",
                "Sell_Da_Dan_Percent"]
    TypesD = ["NPrice", "NPrice", "Percent", "Percent"]
    da_dan_threadhold = 1000000
    xiao_dan_threadhold = 100000

    def Gen(self, inputs):
        df = inputs[0]
        result = df[["SaleOrderID", "Volume", "Money"]].groupby("SaleOrderID").sum()
        result["average_price_dan"] = result["Money"] / result["Volume"]  # ??
        total_money = result["Money"].sum()
        r_da_dan = result[result["Money"] > self.da_dan_threadhold]
        if len(r_da_dan) > 0:
            sell_da_dan_median_price = r_da_dan["average_price_dan"].median()
            sell_da_dan_total_money = r_da_dan["Money"].sum()
            sell_da_dan_total_volume = r_da_dan["Volume"].sum()
            sell_da_dan_percent = sell_da_dan_total_money / total_money
            sell_da_dan_average_price = sell_da_dan_total_money / sell_da_dan_total_volume
        else:
            sell_da_dan_median_price = 0.0  # ??
            sell_da_dan_average_price = 0.0  # ??
            sell_da_dan_percent = 0.0
        r_xiao_dan = result[result["Money"] < self.xiao_dan_threadhold]
        if len(r_xiao_dan) > 0:
            sell_xiao_dan_total_money = r_xiao_dan["Money"].sum()
            sell_xiao_dan_percent = sell_xiao_dan_total_money / total_money
        else:
            sell_xiao_dan_percent = 0.0
        result = [sell_da_dan_median_price, sell_da_dan_average_price, sell_xiao_dan_percent, sell_da_dan_percent]
        self.Result_Check_Shape(result)
        return result
