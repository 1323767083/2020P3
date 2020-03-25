from sklearn import preprocessing
from copy import deepcopy

import os,h5py, sys
import config as sc
import pandas as pd
import numpy as np
from data_common import API_qz_from_file,API_HFQ_from_file, API_index_from_file, ginfo_one_stock,API_qz_data_source_related
from data_common import API_SH_SZ_total_sl,API_trade_date,API_SH_sl,API_G_IPO_sl,exclude_stock_list,keyboard_input
from sklearn import preprocessing



#File handler
class FH_summary_data_1stock:
    def __init__(self, data_name, stock):
        self.data_name=data_name
        self.stock=stock
    def get_dump_fnwp(self,stock):
        stock_number=int(stock[2:])
        stock_head=stock[0:2]
        stock_divide_dir="{0}__{1}".format(stock_head,stock_number % 10)
        working_dir=sc.base_dir_RL_data
        sub_dir_list=[self.data_name,stock_divide_dir]
        for sub_dir in sub_dir_list:
            working_dir=os.path.join(working_dir,sub_dir)
            if not os.path.exists(working_dir): os.mkdir(working_dir)
        fnwp=os.path.join(working_dir,"{0}_Summary_raw.h5py".format(stock))
        return fnwp

    def save_data(self, data):
        num_periods=len(data.l_np_date_s)
        assert len(data.l_np_price_vs_mount) ==num_periods
        assert len(data.l_np_sell_dan) ==num_periods
        assert len(data.l_np_buy_dan) ==num_periods
        assert len(data.l_np_potential_price) ==num_periods
        assert len(data.l_np_norm_average_price_and_mount) ==num_periods
        assert len(data.l_np_hfq_ratio) ==num_periods

        fnwp = self.get_dump_fnwp(self.stock)
        with h5py.File(fnwp, "w") as hf:
            for idx in xrange(num_periods):
                hf_wg=hf.create_group("period_{0}".format(idx))
                hf_wg.create_dataset("np_date_s",                      data=data.l_np_date_s[idx])
                hf_wg.create_dataset("np_price_vs_mount",              data=data.l_np_price_vs_mount[idx])
                hf_wg.create_dataset("np_sell_dan",                    data=data.l_np_sell_dan[idx])
                hf_wg.create_dataset("np_buy_dan",                     data=data.l_np_buy_dan[idx])
                hf_wg.create_dataset("np_norm_average_price_and_mount",data=data.l_np_norm_average_price_and_mount[idx])
                hf_wg.create_dataset("np_potential_price",             data=data.l_np_potential_price[idx])
                hf_wg.create_dataset("np_hfq_ratio",                   data=data.l_np_hfq_ratio[idx])

    def load_all_data(self):
        data=DS_summary_data_1stock()
        fnwp = self.get_dump_fnwp(self.stock)
        with h5py.File(fnwp, "r") as hf:
            l_period=hf.keys()
            for period in l_period:
                hf_wg=hf[period]
                assert isinstance(hf_wg, h5py.Group)
                data.l_np_date_s.append(hf_wg["np_date_s"][:])
                data.l_np_price_vs_mount.append(hf_wg["np_price_vs_mount"][:])
                data.l_np_sell_dan.append(hf_wg["np_sell_dan"][:])
                data.l_np_buy_dan.append(hf_wg["np_buy_dan"][:])
                data.l_np_norm_average_price_and_mount.append(hf_wg["np_norm_average_price_and_mount"][:])
                data.l_np_potential_price.append(hf_wg["np_potential_price"][:])
                data.l_np_hfq_ratio.append(hf_wg["np_hfq_ratio"][:])
        return data

    def load_one_data(self,period_num):
        data=DS_summary_data_1stock()
        fnwp = self.get_dump_fnwp(self.stock)
        with h5py.File(fnwp, "r") as hf:
            l_period=hf.keys()
            assert period_num<=len(l_period)-1
            hf_wg = hf["period_{0}".format(period_num)]
            assert isinstance(hf_wg, h5py.Group)
            data.l_np_date_s.append(hf_wg["np_date_s"][:])
            data.l_np_price_vs_mount.append(hf_wg["np_price_vs_mount"][:])
            data.l_np_sell_dan.append(hf_wg["np_sell_dan"][:])
            data.l_np_buy_dan.append(hf_wg["np_buy_dan"][:])
            data.l_np_norm_average_price_and_mount.append(hf_wg["np_norm_average_price_and_mount"][:])
            data.l_np_potential_price.append(hf_wg["np_potential_price"][:])
            data.l_np_hfq_ratio.append(hf_wg["np_hfq_ratio"][:])
        return data

    def get_total_period_num(self):
        fnwp = self.get_dump_fnwp(self.stock)
        with h5py.File(fnwp, "r") as hf:
            l_period=hf.keys()
        return len(l_period)

    def check_data_avalaible(self):
        fnwp = self.get_dump_fnwp(self.stock)
        return True if os.path.exists(fnwp) else False
#prepare intermediate data
class DS_summary_data_1stock:
    def __init__(self):
        self.l_np_date_s = []
        self.l_np_price_vs_mount = []
        self.l_np_sell_dan = []
        self.l_np_buy_dan = []
        self.l_np_potential_price = []
        self.l_np_norm_average_price_and_mount = []
        #self.l_np_hfq_amount = []
        self.l_np_hfq_ratio = []
class G_summary_data_1stock1day:
    def __init__(self):
        self.param_price_vs_mount = {"l_percent": [0, 0.25, 0.5, 0.75, 1.0]}
        self.param_norm_average_price_and_mount = {
            "time_interval": 1000,  # 10 minutes
            "result_len": 25,
            "result_should_contain": [92, 93, 94, 95, 100, 101, 102, 103, 104, 105, 110, 111, 112,
                                      130, 131, 132, 133, 134, 135, 140, 141, 142, 143, 144, 145]
        }
        self.df_template_25_row=pd.DataFrame(columns=["Time", "Price", "Volume", "SaleOrderVolume", "BuyOrderVolume",
                                    "Type", "SaleOrderID", "SaleOrderPrice", "BuyOrderID", "BuyOrderPrice", "Money"])
        self.param_potential_price_time_interval_list = [[93000, 96000], [100000, 103000], [103000, 106000],
                                                         [106000, 113000], [130000, 133000], [133000, 140000],
                                                         [140000, 143000], [143000, 150000],[92500,92559]]
        #[92500,92559] is add for 20170801 only have 9:25 data no further trading in the day like SH600679
        #self.da_dan_threadhold = 1000000
        #self.xiao_dan_threadhold = 100000
        self.da_dan_threadhold = sc.RL_da_dan_threadhold
        self.xiao_dan_threadhold =sc.RL_xiao_dan_threadhold

    def price_vs_mount(self, df):
        param = self.param_price_vs_mount
        df_acc = df[['Price', 'Volume']].groupby(["Price"]).sum().cumsum()
        total = df_acc.iloc[-1]
        result = [df_acc[df_acc["Volume"] >= int(total * percent)].index[0] for percent in param["l_percent"]]
        return np.expand_dims(np.array(result), axis=0)

    def buy_dan(self, df):
        da_dan_threadhold = self.da_dan_threadhold
        xiao_dan_threadhold = self.xiao_dan_threadhold

        result=df[["BuyOrderID", "Volume", "Money"]].groupby("BuyOrderID").sum()
        result["average_price_dan"] = result["Money"] / result["Volume"]
        total_money = result["Money"].sum()
        r_da_dan=result[result["Money"] > da_dan_threadhold]
        if len(r_da_dan)>0:
            buy_da_dan_median_price = r_da_dan["average_price_dan"].median()
            buy_da_dan_total_money = r_da_dan["Money"].sum()
            buy_da_dan_total_volume = r_da_dan["Volume"].sum()
            buy_da_dan_average_price = buy_da_dan_total_money / buy_da_dan_total_volume
            buy_da_dan_percent = buy_da_dan_total_money / total_money
        else:
            buy_da_dan_median_price = 0.0  # ?
            buy_da_dan_average_price = 0.0  # ?
            buy_da_dan_percent = 0.0
        r_xiao_dan=result[result["Money"]  < xiao_dan_threadhold]
        if len(r_xiao_dan)>0:
            buy_xiao_dan_total_money = r_xiao_dan["Money"].sum()
            buy_xiao_dan_percent = buy_xiao_dan_total_money / total_money
            # buy_xiao_dan_percent = buy_xiao_dan_total_money / total_money if total_money != 0 else 0.0
        else:
            buy_xiao_dan_percent = 0.0

        result = [buy_da_dan_median_price, buy_da_dan_average_price, buy_xiao_dan_percent, buy_da_dan_percent]
        return np.expand_dims(np.array(result), axis=0)

    def sell_dan(self, df):
        da_dan_threadhold = self.da_dan_threadhold
        xiao_dan_threadhold = self.xiao_dan_threadhold
        result = df[["SaleOrderID", "Volume", "Money"]].groupby("SaleOrderID").sum()

        result["average_price_dan"] = result["Money"] / result["Volume"]  # ??
        total_money = result["Money"].sum()

        r_da_dan=result[result["Money"] > da_dan_threadhold]
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

        r_xiao_dan=result[result["Money"] < xiao_dan_threadhold]
        if len(r_xiao_dan)>0:
            sell_xiao_dan_total_money = r_xiao_dan["Money"].sum()
            sell_xiao_dan_percent = sell_xiao_dan_total_money / total_money
        else:
            sell_xiao_dan_percent = 0.0

        result = [sell_da_dan_median_price, sell_da_dan_average_price, sell_xiao_dan_percent, sell_da_dan_percent]
        return np.expand_dims(np.array(result), axis=0)

    def potential_price(self, df):

        # time_interval_list=[[93000,96000],[100000,103000],[103000,106000],[106000,113000],
        #                    [130000,133000],[133000,140000],[140000,143000],[143000,150000],[92500,92559]]
        time_interval_list = self.param_potential_price_time_interval_list
        # 15min
        for interval in time_interval_list:
            df_acc = df[(df["Time"] >= interval[0]) & (df["Time"] < interval[1])][["Price", "Volume"]].groupby(
                "Price").sum().cumsum()
            if len(df_acc) > 0:
                total_amount = df_acc["Volume"].iloc[-1]
                return df_acc[df_acc["Volume"] >= 0.5 * total_amount].index[0]

        raise ValueError("can not find the price,{0}".format(df))

    def norm_average_price_and_mount(self, df_src):
        # param={
        #    "time_interval":1000,    #10 minutes
        #    "result_len":25,
        #    "result_should_contain": [92,93,94,95,100,101,102,103,104,105, 110,111,112,
        #                              130,131,132,133,134,135,140,141,142,143,144,145]
        # }
        param = self.param_norm_average_price_and_mount
        df = deepcopy(df_src)
        df["Time"] = df["Time"] / param["time_interval"]  # ten minutes
        df["Time"] = df["Time"].astype(int)

        np_time = df["Time"].values
        np.unique(np_time)
        set_diff = set(param["result_should_contain"]) - set(np_time)
        if len(set_diff) > 0:
            for item in set_diff:
                df.loc[len(df)] = [item, np.nan, 0,0,0,"B", 0,0.0,0, 0.0,0.0]
            df.sort_values(['Time'], ascending=True, inplace=True)
            df = df.ffill().bfill()

        df_result = df[["Time", "Volume", "Money"]].groupby(["Time"]).sum()
        assert len(df_result) == param["result_len"], "{0} _{1}".format(len(df_result), df_result)
        df_result["average_price"] = df_result["Money"] / df_result["Volume"]
        #df_result.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_result = df_result.ffill().bfill()
        result = df_result[["average_price", "Volume"]].values


        for idx in range(result.shape[1]):
            scaler = preprocessing.StandardScaler().fit(result[:, idx].reshape(-1, 1))
            result[:, idx] = scaler.transform(result)[:, idx]

        return np.expand_dims(result, axis=0)
class G_summary_data_1stock:
    #def __init__(self,data_name,stock,skip_days=100,least_length=23):
    def __init__(self, data_name, stock):
        self.data_name=data_name
        self.stock=stock
        self.skip_days=sc.RL_data_skip_days
        self.least_length=sc.RL_data_least_length

        self.flag_prepare_data_ready = False
        self.i_fh=FH_summary_data_1stock(self.data_name,self.stock)

        if self.i_fh.check_data_avalaible():
            self.data=self.i_fh.load_all_data()
            if len(self.data.l_np_date_s)!=0:
                self.flag_prepare_data_ready = True
                return
            else:
                raise ValueError ("no data exists in the existing h5py file for {0} ".format(self.stock))

        self.i_ginform = ginfo_one_stock(self.stock)
        self.i_view = G_summary_data_1stock1day()
        self.data=DS_summary_data_1stock()
        self.i_qz=API_qz_from_file()
        data_spource_type=API_qz_data_source_related().check_according_data_name(self.data_name)
        if data_spource_type=="_V1_":
            self.prepare_data=self.prepare_data_V1
        elif data_spource_type=="_V2_":
            self.prepare_data=self.prepare_data_V2


    def _prepare_data_add_to_list(self,data):
        l_date_s, l_price_vs_mount, l_sell_dan, l_buy_dan, l_potential_price, l_norm_average_price_and_mount, l_hfq_ratio = data
        self.data.l_np_date_s.append(np.array(l_date_s))
        self.data.l_np_price_vs_mount.append(np.concatenate(l_price_vs_mount, axis=0))
        self.data.l_np_sell_dan.append(np.concatenate(l_sell_dan, axis=0))
        self.data.l_np_buy_dan.append(np.concatenate(l_buy_dan, axis=0))
        self.data.l_np_potential_price.append(np.array(l_potential_price))
        self.data.l_np_norm_average_price_and_mount.append(np.concatenate(l_norm_average_price_and_mount, axis=0))
        self.data.l_np_hfq_ratio.append(np.array(l_hfq_ratio))


    def prepare_data_V1(self):
        if self.flag_prepare_data_ready:
            return True
        periods=API_qz_data_source_related().get_valid_periods("_V1_",self.stock)
        #self.prepare_data_period = qz_extract_data().stock_avail_periods(self.stock, self.skip_days, self.least_length)
        if len(periods)==0:
            return False
        for period in periods:
            l_date_s, l_price_vs_mount, l_sell_dan, l_buy_dan, l_potential_price, l_norm_average_price_and_mount, l_hfq_ratio \
                = [], [], [], [], [], [], []
            for date_s in period:
                print "\thandling {0} {1} summary data".format(self.stock, date_s)
                if not self.i_ginform.check_not_tinpai(date_s):
                    continue
                df = self.i_qz.get_df_qz(date_s, self.stock)

                l_date_s.append(date_s)
                l_price_vs_mount.append(self.i_view.price_vs_mount(df))
                l_sell_dan.append(self.i_view.sell_dan(df))
                l_buy_dan.append(self.i_view.buy_dan(df))
                l_potential_price.append(self.i_view.potential_price(df))
                l_norm_average_price_and_mount.append(self.i_view.norm_average_price_and_mount(df))
                l_hfq_ratio.append(self.i_ginform.hfq_ratio(date_s))
            if len(l_date_s)==0:
                continue
            self._prepare_data_add_to_list([l_date_s, l_price_vs_mount, l_sell_dan, l_buy_dan,
                                                       l_potential_price, l_norm_average_price_and_mount, l_hfq_ratio])
        if len(self.data.l_np_date_s)==0:
            return False
        self.flag_prepare_data_ready = True
        self.i_fh.save_data(self.data)
        return True


    def prepare_data_V2(self):
        if self.flag_prepare_data_ready:
            return True
        periods = API_qz_data_source_related().get_valid_periods("_V2_",self.stock)
        l_date_s,l_price_vs_mount,l_sell_dan,l_buy_dan,l_potential_price,l_norm_average_price_and_mount,l_hfq_ratio\
            =[],[],[],[],[],[],[]
        assert len(periods)==1
        period = periods[0]
        if len(period)==0:
            print "{0} does not have valid data len(period) is 0".format(self.stock)
            exclude_stock_list(self.data_name).add_to_exlude_list(self.stock, reason="no_data_in_specified_period")
            return False

        for date_s in period:
            print "\thandling {0} {1} summary data".format(self.stock, date_s)
            if not self.i_ginform.check_not_tinpai(date_s):
                continue
            df = self.i_qz.get_df_qz(date_s, self.stock)
            if len(df)==0:
                if len(l_date_s) != 0:
                    print "{0} period start {1} to {2} data created".format(self.stock,l_date_s[0], l_date_s[-1])
                    self._prepare_data_add_to_list([l_date_s, l_price_vs_mount, l_sell_dan, l_buy_dan,
                                                       l_potential_price, l_norm_average_price_and_mount, l_hfq_ratio])
                    l_date_s, l_price_vs_mount, l_sell_dan, l_buy_dan, l_potential_price, \
                    l_norm_average_price_and_mount, l_hfq_ratio = [], [], [], [], [], [], []
                else:
                    continue
            else:
                l_date_s.append(date_s)
                l_price_vs_mount.append(self.i_view.price_vs_mount(df))
                l_sell_dan.append(self.i_view.sell_dan(df))
                l_buy_dan.append(self.i_view.buy_dan(df))
                l_potential_price.append(self.i_view.potential_price(df))
                l_norm_average_price_and_mount.append(self.i_view.norm_average_price_and_mount(df))
                l_hfq_ratio.append(self.i_ginform.hfq_ratio(date_s))
        else:
            if len(l_date_s) != 0:
                print "{0} last period start {1} to {2} data created".format(self.stock,l_date_s[0], l_date_s[-1])
                #continue
                self._prepare_data_add_to_list([l_date_s, l_price_vs_mount, l_sell_dan, l_buy_dan,
                                                l_potential_price, l_norm_average_price_and_mount, l_hfq_ratio])

        if len(self.data.l_np_date_s)==0:
            exclude_stock_list(self.data_name).add_to_exlude_list(self.stock, reason="no_data_in_specified_period")
            return False
        self.flag_prepare_data_ready = True
        self.i_fh.save_data(self.data)
        return True

class FH_addon_data_1stock:
    def __init__(self, data_name):
        self.data_name=data_name
    #common
    def __get_fnwp(self, data_name,stock):
        stock_number = int(stock[2:])
        stock_head = stock[0:2]
        stock_divide_dir = "{0}__{1}".format(stock_head, stock_number % 10)
        working_dir = sc.base_dir_RL_data
        sub_dir_list = [data_name, stock_divide_dir]
        for sub_dir in sub_dir_list:
            working_dir = os.path.join(working_dir, sub_dir)
            if not os.path.exists(working_dir): os.mkdir(working_dir)
        fnwp = os.path.join(working_dir, "{0}_RL_addon_input.h5py".format(stock))
        return fnwp

    #data name API
    def _get_fnwp(self, stock):
        return self.__get_fnwp(self.data_name, stock)

    def _check_avalaible(self, stock):
        fnwp = self._get_fnwp(stock)
        return os.path.exists(fnwp)

    def _save(self, stock, l_np_date_s, l_np_stock_SwhV1, l_np_stock_S20V20, l_np_syuan_SwhV20,
                           np_scale_param,np_start_end_date):
        num_periods = len(l_np_date_s)
        assert len(l_np_stock_SwhV1) == num_periods
        assert len(l_np_stock_S20V20) == num_periods
        assert len(l_np_syuan_SwhV20) == num_periods
        fnwp = self._get_fnwp(stock)
        with h5py.File(fnwp, "w") as hf:
            hg_xd=hf.create_group("__param")
            hg_xd.create_dataset("np_scale_param", data=np_scale_param)
            hg_xd.create_dataset("np_start_end_date", data=np_start_end_date)
            for idx in xrange(num_periods):
                hf_wg = hf.create_group("period_{0}".format(idx))
                hf_wg.create_dataset("np_date_s", data=l_np_date_s[idx])
                hf_wg.create_dataset("np_stock_SwhV1", data=l_np_stock_SwhV1[idx])
                hf_wg.create_dataset("np_stock_S20V20", data=l_np_stock_S20V20[idx])
                hf_wg.create_dataset("np_syuan_SwhV20", data=l_np_syuan_SwhV20[idx])

    def _load(self, stock):
        l_np_date_s=[]
        l_np_stock_SwhV1 = []
        l_np_stock_S20V20 = []
        l_np_syuan_SwhV20 = []
        fnwp = self._get_fnwp(stock)
        with h5py.File(fnwp, "r") as hf:
            hg_xd = hf["__param"]
            np_scale_param=hg_xd["np_scale_param"][:]
            np_start_end_date=hg_xd["np_start_end_date"][:]
            l_period_raw=hf.keys()
            l_period = [item for item in l_period_raw if not str(item).startswith("__")]
            for idx,period in enumerate(l_period):
                hf_wg=hf[period]
                l_np_date_s.append(hf_wg["np_date_s"][:])
                l_np_stock_S20V20.append(hf_wg["np_stock_S20V20"][:])
                l_np_stock_SwhV1.append(hf_wg["np_stock_SwhV1"][:])
                l_np_syuan_SwhV20.append(hf_wg["np_syuan_SwhV20"][:])
        return l_np_date_s, l_np_stock_SwhV1, l_np_stock_S20V20, l_np_syuan_SwhV20, np_scale_param,np_start_end_date
class G_addon_data_1stock(FH_addon_data_1stock):
    def __init__(self, data_name,refer_data_name):
        FH_addon_data_1stock.__init__(self,data_name)
        self.refer_data_name =   refer_data_name
        self.flag_no_refer    =   True if self.refer_data_name=="" else False

        self.data_source_type =API_qz_data_source_related().check_according_data_name(self.data_name)
        self.td = API_trade_date().np_date_s

    #refer data name API
    def _get_refer_fnwp(self, stock):
        assert not self.flag_no_refer
        return self.__get_fnwp(self.refer_data_name, stock)

    def _load_refer_param(self,stock):
        assert not self.flag_no_refer
        fnwp = self._get_refer_fnwp(stock)
        if not os.path.exists(fnwp):
            return False, "", ""
        with h5py.File(fnwp, "r") as hf:
            hg_xd = hf["__param"]
            np_scale_param=hg_xd["np_scale_param"][:]
            np_start_end_date=hg_xd["np_start_end_date"][:]
        return True, np_scale_param,np_start_end_date

    def _get_refer_data_StandardScalers(self, stock):
        assert not self.flag_no_refer
        flag_result,np_scale_param, np_start_end_date=self._load_refer_param(stock)
        if not flag_result:
            return False, "","","",""
        stock_Swh_mean_, stock_Swh_scale_, Syuan_Swh_mean_, Syuan_Swh_scale_ =np_scale_param
        stock_Swh=preprocessing.StandardScaler()
        stock_Swh.mean_=stock_Swh_mean_
        stock_Swh.scale_=stock_Swh_scale_

        Syuan_Swh=preprocessing.StandardScaler()
        Syuan_Swh.mean_=Syuan_Swh_mean_
        Syuan_Swh.scale_=Syuan_Swh_scale_

        return True, stock_Swh, Syuan_Swh, np_scale_param, np_start_end_date


    def prepare_data(self,stock):

        #periods = get_periods(self.data_type, stock).periods

        if self._check_avalaible(stock):
            l_np_date_s, l_np_stock_SwhV1, l_np_stock_S20V20, l_np_syuan_SwhV20, np_scale_param, np_start_end_date=self._load(stock)
            return True, l_np_date_s, l_np_stock_SwhV1, l_np_stock_S20V20, l_np_syuan_SwhV20, np_scale_param, np_start_end_date

        periods = API_qz_data_source_related().get_valid_periods(self.data_source_type, stock)
        if len(periods[0])==0:
            exclude_stock_list(self.data_name).add_to_exlude_list(stock, reason="no_data_in_specified_period")
            return False,"","","","","",""
        first_date=periods[0][0]
        last_date = periods[-1][-1]
        start_date=self.td[self.td<=first_date][-20]
        end_date=self.td[self.td>=last_date][1]
        dfsi = API_HFQ_from_file().get_df_HFQ(stock)
        dfsi=  dfsi[(dfsi["date"]>=start_date) &(dfsi["date"]<=end_date)]
        tdsi = self.td[(self.td >= start_date) & (self.td <= end_date)]
        #periods from get_valid_periods already is trand days include tingpai
        #period=periods[0]
        dfti = pd.DataFrame(tdsi, columns=["date"])
        dfs  = pd.merge(dfsi, dfti, how='outer', left_on=['date'], right_on=["date"])
        for item in ["code", "open_price", "highest_price", "lowest_price", "close_price", "coefficient_fq"]:
            dfs[item].ffill(inplace=True)
        for item in ["amount_gu", "amount_yuan", "exchange_ratio_for_tradable_part", "exchange_ratio_for_whole"]:
            dfs[item].fillna(0, inplace=True)

        dfs.sort_values(["date"], inplace=True)
        dfs.reset_index(drop=True,inplace=True)

        if self.flag_no_refer:
            stock_Swh = preprocessing.StandardScaler().fit(dfs["open_price"].values.reshape(-1, 1))
            Syuan_Swh = preprocessing.StandardScaler().fit(dfs["amount_yuan"].values.reshape(-1, 1))
            np_scale_param = np.array([stock_Swh.mean_, stock_Swh.scale_, Syuan_Swh.mean_, Syuan_Swh.scale_])
            np_start_end_date = np.array([start_date, end_date])
        else:
            flag_result,stock_Swh, Syuan_Swh, np_scale_param, np_start_end_date=self._get_refer_data_StandardScalers(stock)
            if not flag_result:
                exclude_stock_list(self.data_name).add_to_exlude_list(stock, reason="new_in_{0}".format(self.data_name))
                return False,"","","","","",""

        np_stock_SwhV1 = stock_Swh.transform(dfs["open_price"].values.reshape(-1, 1)).reshape((-1,))
        np_stock_SwhV1[np_stock_SwhV1 > 1.5] = 1.5

        np_Syuan_SwhV1 = Syuan_Swh.transform(dfs["amount_yuan"].values.reshape(-1, 1)).reshape((-1,))
        np_Syuan_SwhV1[np_Syuan_SwhV1 > 1.5] = 1.5

        l_np_date_s = []
        l_np_stock_S20V20 = []
        l_np_syuan_SwhV20 = []
        l_np_stock_SwhV1  = []
        for np_date_s in periods:
            l_date_s = []
            l_stock_S20V20 = []
            l_syuan_SwhV20 = []
            l_stock_SwhV1  = []
            print "handling {0} from {1} to {2}".format(stock, np_date_s[0], np_date_s[-1])
            for date_s in np_date_s:
                l_date_s.append(date_s)
                period_index = dfs[dfs["date"] <= date_s].index[-20:]
                l_stock_SwhV1.append(np_stock_SwhV1[period_index[-1]])
                df_period = dfs.loc[period_index]
                assert str(df_period["date"].iloc[-1]) == date_s,"{0} {1}".format(df_period["date"], date_s)
                #np_data = df_period[["open_price", "highest_price", "lowest_price", "close_price"]].values
                np_data = df_period[["highest_price", "lowest_price"]].values
                Index_S20V20 = preprocessing.StandardScaler().fit_transform(np_data)
                IYuan_SwhV20 = np_Syuan_SwhV1[period_index]
                l_stock_S20V20.append(np.expand_dims(Index_S20V20, axis=0))
                l_syuan_SwhV20.append(np.expand_dims(IYuan_SwhV20, axis=0))
            l_np_date_s.append(np.array(l_date_s))
            l_np_stock_SwhV1.append(np.array(l_stock_SwhV1))
            l_np_stock_S20V20.append(np.expand_dims(np.concatenate(l_stock_S20V20, axis=0), axis=1))
            l_np_syuan_SwhV20.append(np.expand_dims(np.concatenate(l_syuan_SwhV20, axis=0), axis=1))
        self._save(stock, l_np_date_s,l_np_stock_SwhV1, l_np_stock_S20V20, l_np_syuan_SwhV20,
                                np_scale_param,np_start_end_date)
        return True, l_np_date_s,l_np_stock_SwhV1, l_np_stock_S20V20, l_np_syuan_SwhV20,np_scale_param,np_start_end_date

#legacy
class G_RL_data_1index:
    def __init__(self,data_name,index,start_date, end_date):
        dfinput= API_index_from_file().get_df_index(index)
        working_dir=os.path.join(sc.base_dir_RL_data, data_name, "index")
        if not os.path.exists(working_dir): os.mkdir(working_dir)
        output_fnwp=os.path.join(working_dir,"{0}_RL_input.h5py".format(index))
        if os.path.exists(output_fnwp):
            self.np_date, self.np_Index_SwhV1, self.np_Index_S20V20, self.np_Iyuan_SwhV20, \
                        np_scale_param, np_start_end_date=self._load(output_fnwp)
            self.mean_Index_Swh, self.scale_Index_Swh,self.mean_Iyuan_Swh, self.scale_Iyuan_Swh= np_scale_param
            self.start_date, self.end_date=np_start_end_date
            assert self.start_date==start_date
            assert self.end_date==end_date
        else:
            td=API_trade_date().np_date_s
            period=td[(td>=start_date)&(td<=end_date)]
            dfi=dfinput[(dfinput["date"]>=start_date)&(dfinput["date"]<=end_date)]
            dfi.reset_index(drop=True, inplace=True)  #later need use index also on np array, so need to reset index
            assert len(period)==len(dfi)

            Index_Swh = preprocessing.StandardScaler().fit(dfi["open_price"].values.reshape(-1, 1))
            np_Index_SwhV1=Index_Swh.transform(dfi["open_price"].values.reshape(-1, 1)).reshape((-1,))
            np_Index_SwhV1[np_Index_SwhV1>1.5]=1.5
            self.np_Index_SwhV1=np_Index_SwhV1[19:-1] #this is to make the date same as l_Index_S20V20 and l_Iyuan_SwhV20

            Iyuan_Swh = preprocessing.StandardScaler().fit(dfi["amount_yuan"].values.reshape(-1, 1))
            np_Iyuan_SwhV1=Iyuan_Swh.transform(dfi["amount_yuan"].values.reshape(-1, 1)).reshape((-1,))
            np_Iyuan_SwhV1[np_Iyuan_SwhV1>1.5]=1.5

            self.mean_Index_Swh,self.scale_Index_Swh=Index_Swh.mean_,Index_Swh.scale_
            self.mean_Iyuan_Swh, self.scale_Iyuan_Swh=Iyuan_Swh.mean_,Iyuan_Swh.scale_


            l_date=[]
            l_Index_S20V20=[]
            l_Iyuan_SwhV20=[]
            print "handling  {0} from {1} to {2}".format(index, start_date, end_date)
            for date_s in period[19:-1]:
                #print "handling {0}".format(date_s)
                period_index=dfi[dfi["date"] <= date_s].index[-20:]
                df_period=dfi.iloc[period_index]
                assert str(df_period["date"].iloc[-1])==date_s
                #np_data=df_period[["open_price", "highest_price", "lowest_price", "close_price"]].values
                np_data = df_period[["highest_price", "lowest_price"]].values
                Index_S20V20 = preprocessing.StandardScaler().fit_transform(np_data)
                IYuan_SwhV20=  np_Iyuan_SwhV1[period_index]
                l_date.append(date_s)
                l_Index_S20V20.append(np.expand_dims(Index_S20V20, axis=0))
                l_Iyuan_SwhV20.append(np.expand_dims(IYuan_SwhV20, axis=0))

            self.np_date=np.array(l_date)
            self.np_Index_S20V20=np.concatenate(l_Index_S20V20,axis=0)
            self.np_Iyuan_SwhV20=np.concatenate(l_Iyuan_SwhV20,axis=0)

            np_scale_param=np.array([self.mean_Index_Swh,self.scale_Index_Swh,self.mean_Iyuan_Swh,self.scale_Iyuan_Swh])
            np_start_end_date=np.array([start_date, end_date])
            self.start_date, self.end_date = np_start_end_date
            self._save(output_fnwp,self.np_date,self.np_Index_SwhV1, self.np_Index_S20V20,self.np_Iyuan_SwhV20,
                       np_scale_param,np_start_end_date)

    def _save(self,output_fnwp,np_date,np_Index_SwhV1,np_Index_S20V20,np_Iyuan_SwhV20,np_scale_param,np_start_end_date):
        with h5py.File(output_fnwp, "w") as hf:
            hf.create_dataset("np_date_s", data=np_date)
            hf.create_dataset("np_Index_SwhV1", data=np_Index_SwhV1)
            hf.create_dataset("np_Index_S20V20", data=np_Index_S20V20)
            hf.create_dataset("np_Iyuan_SwhV20", data=np_Iyuan_SwhV20)
            hf.create_dataset("np_scale_param", data=np_scale_param)
            hf.create_dataset("np_start_end_date", data=np_start_end_date)

    def _load(self,output_fnwp):
        with h5py.File(output_fnwp, "r") as hf:
            np_date = hf["np_date_s"][:]
            np_Index_SwhV1 = hf["np_Index_SwhV1"][:]
            np_Index_S20V20 = hf["np_Index_S20V20"][:]
            np_Iyuan_SwhV20 = hf["np_Iyuan_SwhV20"][:]
            np_scale_param = hf["np_scale_param"][:]
            np_start_end_date =  hf["np_start_end_date"][:]
            return np_date,np_Index_SwhV1,np_Index_S20V20,np_Iyuan_SwhV20,np_scale_param,np_start_end_date


def main(argv):
    '''
    ###legacy
    if argv[0]=="TT":
        Prepare_data("TT", int(argv[1]), int(argv[2]))
    elif argv[0] =="FT":
        #G_RL_data_1stock_addon_xd().Prepare_RL_data(["SH600000"], "TT", "FT")
        stock_list=API_G_IPO_sl( "TT","SH", "20120601").selected_sl["stock"].tolist()
        G_RL_data_1stock_addon_xd().Prepare_RL_data(stock_list, "TT", "FT")
    '''
    data_name, stock_type, date_i=keyboard_input()
    if argv[0]=="prepare_intermediate_summary_data":
        stock_list = API_G_IPO_sl(data_name, stock_type, str(date_i)).load_stock_list(1, 0) # 1, 0 means all
        for idx, stock in enumerate(stock_list):
            if G_summary_data_1stock(data_name, stock).prepare_data():
                print "finished_{0}_{1}**************************************************".format(idx, stock)
            else:
                print "failed create_{0}_{1}*********************************************".format(idx, stock)

    if argv[0]=="prepare_intermediate_addon_data":
        stock_list = API_G_IPO_sl(data_name, stock_type, str(date_i)).load_stock_list(1, 0) # 1, 0 means all
        refer_data_name=input("input the data_name where scaler read from or (N)ew to create: ")
        if refer_data_name=="N":
            refer_data_name=""
        g = G_addon_data_1stock(data_name, refer_data_name)
        for idx, stock in enumerate(stock_list):
            flag_result,_,_,_,_,_,_=g.prepare_data(stock)
            if not flag_result:
                print "Wrong_{0}_{1}_store in exclude list*******************************".format(idx,stock)
            else:
                print "finished_{0}_{1}**************************************************".format(idx, stock)

if __name__ == '__main__':
    main(sys.argv[1:])




##to delete
'''
class G_RL_data_1stock_addon_xd:
    def __init__(self):
        self.td = API_trade_date().np_date_s

    def _get_addon_data_1stock_fnwp(self, data_name, stock):
        stock_number = int(stock[2:])
        stock_head = stock[0:2]
        stock_divide_dir = "{0}__{1}".format(stock_head, stock_number % 10)
        working_dir = sc.base_dir_RL_data
        sub_dir_list = [data_name, stock_divide_dir]
        for sub_dir in sub_dir_list:
            working_dir = os.path.join(working_dir, sub_dir)
            if not os.path.exists(working_dir): os.mkdir(working_dir)
        fnwp = os.path.join(working_dir, "{0}_RL_addon_input.h5py".format(stock))
        return fnwp

    def _check_addon_xd_1stock_avalaible(self, data_name, stock):
        fnwp = self._get_addon_data_1stock_fnwp(data_name,stock)
        return os.path.exists(fnwp)

    def _save_data_addon_xd_1stock(self, data_name, stock, l_np_date_s, l_np_stock_SwhV1, l_np_stock_S20V20, l_np_syuan_SwhV20,
                           np_scale_param,np_start_end_date):
        num_periods = len(l_np_date_s)
        assert len(l_np_stock_SwhV1) == num_periods
        assert len(l_np_stock_S20V20) == num_periods
        assert len(l_np_syuan_SwhV20) == num_periods
        fnwp = self._get_addon_data_1stock_fnwp(data_name,stock)
        with h5py.File(fnwp, "w") as hf:
            hg_xd=hf.create_group("__param")
            hg_xd.create_dataset("np_scale_param", data=np_scale_param)
            hg_xd.create_dataset("np_start_end_date", data=np_start_end_date)
            for idx in xrange(num_periods):
                hf_wg = hf.create_group("period_{0}".format(idx))
                hf_wg.create_dataset("np_date_s", data=l_np_date_s[idx])
                hf_wg.create_dataset("np_stock_SwhV1", data=l_np_stock_SwhV1[idx])
                hf_wg.create_dataset("np_stock_S20V20", data=l_np_stock_S20V20[idx])
                hf_wg.create_dataset("np_syuan_SwhV20", data=l_np_syuan_SwhV20[idx])

    def _load_data_addon_xd_1stock(self, data_name, stock):
        l_np_date_s=[]
        l_np_stock_SwhV1 = []
        l_np_stock_S20V20 = []
        l_np_syuan_SwhV20 = []
        fnwp = self._get_addon_data_1stock_fnwp(data_name,stock)
        with h5py.File(fnwp, "r") as hf:
            hg_xd = hf["__param"]
            np_scale_param=hg_xd["np_scale_param"][:]
            np_start_end_date=hg_xd["np_start_end_date"][:]
            l_period_raw=hf.keys()
            l_period = [item for item in l_period_raw if not str(item).startswith("__")]
            for idx,period in enumerate(l_period):
                hf_wg=hf[period]
                l_np_date_s.append(hf_wg["np_date_s"][:])
                l_np_stock_S20V20.append(hf_wg["np_stock_S20V20"][:])
                l_np_stock_SwhV1.append(hf_wg["np_stock_SwhV1"][:])
                l_np_syuan_SwhV20.append(hf_wg["np_syuan_SwhV20"][:])
        return l_np_date_s, l_np_stock_SwhV1, l_np_stock_S20V20, l_np_syuan_SwhV20, np_scale_param,np_start_end_date

    def prepare_data_addon_xd_1stock(self, base_l_np_date_s,data_name,stock):
        if self._check_addon_xd_1stock_avalaible(data_name, stock):
            return self._load_data_addon_xd_1stock(data_name, stock)
        first_date=base_l_np_date_s[0][0]
        try:
            last_date=base_l_np_date_s[-1][-1]
        except Exception as e:
            print stock
            raise ValueError (e)
        start_date=self.td[self.td<=first_date][-20]
        end_date=self.td[self.td>=last_date][1]
        dfsi = API_HFQ_from_file().get_df_HFQ(stock)
        dfsi=dfsi[(dfsi["date"]>=start_date) &(dfsi["date"]<=end_date)]
        tdsi = self.td[(self.td >= start_date) & (self.td <= end_date)]
        dfti = pd.DataFrame(tdsi, columns=["date"])
        dfs = pd.merge(dfsi, dfti, how='outer', left_on=['date'], right_on=["date"])
        for item in ["code", "open_price", "highest_price", "lowest_price", "close_price", "coefficient_fq"]:
            dfs[item].ffill(inplace=True)
        for item in ["amount_gu", "amount_yuan", "exchange_ratio_for_tradable_part", "exchange_ratio_for_whole"]:
            dfs[item].fillna(0, inplace=True)

        dfs.sort_values(["date"], inplace=True)
        dfs.reset_index(drop=True,inplace=True)

        stock_Swh = preprocessing.StandardScaler().fit(dfs["open_price"].values.reshape(-1, 1))
        np_stock_SwhV1 = stock_Swh.transform(dfs["open_price"].values.reshape(-1, 1)).reshape((-1,))
        np_stock_SwhV1[np_stock_SwhV1 > 1.5] = 1.5

        Syuan_Swh = preprocessing.StandardScaler().fit(dfs["amount_yuan"].values.reshape(-1, 1))
        np_Syuan_SwhV1 = Syuan_Swh.transform(dfs["amount_yuan"].values.reshape(-1, 1)).reshape((-1,))
        np_Syuan_SwhV1[np_Syuan_SwhV1 > 1.5] = 1.5


        #self.mean_Index_Swh, self.scale_Index_Swh = stock_Swh.mean_, stock_Swh.scale_
        #self.mean_Iyuan_Swh, self.scale_Iyuan_Swh = Syuan_Swh.mean_, Syuan_Swh.scale_
        l_np_date_s = []
        l_np_stock_S20V20 = []
        l_np_syuan_SwhV20 = []
        l_np_stock_SwhV1  = []
        for np_date_s in base_l_np_date_s:
            l_date_s = []
            l_stock_S20V20 = []
            l_syuan_SwhV20 = []
            l_stock_SwhV1  = []
            print "handling {0} from {1} to {2}".format(stock, np_date_s[0], np_date_s[-1])
            for date_s in np_date_s:
                l_date_s.append(date_s)
                period_index = dfs[dfs["date"] <= date_s].index[-20:]
                l_stock_SwhV1.append(np_stock_SwhV1[period_index[-1]])
                df_period = dfs.loc[period_index]
                assert str(df_period["date"].iloc[-1]) == date_s,"{0} {1}".format(df_period["date"], date_s)
                #np_data = df_period[["open_price", "highest_price", "lowest_price", "close_price"]].values
                np_data = df_period[["highest_price", "lowest_price"]].values
                Index_S20V20 = preprocessing.StandardScaler().fit_transform(np_data)
                IYuan_SwhV20 = np_Syuan_SwhV1[period_index]
                l_stock_S20V20.append(np.expand_dims(Index_S20V20, axis=0))
                l_syuan_SwhV20.append(np.expand_dims(IYuan_SwhV20, axis=0))
            l_np_date_s.append(np.array(l_date_s))
            l_np_stock_SwhV1.append(np.array(l_stock_SwhV1))
            l_np_stock_S20V20.append(np.expand_dims(np.concatenate(l_stock_S20V20, axis=0), axis=1))
            l_np_syuan_SwhV20.append(np.expand_dims(np.concatenate(l_syuan_SwhV20, axis=0), axis=1))
        np_scale_param = np.array([stock_Swh.mean_, stock_Swh.scale_, Syuan_Swh.mean_, Syuan_Swh.scale_])
        np_start_end_date = np.array([start_date, end_date])
        self._save_data_addon_xd_1stock(data_name,stock, l_np_date_s,l_np_stock_SwhV1, l_np_stock_S20V20, l_np_syuan_SwhV20,
                                np_scale_param,np_start_end_date)
        return l_np_date_s,l_np_stock_SwhV1, l_np_stock_S20V20, l_np_syuan_SwhV20,np_scale_param,np_start_end_date

    def Prepare_RL_data(self, stock_list, base_data_name, data_name):
        ii=G_RL_data_1index(data_name="FT", index="SH000001", start_date="20130101", end_date="20171231")
        index_np_date, np_Index_SwhV1, np_Index_S20V20, np_Iyuan_SwhV20=\
            ii.np_date, ii.np_Index_SwhV1, ii.np_Index_S20V20, ii.np_Iyuan_SwhV20

        for stock in stock_list:
            #i_base_data = G_RL_data_1stock(self.base_data_name, stock)
            ist = FH_RL_data_1stock(base_data_name, stock)
            if not ist.check_data_avalaible():
                print "{0} {1} data does not exists, fail to prepare {2} {1} data ".format(base_data_name, stock, data_name)
                continue
            else:
                base_l_np_date_s, l_np_large_view, l_np_small_view, l_l_support_view = ist.load_main_data()

                l_np_date_s,l_np_stock_SwhV1, l_np_stock_S20V20, l_np_syuan_SwhV20,_,_ = \
                    self.prepare_data_addon_xd_1stock(base_l_np_date_s,data_name,stock)
                new_l_lv = []
                assert len(base_l_np_date_s) == len(l_np_date_s)
                num_period = len(base_l_np_date_s)
                for idx in range(num_period):
                    assert l_np_date_s[idx][0] == base_l_np_date_s[idx][0]
                    assert l_np_date_s[idx][-1] == base_l_np_date_s[idx][-1]
                    a = np.expand_dims(l_np_syuan_SwhV20[idx], axis=3)  # a shape (93, 1, 20, 1)
                    period_idx_index = (index_np_date >= l_np_date_s[idx][0]) & (
                    (index_np_date <= l_np_date_s[idx][-1]))
                    b = np.expand_dims(ii.np_Index_S20V20[period_idx_index], axis=1)
                    c = np.expand_dims(np.expand_dims(ii.np_Iyuan_SwhV20[period_idx_index], axis=1), axis=3)
                    d = np.concatenate([l_np_large_view[idx], l_np_stock_S20V20[idx], a, b, c], axis=3) # d shape (93, 1, 20, 20)
                    new_l_lv.append(d)
                new_ll_support_view = []
                for idx in range(num_period):
                    assert l_np_date_s[idx][0] == base_l_np_date_s[idx][0]
                    assert l_np_date_s[idx][-1] == base_l_np_date_s[idx][-1]
                    period_idx_index = (index_np_date >= l_np_date_s[idx][0]) & (
                        (index_np_date <= l_np_date_s[idx][-1]))
                    ia=np_Index_SwhV1[period_idx_index]
                    ib=l_np_stock_SwhV1[idx]

                    len_list=len(l_l_support_view[idx])
                    new_l_support_view=[]
                    for list_idx in range(len_list):
                        new_l=[]
                        new_l.extend(l_l_support_view[idx][list_idx])
                        new_l.append(ia[list_idx])
                        new_l.append(ib[list_idx])
                        new_l_support_view.append(new_l)
                    new_ll_support_view.append(new_l_support_view)
                i_fh=FH_RL_data_1stock(data_name,stock)
                i_fh.save_main_data([base_l_np_date_s,new_l_lv,l_np_small_view,new_ll_support_view])
'''