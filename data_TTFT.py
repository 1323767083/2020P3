from sklearn import preprocessing
from copy import deepcopy
import numpy as np
import os,h5py, sys
import config as sc
from data_common import API_qz_from_file,API_HFQ_from_file, API_index_from_file,API_SH_SZ_total_sl,API_trade_date,API_qz_data_source_related,ginfo_one_stock,hfq_toolbox,exclude_stock_list,API_G_IPO_sl
from data_intermediate_result import G_summary_data_1stock,G_RL_data_1index,G_addon_data_1stock,FH_addon_data_1stock

##File handler
class FH_RL_data_1stock:
    def __init__(self, data_name,stock):
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
        fnwp=os.path.join(working_dir,"{0}_RL_input.h5py".format(stock))
        return fnwp

    def save_main_data(self, data):
        l_np_date_s, l_np_large_view, l_np_small_view, l_np_support_view=data
        num_periods = len(l_np_date_s)
        assert len(l_np_large_view)==num_periods
        assert len(l_np_small_view) == num_periods
        assert len(l_np_support_view) == num_periods

        fnwp = self.get_dump_fnwp(self.stock)
        with h5py.File(fnwp, "w") as hf:
            for idx in range(num_periods):
                hf_wg=hf.create_group("period_{0}".format(idx))
                hf_wg.create_dataset("np_date_s", data=l_np_date_s[idx])
                hf_wg.create_dataset("np_large_view", data=l_np_large_view[idx])
                hf_wg.create_dataset("np_np_small_view", data=l_np_small_view[idx])
                hf_wg.create_dataset("np_support_view", data=l_np_support_view[idx])

    def load_main_data(self):
        l_np_date_s=[]
        l_np_large_view=[]
        l_np_small_view=[]
        l_np_support_view=[]
        fnwp = self.get_dump_fnwp(self.stock)
        with h5py.File(fnwp, "r") as hf:
            l_period_raw=list(hf.keys())
            l_period = [item for item in l_period_raw if not str(item).startswith("__")]
            for period in l_period:
                hf_wg=hf[period]
                assert isinstance(hf_wg, h5py.Group)
                l_np_date_s.append(hf_wg["np_date_s"][:])
                l_np_large_view.append(hf_wg["np_large_view"][:])
                l_np_small_view.append(hf_wg["np_np_small_view"][:])
                l_np_support_view.append(hf_wg["np_support_view"][:])
        return l_np_date_s,l_np_large_view,l_np_small_view, l_np_support_view

    def load_main_data_one(self,period_num):
        l_np_date_s=[]
        l_np_large_view=[]
        l_np_small_view=[]
        l_np_support_view=[]
        fnwp = self.get_dump_fnwp(self.stock)
        with h5py.File(fnwp, "r") as hf:
            l_period=list(hf.keys())
            assert period_num<=len(l_period)-1
            hf_wg = hf["period_{0}".format(period_num)]
            assert isinstance(hf_wg, h5py.Group)
            l_np_date_s.append(hf_wg["np_date_s"][:])
            l_np_large_view.append(hf_wg["np_large_view"][:])
            l_np_small_view.append(hf_wg["np_np_small_view"][:])
            l_np_support_view.append(hf_wg["np_support_view"][:])
        return l_np_date_s,l_np_large_view,l_np_small_view, l_np_support_view

    def get_total_period_num(self):
        fnwp = self.get_dump_fnwp(self.stock)
        with h5py.File(fnwp, "r") as hf:
            l_period=list(hf.keys())
        return len(l_period)

    def check_data_avalaible(self):
        fnwp = self.get_dump_fnwp(self.stock)
        if not os.path.exists(fnwp):
            return False
        else:
            return True
##read interface
class DS_RL_data:
    def __init__(self):
        self.l_np_date_s = []
        self.l_np_large_view = []
        self.l_np_small_view = []
        self.l_l_support_view = []
#class R_RL_data_T3:
class R_T3:
    def __init__(self,data_name,stock,FH_class_name="FH_RL_data_1stock"):
        self.data_name=data_name
        self.stock=stock
        self.data =DS_RL_data()
        self.i_fh=globals()[FH_class_name](data_name,stock)
        if self.i_fh.check_data_avalaible():
            self.flag_prepare_data_ready=True
            self.data.l_np_date_s, self.data.l_np_large_view, self.data.l_np_small_view, self.data.l_l_support_view = \
                self.i_fh.load_main_data()
        else:
            self.flag_prepare_data_ready = False
            print("{0} {1} RL data file does not exists at {2}".format(self.data_name, self.stock,
                                                                       self.i_fh.get_dump_fnwp(stock)))
    def _get_indexs(self, date_s):
        if not self.flag_prepare_data_ready:
            return  False, None, None
        for period_idx, period_date_s in enumerate(self.data.l_np_date_s):
            if len(period_date_s)==0: # this is to skip empyt period , which can befound in pervious assert like SH600309 las period
                continue
            if (date_s>=period_date_s[0]) &(date_s<=period_date_s[-1]):
                idx=np.where(period_date_s == date_s)[0][0]
                return True, period_idx, idx
        else:
            return False, None, None
    def _check_indexs(self, period_idx, idx):
        if not self.flag_prepare_data_ready:
            return  False
        if period_idx<len(self.data.l_np_date_s):
            if idx <len(self.data.l_np_date_s[period_idx]):
                return True
        return False
    def convert_support_view_row( self,row):
        coverted_row = []
        coverted_row.append(eval(row[0]))
        coverted_row.append(float(row[1]))
        coverted_row.append(float(row[2]))
        coverted_row.append(row[3])
        coverted_row.append(row[4])
        return coverted_row
    def convert_support_view_dic(self, row):
        support_view_dic={"last_day_flag":              eval(row[0]),
                          "this_trade_day_Nprice":      float(row[1]),
                          "this_trade_day_hfq_ratio":   float(row[2]),
                          "stock":                      row[3],
                          "date" :                      row[4]}
        return support_view_dic
    def read_one_day_data(self, date_s):
        if not self.flag_prepare_data_ready:
            #return  False, None,None,None
            raise ValueError ("{0} {1} data not exists".format(self.data_name, self.stock))
        flag_exists, period_idx, idx=self._get_indexs(date_s)
        if not flag_exists:
            raise ValueError("{0} {1} {2} data not exists".format(self.data_name, self.stock, date_s))
        support_view_dic =self.convert_support_view_dic(self.data.l_l_support_view[period_idx][idx])
        return [self.data.l_np_large_view[period_idx][idx], self.data.l_np_small_view[period_idx][idx]], support_view_dic

    def BT_read_one_day_data(self, date_s):
        if not self.flag_prepare_data_ready:
            return  [False, "Fail_data_not_avaliable_{0}".format(self.stock)],None
        flag_exisits, period_idx, idx=self._get_indexs(date_s)
        if not flag_exisits:
            return [False, "Fail_get_data_date_{0}_{1}".format(date_s, self.stock)],None
        return [True,"Success_get_data"], [[self.data.l_np_large_view[period_idx][idx], self.data.l_np_small_view[period_idx][idx]],
                                           self.convert_support_view_row(self.data.l_l_support_view[period_idx][idx])]

    def get_td_period_start_end_dates_from_period_idx(self, period_idx):
        start_s = self.data.l_np_date_s[period_idx][0]

        end_s = self.data.l_np_date_s[period_idx][-1]
        return start_s, end_s
class R_T4(R_T3):
    def __init__(self, data_name, stock):
        R_T3.__init__(self,data_name, stock, FH_class_name="FH_RL_data_1stock")
    def convert_support_view_row( self, row):
        coverted_row = []
        coverted_row.append(eval(row[0]))
        coverted_row.append(float(row[1]))
        coverted_row.append(float(row[2]))
        coverted_row.append(row[3])
        coverted_row.append(row[4])
        coverted_row.append(float(row[5]))      #this is added for FT
        coverted_row.append(float(row[6]))  # this is added for FT
        return coverted_row
    def convert_support_view_dic( self, row):
        support_view_dic={"last_day_flag":              eval(row[0]),
                          "this_trade_day_Nprice":      float(row[1]),
                          "this_trade_day_hfq_ratio":   float(row[2]),
                          "stock":                      row[3],
                          "date" :                      row[4],
                          "Index_SwhV1":                float(row[5]),
                          "stock_SwhV1":                float(row[6]) }
        return support_view_dic
    def read_one_day_data_by_index(self, period_idx, idx):
        if not self.flag_prepare_data_ready:
            raise ValueError ("{0} {1} data not exists".format(self.data_name, self.stock))
        if not self._check_indexs(period_idx,idx):
            raise ValueError("{0} {1} period idx={2} idx={3} data not exists".
                             format(self.data_name, self.stock, period_idx, idx))
        support_view_dic = self.convert_support_view_dic(self.data.l_l_support_view[period_idx][idx])
        return [self.data.l_np_large_view[period_idx][idx], self.data.l_np_small_view[period_idx][idx]], support_view_dic
class G_T3:
    def __init__(self, data_name):
        self.data_name=data_name
        self.skip_days = sc.RL_data_skip_days
        self.least_length = sc.RL_data_least_length
        self.td=API_trade_date().np_date_s

    def _check_in_periods(self,date_s, sdata):
        for idx, period in enumerate(sdata.l_np_date_s):
            if (date_s>=period[0]) and (date_s<=period[-1]):
                return True, idx
        return False,np.nan

    def _get_period_start_end_dates_from_summary(self,sdata, period_idx ):
        data_period_start = sdata.l_np_date_s[period_idx][0]
        data_period_end = sdata.l_np_date_s[period_idx][-1]
        data_start=self.td[self.td >= data_period_start][19]
        data_end = self.td[self.td <= data_period_end][-2]
        return data_start, data_end

    # in state use hfq_price in support view use normal price
    #def prepare_one_day_data(self, date_s, flag_last_day,flag_sanity_check=True ):
    def prepare_1day_lv(self, one_view_period,sdata,i_ginform,period_idx):
        l_hprice_view=[]
        l_sell_dan_view=[]
        l_buy_dan_view = []
        l_mount_view=[]
        for day in one_view_period:
            if i_ginform.check_not_tinpai(day):
                #idx = np.where(sdata.l_np_date_s[period_idx] == day)[0][0]
                idx = np.where(sdata.l_np_date_s[period_idx] == day)
                hfq_ratio=sdata.l_np_hfq_ratio[period_idx][idx]
                Nprice = sdata.l_np_price_vs_mount[period_idx][idx]
                hprice=hfq_toolbox().get_hfqprice_from_Nprice(Nprice,hfq_ratio)
                l_hprice_view.append(hprice)

                sell_da_dan_median_Nprice = sdata.l_np_sell_dan[period_idx][idx, 0]
                sell_da_dan_median_hprice = hfq_toolbox().get_hfqprice_from_Nprice(sell_da_dan_median_Nprice, hfq_ratio)
                sell_da_dan_average_Nprice = sdata.l_np_sell_dan[period_idx][idx, 1]
                sell_da_dan_average_hprice = hfq_toolbox().get_hfqprice_from_Nprice(sell_da_dan_average_Nprice, hfq_ratio)
                sell_xiao_dan_percent= sdata.l_np_sell_dan[period_idx][idx,2]
                sell_da_dan_percent= sdata.l_np_sell_dan[period_idx][idx,3]

                sell_dan_one_record=np.concatenate([sell_da_dan_median_hprice, sell_da_dan_average_hprice,
                                     sell_xiao_dan_percent, sell_da_dan_percent],axis=1)
                l_sell_dan_view.append(sell_dan_one_record)

                buy_da_dan_median_Nprice=sdata.l_np_buy_dan[period_idx][idx,0]
                buy_da_dan_median_hprice=hfq_toolbox().get_hfqprice_from_Nprice(buy_da_dan_median_Nprice, hfq_ratio)
                buy_da_dan_average_Nprice=sdata.l_np_buy_dan[period_idx][idx,1]
                buy_da_dan_average_hprice=hfq_toolbox().get_hfqprice_from_Nprice(buy_da_dan_average_Nprice, hfq_ratio)

                buy_xiao_dan_percent=sdata.l_np_buy_dan[period_idx][idx,2]
                buy_da_dan_percent=sdata.l_np_buy_dan[period_idx][idx,3]
                buy_day_one_record=np.concatenate([buy_da_dan_median_hprice, buy_da_dan_average_hprice,
                                    buy_xiao_dan_percent, buy_da_dan_percent],axis=1)
                l_buy_dan_view.append(buy_day_one_record)

                l_mount_view.append(i_ginform.get_exchange_ratio_for_tradable_part(day))
            else:
                close_Nprice, hfq_ratio = i_ginform.get_closest_close_Nprice(day)
                close_hprice = hfq_toolbox().get_hfqprice_from_Nprice(close_Nprice, hfq_ratio)

                hprice = np.ones((1, 5), dtype=float) * close_hprice
                l_hprice_view.append(hprice)
                l_sell_dan_view.append(np.zeros((1,4), dtype=float))
                l_buy_dan_view.append(np.zeros((1,4), dtype=float))
                l_mount_view.append(0.0)

        np_hprice_view = np.concatenate(l_hprice_view, axis=0)
        scaler_hprice = preprocessing.StandardScaler().fit(np_hprice_view.reshape([-1,1]))
        np_hprice_view=scaler_hprice.transform(np_hprice_view)  #?
        np_sell_dan_view=np.concatenate(l_sell_dan_view, axis=0)
        np_sell_dan_view[:,:2] = scaler_hprice.transform(np_sell_dan_view[:,:2])
        np_sell_dan_view[:, :2][np_sell_dan_view[:, :2] < -2.0] = -2.0   #trunct 0 at -2*sigma
        np_buy_dan_view = np.concatenate(l_buy_dan_view, axis=0)
        np_buy_dan_view[:,:2] = scaler_hprice.transform(np_buy_dan_view[:,:2])
        np_buy_dan_view[:, :2][np_buy_dan_view[:, :2] < -2.0] = -2.0     #trunct 0 at -2*sigma
        np_mount_view = np.expand_dims(np.array(l_mount_view), axis=1)
        np_large_view = np.concatenate([np_hprice_view, np_mount_view, np_sell_dan_view, np_buy_dan_view], axis=1)
        return np_large_view

    def prepare_1day_sv(self, date_s, sdata,i_ginform, period_idx):
        if i_ginform.check_not_tinpai(date_s):
            idx = np.where(sdata.l_np_date_s[period_idx] == date_s)[0][0]
            np_small_view=np.reshape(sdata.l_np_norm_average_price_and_mount[period_idx][idx], (25,2))
        else:
            np_small_view=np.zeros((25,2),dtype=float)
        return np_small_view

    def prepare_1day_support_inform(self, date_s,stock,flag_last_day,sdata,i_ginform,period_idx):
        last_day_flag=flag_last_day

        #this trade day prive should be normal price not hfq_price
        if i_ginform.check_not_tinpai(date_s):
            #idx = np.where(sdata.l_np_date_s[period_idx] == date_s)[0][0]
            idx = np.where(sdata.l_np_date_s[period_idx] == date_s)
            this_trade_day_Nprice=sdata.l_np_potential_price[period_idx][idx][0]  #normal price
            this_trade_day_hfq_ratio=sdata.l_np_hfq_ratio[period_idx][idx][0]
        else:
            this_trade_day_Nprice,this_trade_day_hfq_ratio=i_ginform.get_closest_close_Nprice(date_s) #normal price

        support_view=[last_day_flag,this_trade_day_Nprice,this_trade_day_hfq_ratio,stock,date_s]

        return support_view

    def prepare_1stock(self, stock):
        i_summary=G_summary_data_1stock(self.data_name,stock)
        if not i_summary.flag_prepare_data_ready:
            exclude_stock_list(self.data_name).add_to_exlude_list(stock, reason="no_summary_data")
            print("Summary data not exists {0}".format(stock))
            return False,"" ,"","",""

        i_ginform = ginfo_one_stock(stock)
        ll_np_data_s,ll_np_large_view, ll_np_small_view, ll_support_view = [],[],[],[]
        for period_idx in range(len(i_summary.data.l_np_date_s)):
            l_data_s,l_large_view,l_small_view, l_support_view = [],[],[],[]
            data_start_s, data_end_s = self._get_period_start_end_dates_from_summary(i_summary.data,period_idx)
            period=self.td[(self.td>=data_start_s) &(self.td<=data_end_s)]
            for date_s in period:
                print("\thandling {0} {1} RL data period {2}".format(stock, date_s,period_idx))
                flag_last_day = True if date_s==i_summary.data.l_np_date_s[period_idx][-2] else False
                #result = self.prepare_1day(stock, date_s, i_summary.data, flag_last_day, i_ginform)
                flag_found, period_idx = self._check_in_periods(date_s, i_summary.data)
                if not flag_found:
                    raise ValueError("date_s not in period {0} {1}".format(stock, date_s))

                one_view_period = self.td[self.td <= date_s][-20:]
                assert len(one_view_period) == 20
                np_large_view = self.prepare_1day_lv(one_view_period, i_summary.data, i_ginform, period_idx)
                np_small_view = self.prepare_1day_sv(date_s, i_summary.data, i_ginform, period_idx)
                support_view = self.prepare_1day_support_inform(date_s, stock, flag_last_day, i_summary.data, i_ginform,
                                                                period_idx)
                l_data_s.append(date_s)
                l_large_view.append(np.expand_dims(np_large_view,axis=0))
                l_small_view.append(np.expand_dims(np_small_view, axis=0))
                l_support_view.append(support_view)
            ll_np_data_s.append(np.array(l_data_s))
            ll_np_large_view.append(np.array(l_large_view))
            ll_np_small_view.append(np.array(l_small_view))
            ll_support_view.append(l_support_view)
        return True, ll_np_data_s, ll_np_large_view, ll_np_small_view, ll_support_view

    def prepare_data(self,stock_list,flag_overwrite=False):
        for idx,stock in enumerate(stock_list):
            i_fh = FH_RL_data_1stock(self.data_name, stock)
            if not flag_overwrite and i_fh.check_data_avalaible():
                print("already exists RL data for {0} {1}".format(self.data_name, stock))
                continue
            result_flag,ll_np_data_s, ll_np_large_view, ll_np_small_view, ll_support_view=self.prepare_1stock(stock)
            if result_flag:
                i_fh.save_main_data([ll_np_data_s, ll_np_large_view, ll_np_small_view, ll_support_view])
                print("Finish prepare {0} RL data  {1}".format(stock, idx))
            else:
                print("Fail prepare {0} RL data  {1}".format(stock, idx))


#FT is obsolete, only due to addon data in its folder, not delete it
class G_T4:
    def __init__(self, data_name,base_data_name ):
        self.data_name=data_name
        self.i_FH_addon=FH_addon_data_1stock(self.data_name)
        self.base_data_name=base_data_name

    def prepare_data(self):
        i_stock_list = API_G_IPO_sl(self.data_name, "SH", "20170601")
        stock_list = i_stock_list.load_stock_list(1,0) #(1,0) means all

        ii=G_RL_data_1index(data_name="FT", index="SH000001", start_date="20130101", end_date="20171231")
        index_np_date, np_Index_SwhV1, np_Index_S20V20, np_Iyuan_SwhV20=\
            ii.np_date, ii.np_Index_SwhV1, ii.np_Index_S20V20, ii.np_Iyuan_SwhV20

        for stock in stock_list:
            #i_base_data = G_RL_data_1stock(self.base_data_name, stock)
            ist = FH_RL_data_1stock(self.base_data_name, stock)
            if not ist.check_data_avalaible():
                print("{0} {1} data does not exists, fail to prepare {2} {1} data ".format(self.base_data_name, stock, self.data_name))
                continue
            else:
                base_l_np_date_s, l_np_large_view, l_np_small_view, l_l_support_view = ist.load_main_data()

                l_np_date_s,l_np_stock_SwhV1, l_np_stock_S20V20, l_np_syuan_SwhV20,_,_ = self.i_FH_addon._load(stock)

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
                i_fh=FH_RL_data_1stock(self.data_name,stock)
                i_fh.save_main_data([base_l_np_date_s,new_l_lv,l_np_small_view,new_ll_support_view])




#to delete
class G_T3_old:
    def __init__(self,data_name,stock,skip_days=100,least_length=23):
        self.data_name=data_name
        self.stock=stock
        self.skip_days=skip_days
        self.least_length=least_length

        self.i_summary_data=G_summary_data_1stock(data_name,stock)
        if not self.i_summary_data.flag_prepare_data_ready:
            self.flag_summary_day_ready = False
            return
        else:
            self.sdata=self.i_summary_data.data
            self.flag_summary_day_ready = True

        self.i_fh=FH_RL_data_1stock(data_name,stock)

        if self.i_fh.check_data_avalaible():
            print("{0} {1} RL data file exists at {2}".format(self.data_name, self.stock, self.i_fh.get_dump_fnwp(self.stock)))
            self.flag_RL_day_avalaible=True
            return
        else:
            self.flag_RL_day_avalaible = False

        self.td=API_trade_date().np_date_s
        self.i_ginform = ginfo_one_stock(self.stock)

    def check_in_periods(self,date_s):
        if not self.flag_summary_day_ready:
            return False, np.nan
        #for idx,period in enumerate(self.data_periods):
        for idx, period in enumerate(self.sdata.l_np_date_s):
            if (date_s>=period[0]) and (date_s<=period[-1]):
                return True, idx
        return False,np.nan

    def _get_td_period_start_end_dates_from_summary_period_idx(self,period_idx ):
        data_period_start = self.sdata.l_np_date_s[period_idx][0]
        data_period_end = self.sdata.l_np_date_s[period_idx][-1]
        data_start=self.td[self.td >= data_period_start][19]
        data_end = self.td[self.td <= data_period_end][-2]
        return data_start, data_end

    # in state use hfq_price in support view use normal price
    #def prepare_one_day_data(self, date_s, flag_last_day,flag_sanity_check=True ):
    def prepare_one_day_data(self, date_s, flag_last_day):
        if self.flag_RL_day_avalaible:
            print("{0} {1} RL data file exists at {2}".format(self.data_name, self.stock, self.i_fh.get_dump_fnwp(self.stock)))
            return True

        if not self.flag_summary_day_ready:
            raise ValueError("summery data for {0} does not exsits".format(self.stock))

        flag_found, period_idx=self.check_in_periods(date_s)
        if not flag_found:
            raise ValueError("date_s not in period {0} {1}".format(self.stock, date_s))

        data_start, data_end=self._get_td_period_start_end_dates_from_summary_period_idx(period_idx)
        if date_s < data_start:
            raise ValueError("not enough day (20) before {0}".format(date_s))
        if date_s>data_end:  # less than 20 trade days for this stock this period trade date
            #print date_s ,  data_end
            #print period_idx
            #print self.sdata.l_np_date_s[period_idx][19:]
            raise ValueError("not enough day (1) after {0}".format(date_s))


        one_view_start_s=self.td[self.td<=date_s][-20]
        #next_trade_days=self.td[self.td>date_s][0]
        one_view_period=self.td[(self.td>=one_view_start_s)&(self.td<=date_s)]
        assert len(one_view_period)==20


        # make large view
        l_hprice_view=[]
        l_sell_dan_view=[]
        l_buy_dan_view = []
        l_mount_view=[]
        for day in one_view_period:
            if self.i_ginform.check_not_tinpai(day):
                idx = np.where(self.sdata.l_np_date_s[period_idx] == day)[0][0]
                hfq_ratio=self.sdata.l_np_hfq_ratio[period_idx][idx]
                #price=self.sdata.l_np_price_vs_mount[period_idx][idx]*hfq_ratio
                #l_price_view.append(price) #normal price
                Nprice = self.sdata.l_np_price_vs_mount[period_idx][idx]
                hprice=hfq_toolbox().get_hfqprice_from_Nprice(Nprice,hfq_ratio)
                l_hprice_view.append(hprice)

                #sell_da_dan_median_hprice=hfq_toolbox().get_hfqprice_from_Nprice(Nprice,hfq_ratio)

                sell_da_dan_median_Nprice = self.sdata.l_np_sell_dan[period_idx][idx, 0]
                sell_da_dan_median_hprice = hfq_toolbox().get_hfqprice_from_Nprice(sell_da_dan_median_Nprice, hfq_ratio)
                sell_da_dan_average_Nprice = self.sdata.l_np_sell_dan[period_idx][idx, 1]
                sell_da_dan_average_hprice = hfq_toolbox().get_hfqprice_from_Nprice(sell_da_dan_average_Nprice, hfq_ratio)
                sell_xiao_dan_percent= self.sdata.l_np_sell_dan[period_idx][idx,2]
                sell_da_dan_percent= self.sdata.l_np_sell_dan[period_idx][idx,3]
                sell_dan_one_record=np.concatenate([sell_da_dan_median_hprice, sell_da_dan_average_hprice,
                                     sell_xiao_dan_percent, sell_da_dan_percent],axis=1)
                l_sell_dan_view.append(sell_dan_one_record)

                buy_da_dan_median_Nprice=self.sdata.l_np_buy_dan[period_idx][idx,0]
                buy_da_dan_median_hprice=hfq_toolbox().get_hfqprice_from_Nprice(buy_da_dan_median_Nprice, hfq_ratio)
                buy_da_dan_average_Nprice=self.sdata.l_np_buy_dan[period_idx][idx,1]
                buy_da_dan_average_hprice=hfq_toolbox().get_hfqprice_from_Nprice(buy_da_dan_average_Nprice, hfq_ratio)

                buy_xiao_dan_percent=self.sdata.l_np_buy_dan[period_idx][idx,2]
                buy_da_dan_percent=self.sdata.l_np_buy_dan[period_idx][idx,3]
                buy_day_one_record=np.concatenate([buy_da_dan_median_hprice, buy_da_dan_average_hprice,
                                    buy_xiao_dan_percent, buy_da_dan_percent],axis=1)
                l_buy_dan_view.append(buy_day_one_record)

                l_mount_view.append(self.i_ginform.get_exchange_ratio_for_tradable_part(day))
            else:
                #close_price, fq_coeffecint=self.i_ginform.get_closest_close_Nprice(day)
                #price=np.ones((1,5),dtype=float)*close_price*fq_coeffecint
                close_Nprice, hfq_ratio = self.i_ginform.get_closest_close_Nprice(day)
                close_hprice = hfq_toolbox().get_hfqprice_from_Nprice(close_Nprice, hfq_ratio)

                hprice = np.ones((1, 5), dtype=float) * close_hprice
                l_hprice_view.append(hprice)
                l_sell_dan_view.append(np.zeros((1,4), dtype=float))
                l_buy_dan_view.append(np.zeros((1,4), dtype=float))
                l_mount_view.append(0.0)

        np_hprice_view = np.concatenate(l_hprice_view, axis=0)

        scaler_hprice = preprocessing.StandardScaler().fit(np_hprice_view.reshape([-1,1]))
        np_hprice_view=scaler_hprice.transform(np_hprice_view)  #?

        np_sell_dan_view=np.concatenate(l_sell_dan_view, axis=0)

        np_sell_dan_view[:,:2] = scaler_hprice.transform(np_sell_dan_view[:,:2])
        np_sell_dan_view[:, :2][np_sell_dan_view[:, :2] < -2.0] = -2.0   #trunct 0 at -2*sigma

        np_buy_dan_view = np.concatenate(l_buy_dan_view, axis=0)
        np_buy_dan_view[:,:2] = scaler_hprice.transform(np_buy_dan_view[:,:2])
        np_buy_dan_view[:, :2][np_buy_dan_view[:, :2] < -2.0] = -2.0     #trunct 0 at -2*sigma


        np_mount_view = np.expand_dims(np.array(l_mount_view), axis=1)
        self.np_large_view = np.concatenate([np_hprice_view, np_mount_view, np_sell_dan_view, np_buy_dan_view], axis=1)
        #build small view
        if self.i_ginform.check_not_tinpai(date_s):
            idx = np.where(self.sdata.l_np_date_s[period_idx] == date_s)[0][0]
            self.np_small_view=np.reshape(self.sdata.l_np_norm_average_price_and_mount[period_idx][idx], (25,2))
        else:
            self.np_small_view=np.zeros((25,2),dtype=float)

        #build support view

        #last_day_flag=True if date_s==data_period_end else False

        #last_day_flag = True if date_s == data_end else False
        last_day_flag=flag_last_day

        #this trade day prive should be normal price not hfq_price
        if self.i_ginform.check_not_tinpai(date_s):
            idx = np.where(self.sdata.l_np_date_s[period_idx] == date_s)[0][0]
            self.this_trade_day_Nprice=self.sdata.l_np_potential_price[period_idx][idx][0]  #normal price
            self.this_trade_day_hfq_ratio=self.sdata.l_np_hfq_ratio[period_idx][idx][0]
        else:
            self.this_trade_day_Nprice,self.this_trade_day_hfq_ratio=self.i_ginform.get_closest_close_Nprice(date_s) #normal price

        self.support_view=[last_day_flag,self.this_trade_day_Nprice,self.this_trade_day_hfq_ratio,self.stock,date_s]

        self.np_large_view = np.expand_dims(self.np_large_view,axis=0)
        self.np_small_view = np.expand_dims(self.np_small_view, axis=0)

        return [self.np_large_view,self.np_small_view],self.support_view

    def prepare_data(self):
        if self.flag_RL_day_avalaible:
            print("{0} {1} RL data file exists at {2}".format(self.data_name, self.stock, self.i_fh.get_dump_fnwp(self.stock)))
            return True

        if not self.flag_summary_day_ready:
            return False

        ll_np_data_s=[]
        ll_np_large_view=[]
        ll_np_small_view = []
        ll_support_view = []
        for period_idx in range(len(self.sdata.l_np_date_s)):
            l_data_s = []
            l_large_view = []
            l_small_view = []
            l_support_view = []

            data_start_s, data_end_s = self._get_td_period_start_end_dates_from_summary_period_idx(period_idx)
            period=self.td[(self.td>=data_start_s) &(self.td<=data_end_s)]
            for date_s in period:
                print("\thandling {0} {1} RL data period {2}".format(self.stock, date_s,period_idx))
                flag_last_day = True if date_s==self.sdata.l_np_date_s[period_idx][-2] else False
                #result=self.prepare_one_day_data(date_s, flag_last_day,flag_sanity_check=False)
                result = self.prepare_one_day_data(date_s, flag_last_day)
                large_small_views, support_view=result
                large_view, small_view = large_small_views

                l_data_s.append(date_s)
                l_large_view.append(large_view)
                l_small_view.append(small_view)
                l_support_view.append(support_view)
            ll_np_data_s.append(np.array(l_data_s))
            ll_np_large_view.append(np.array(l_large_view))
            ll_np_small_view.append(np.array(l_small_view))

            ll_support_view.append(l_support_view)

        self.i_fh.save_main_data([ll_np_data_s,ll_np_large_view,ll_np_small_view,ll_support_view])

