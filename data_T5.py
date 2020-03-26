from data_common import hfq_toolbox
from data_common import API_trade_date,API_G_IPO_sl,API_SH_sl,API_G_IPO_sl,exclude_stock_list,ginfo_one_stock,keyboard_input
from data_intermediate_result import FH_summary_data_1stock,G_summary_data_1stock
#from data_TTFT import FH_RL_data_1stock,R_T4,R_T3,G_T3
from data_intermediate_result import FH_addon_data_1stock
import config as sc
import os,sys,h5py
import numpy as np
from sklearn import preprocessing

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
class R_T5:
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
                if idx <len(self.data.l_np_date_s[period_idx]):  # this strange if is to avoid self.data.l_np_date_s[period_idx] through ecemption
                    return True
        return False

    def convert_support_view_row( self, row):
        coverted_row = []
        coverted_row.append(eval(row[0]))
        coverted_row.append(float(row[1]))
        coverted_row.append(float(row[2]))
        coverted_row.append(row[3])
        coverted_row.append(row[4])
        coverted_row.append(float(row[5]))      #this is added for FT
        return coverted_row

    def convert_support_view_dic( self, row):
        support_view_dic={"last_day_flag":              eval(row[0]),
                          "this_trade_day_Nprice":      float(row[1]),
                          "this_trade_day_hfq_ratio":   float(row[2]),
                          "stock":                      row[3],
                          "date" :                      row[4],
                          "stock_SwhV1":                float(row[5]) }
        return support_view_dic

    def read_one_day_data(self, date_s):
        assert self.flag_prepare_data_ready, "{0} {1} data not exists".format(self.data_name, self.stock)
        flag_exists, period_idx, idx=self._get_indexs(date_s)
        assert flag_exists,"{0} {1} {2} data not exists".format(self.data_name, self.stock, date_s)
        lv=self.data.l_np_large_view[period_idx][idx]
        sv=self.data.l_np_small_view[period_idx][idx]
        support_view_dic =self.convert_support_view_dic(self.data.l_l_support_view[period_idx][idx])
        return [lv, np.expand_dims(sv, axis=0)], support_view_dic

    def read_one_day_data_by_index(self, period_idx, idx):
        assert self.flag_prepare_data_ready,"{0} {1} data not exists".format(self.data_name, self.stock)
        assert self._check_indexs(period_idx,idx),"{0} {1} period idx={2} idx={3} data not exists".\
            format(self.data_name, self.stock, period_idx, idx)
        support_view_dic = self.convert_support_view_dic(self.data.l_l_support_view[period_idx][idx])
        lv = self.data.l_np_large_view[period_idx][idx]
        sv = self.data.l_np_small_view[period_idx][idx]
        return [lv, np.expand_dims(sv, axis=0)], support_view_dic

    def BT_read_one_day_data(self, date_s):
        assert self.flag_prepare_data_ready,"Fail_data_not_avaliable_{0}".format(self.stock)
        flag_exisits, period_idx, idx=self._get_indexs(date_s)
        assert flag_exisits,"Fail_get_data_date_{0}_{1}_{2}".format(date_s, self.stock, date_s)
        lv = self.data.l_np_large_view[period_idx][idx]
        sv = self.data.l_np_small_view[period_idx][idx]
        support_view_dic = self.convert_support_view_row(self.data.l_l_support_view[period_idx][idx])
        return [lv, np.expand_dims(sv, axis=0)], support_view_dic
'''
250 as sample
LV data describe
                    index     type        trunct before         ways to scale
hprice              0-5(5)       Nprice        No                  -4 +4 trunct  (x+4)* 30
mount               5-6(1)         percent       no                  *250
Sell dan(first)     6-8(2)      Nprice      -2(lowest)         -2 +4   trunct    (x+2)*40
Sell dan(second)    8-10(2)     percent       No                    *250
Buy Dan (first)     10-12(2)     Nprice       -2(lowest)         -2 +4  trunct    (x+2)*40
Buy Dan (second)    12-14(2)    percent       No                    *250
Yuan SWhV20         14-15(1)       yuan         -1.5(lowst)        -1.5 +4 trunct   (x+1.5)*45
S20V20              15-17(2)      Hprice        No                    -4 +4 trunct (x+4)*30


sv data describe
                index       type        trunct before        ways to scale
average price   0-1(1)        Nprice          no                  -4 +4 trunct  (x+4)* 30
volume          1-2(1)        volume          no                  -4 +4 trunct  (x+4)* 30
'''
class R_T5_scale(R_T5):  #this not success
    def __init__(self,data_name,stock):
        R_T5.__init__(self, data_name,stock)
        D_factors_250={
            "f8":   30,
            "f6":   40,
            "f55":  45,
            "fpercent":250
        }

        D_factors_20={
            "f8":   2.5,
            "f6":   3.3,
            "f55":  3.6,
            "fpercent":20
        }
        self.D_factors=D_factors_20

    def lv_scale(self, lv):
        return self.lv_scale_base(lv,self.D_factors)
    def sv_scale(self, sv):
        return self.sv_scale_base(sv,self.D_factors)


    def read_one_day_data(self, date_s):
        [lv, sv], support_view_dic= R_T5.read_one_day_data(self, date_s)
        lv = self.lv_scale(lv)
        sv = self.sv_scale(sv)
        return [lv, sv], support_view_dic
        #return [lv, np.expand_dims(sv, axis=0)], support_view_dic

    def read_one_day_data_by_index(self, period_idx, idx):
        [lv, sv], support_view_dic = R_T5.read_one_day_data_by_index(self, period_idx, idx)
        lv = self.lv_scale(lv)
        sv = self.sv_scale(sv)
        return [lv, sv], support_view_dic
        #return [lv, np.expand_dims(sv, axis=0)], support_view_dic

    def BT_read_one_day_data(self, date_s):
        [lv, sv], support_view_dic = R_T5.BT_read_one_day_data(self, date_s)
        lv = self.lv_scale(lv)
        sv = self.sv_scale(sv)
        return [lv, sv], support_view_dic
        #return [lv, np.expand_dims(sv, axis=0)], support_view_dic


    def lv_scale_base(self, lv, D_factors):
        lv[:, 0:5][lv[:, 0:5]<=-4]=-4
        lv[:, 0:5][lv[:, 0:5]>=4] = 4
        lv[:, 0:5] = (lv[:, 0:5] + 4) * D_factors["f8"]


        lv[:, 5:6] = lv[:, 5:6] * D_factors["fpercent"]

        lv[:, 6:8][lv[:, 6:8]>=4]=4
        lv[:, 6:8] = (lv[:, 6:8] +2) * D_factors["f6"]

        lv[:, 8:10] = lv[:, 8:10] * D_factors["fpercent"]

        lv[:, 10:12][lv[:, 10:12]>=4]=4
        lv[:, 10:12] = (lv[:, 10:12] + 2) * D_factors["f6"]

        lv[:, 12:14] = lv[:, 12:14] * D_factors["fpercent"]

        lv[:, 14:15][lv[:, 14:15]>=4]=4
        lv[:, 14:15] = (lv[:, 14:15] +1.5) * D_factors["f55"]

        lv[:, 15:17][lv[:, 15:17]<=-4]=-4
        lv[:, 15:17][lv[:, 15:17]>= 4] = 4
        lv[:, 15:17] = (lv[:, 15:17] + 4) * D_factors["f8"]

        return lv

    def sv_scale_base(self, sv, D_factors):
        '''
        sv[:, 0:1][sv[:, 0:1] <=-4] =-4
        sv[:, 0:1][sv[:, 0:1] >= 4] = 4
        sv[:, 0:1]= (sv[:, 0:1] +4 ) * D_factors["f8"]

        sv[:, 1:2][sv[:, 1:2] <=-4] =-4
        sv[:, 1:2][sv[:, 1:2] >= 4] = 4
        sv[:, 1:2]= (sv[:, 1:2] +4 ) * D_factors["f8"]
        '''
        sv[0,:, 0:1][sv[:, 0:1] <=-4] =-4
        sv[0,:, 0:1][sv[:, 0:1] >= 4] = 4
        sv[0,:, 0:1]= (sv[:, 0:1] +4 ) * D_factors["f8"]

        sv[0,:, 1:2][sv[:, 1:2] <=-4] =-4
        sv[0,:, 1:2][sv[:, 1:2] >= 4] = 4
        sv[0,:, 1:2]= (sv[:, 1:2] +4 ) * D_factors["f8"]
        return sv
class R_T5_balance(R_T5_scale): #this not success and not reasonable
    def __init__(self,data_name,stock):
        R_T5_scale.__init__(self, data_name,stock)

    def lv_scale(self, lv):
        lv[:, 5:6] = lv[:, 5:6] - 0.5

        lv[:, 8:10] = lv[:, 8:10] - 0.5

        lv[:, 12:14] = lv[:, 12:14] - 0.5
        return lv

    def sv_scale(self, sv):
        return sv
class R_T5_skipSwh(R_T5): #lv shape (20 *16)   # this is reasonable
    def __init__(self,data_name, stock):
        R_T5.__init__(self,data_name, stock)
        self.skip_idx=14   # yuan SwhV20

    def _skip_column(self, lv,cidx):
        assert cidx>0 and cidx<lv.shape[2]
        return np.concatenate([lv[:,:,:cidx],lv[:,:,cidx+1:]],axis=2)

    def lv_scale(self, lv):
        lv=self._skip_column(lv, self.skip_idx)
        return lv

    def sv_scale(self, sv):
        return sv


    def read_one_day_data(self, date_s):
        [lv, sv], support_view_dic= R_T5.read_one_day_data(self, date_s)
        lv = self.lv_scale(lv)
        sv = self.sv_scale(sv)
        assert lv.shape[2]==16, "{0}".format(lv.shape)
        return [lv, sv], support_view_dic

        # return [lv, np.expand_dims(sv, axis=0)], support_view_dic

    def read_one_day_data_by_index(self, period_idx, idx):
        [lv, sv], support_view_dic = R_T5.read_one_day_data_by_index(self, period_idx, idx)
        lv = self.lv_scale(lv)
        sv = self.sv_scale(sv)
        assert lv.shape[2] == 16,"{0}".format(lv.shape)
        return [lv, sv], support_view_dic

    def BT_read_one_day_data(self, date_s):
        [lv, sv], support_view_dic = R_T5.BT_read_one_day_data(self, date_s)
        lv = self.lv_scale(lv)
        sv = self.sv_scale(sv)
        assert lv.shape[2] == 16,"{0}".format(lv.shape)
        return [lv, sv], support_view_dic
class R_T5_skipSwh_balance(R_T5): # this is final used
    def __init__(self,data_name, stock):
        R_T5.__init__(self,data_name, stock)
        self.skip_idx=14   # yuan SwhV20
    def _skip_column(self, lv,cidx):
        assert cidx>0 and cidx<lv.shape[2]
        return np.concatenate([lv[:,:,:cidx],lv[:,:,cidx+1:]],axis=2)
    def lv_scale(self, lv):
        lv=self._skip_column(lv, self.skip_idx)
        lv[:, 5:6] = lv[:, 5:6] - 0.5
        lv[:, 8:10] = lv[:, 8:10] - 0.5
        lv[:, 12:14] = lv[:, 12:14] - 0.5
        return lv

    def sv_scale(self, sv):
        return sv

    def read_one_day_data(self, date_s):
        [lv, sv], support_view_dic= R_T5.read_one_day_data(self, date_s)
        lv = self.lv_scale(lv)
        sv = self.sv_scale(sv)
        assert lv.shape[2]==16, "{0}".format(lv.shape)
        return [lv, sv], support_view_dic

    def read_one_day_data_by_index(self, period_idx, idx):
        [lv, sv], support_view_dic = R_T5.read_one_day_data_by_index(self, period_idx, idx)
        lv = self.lv_scale(lv)
        sv = self.sv_scale(sv)
        assert lv.shape[2] == 16,"{0}".format(lv.shape)
        return [lv, sv], support_view_dic


    def BT_read_one_day_data(self, date_s):
        [lv, sv], support_view_dic = R_T5.BT_read_one_day_data(self, date_s)
        lv = self.lv_scale(lv)
        sv = self.sv_scale(sv)
        assert lv.shape[2] == 16,"{0}".format(lv.shape)
        return [lv, sv], support_view_dic

class G_T5:
    def __init__(self, data_name):
        self.data_name=data_name
        self.skip_days = sc.RL_data_skip_days
        self.least_length = sc.RL_data_least_length
        self.td=API_trade_date().np_date_s

    def summary_data_sanity_check(self,stock, sdata):
        to_remove_index=[]
        for idx, _ in enumerate(sdata.l_np_date_s):
            if len(sdata.l_np_date_s[idx])<23:
                to_remove_index.append(idx)
                exclude_stock_list(self.data_name).add_to_exlude_list(stock,"summary_data_length_less_than_23")
                return False
        return True

    def _get_period_start_end_dates_from_date_s(self,correction_start_s, correction_end_s):
        data_start=self.td[self.td >= correction_start_s][19]
        data_end = self.td[self.td <= correction_end_s][-2]
        return data_start, data_end


    def get_addon_data(self, addon_data, period_idx, date_s):
        l_np_date_s, l_np_stock_SwhV1, l_np_stock_S20V20, l_np_syuan_SwhV20, _, _ = addon_data
        date_s_idx = np.where(l_np_date_s[period_idx] == date_s)[0][0]
        a = l_np_syuan_SwhV20[period_idx][date_s_idx]
        b = l_np_stock_S20V20[period_idx][date_s_idx]
        np_result = np.concatenate([np.reshape(a, (20,1)), np.reshape(b, (20,2))], axis=1)
        assert np_result.shape==(20,3)
        c = l_np_stock_SwhV1[period_idx][date_s_idx]  # c should scalar
        return [np_result, c]



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
        one_view_period = self.td[self.td <= date_s][-20:]
        l_np_small_view = []
        for date_s in one_view_period:
            #if i_ginform.check_not_tinpai(sdata.l_np_date_s[period_idx][idx]):
            if i_ginform.check_not_tinpai(date_s):
                idx = np.where(sdata.l_np_date_s[period_idx] == date_s)[0][0]
                np_small_view = np.reshape(sdata.l_np_norm_average_price_and_mount[period_idx][idx], (25, 2))
            else:
                np_small_view = np.zeros((25, 2), dtype=float)
            l_np_small_view.append(np.expand_dims(np_small_view, axis=0))
        np_result=np.vstack(l_np_small_view)
        assert np_result.shape==(20,25,2)
        return np_result


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

        if not self.summary_data_sanity_check(stock,i_summary.data):
            print("Summary data not have enough lenth {0}".format(stock))
            return False, "", "", "", ""

        i_ginform = ginfo_one_stock(stock)

        i_FH_addon = FH_addon_data_1stock(self.data_name)
        addon_data= i_FH_addon._load(stock)
        l_np_date_s=addon_data[0]


        ll_np_data_s,ll_np_large_view, ll_np_small_view, ll_support_view = [],[],[],[]
        for period_idx, _ in enumerate(i_summary.data.l_np_date_s):
            #following way to create period ensure the period has addon data availbe
            #to do this is because tinpai at begining of the start time for the data source or tinpan at end time of data source
            # which cause no data in addon data
            correction_start_s = i_summary.data.l_np_date_s[period_idx][0] \
                if l_np_date_s[period_idx][0] <= i_summary.data.l_np_date_s[period_idx][0] else l_np_date_s[period_idx][0]

            correction_end_s = i_summary.data.l_np_date_s[period_idx][-1] if \
                l_np_date_s[period_idx][-1] >= i_summary.data.l_np_date_s[period_idx][-1] else l_np_date_s[period_idx][-1]

            data_start_s, data_end_s = self._get_period_start_end_dates_from_date_s(correction_start_s, correction_end_s)
            period=self.td[(self.td>=data_start_s) &(self.td<=data_end_s)]

            l_data_s,l_large_view,l_small_view, l_support_view = [],[],[],[]
            for date_s in period:
                print("\thandling {0} {1} RL data period {2}".format(stock, date_s,period_idx))
                flag_last_day = True if date_s == period[-1] else False

                one_view_period = self.td[self.td <= date_s][-20:]
                assert len(one_view_period) == 20
                lv_addon, support_inform_addon = self.get_addon_data(addon_data, period_idx, date_s)

                np_large_view = self.prepare_1day_lv(one_view_period, i_summary.data, i_ginform, period_idx) #np_large_view shape(20,14)

                new_np_large_view=np.concatenate([np_large_view,lv_addon], axis=1)
                assert new_np_large_view.shape == (20,17)
                np_small_view = self.prepare_1day_sv(date_s, i_summary.data, i_ginform, period_idx)
                assert np_small_view.shape==(20,25,2)

                support_view = self.prepare_1day_support_inform(date_s, stock, flag_last_day, i_summary.data, i_ginform,
                                                                period_idx)
                support_view.append(support_inform_addon)

                l_data_s.append(date_s)

                l_large_view.append(np.expand_dims(np.expand_dims(new_np_large_view, axis=0), axis=0))
                #two expand_dims is to fit the legacy means the final result should be (93,1,20,17) not (93,20,17)
                l_small_view.append(np.expand_dims(np_small_view, axis=0))
                l_support_view.append(support_view)

            ll_np_data_s.append(np.vstack(l_data_s))
            ll_np_large_view.append(np.vstack(l_large_view))
            ll_np_small_view.append(np.vstack(l_small_view))
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


def main(argv):
    data_name, stock_type, date_i = keyboard_input()
    if "T5" in data_name: # this is to make data for T5 T5_V2_
        stock_list = API_G_IPO_sl(data_name, stock_type, str(date_i)).load_stock_list(1, 0)  # 1, 0 means all
        while True:
            choice=input("Overwrite exits data for {0}? (Y)es or (N)o: ".format(data_name))
            if choice in ["Y", "N"]:
                break
        G_T5(data_name).prepare_data(stock_list,flag_overwrite= True if choice=="Y" else False)
    else:
        print("data_T5.py on;y support create T5 seriese data, means data name should start with T5 ")

if __name__ == '__main__':
    main(sys.argv[1:])


