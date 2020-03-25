from data_common import API_trade_date,API_G_IPO_sl,API_SH_sl,API_G_IPO_sl,exclude_stock_list,ginfo_one_stock,keyboard_input
from data_intermediate_result import FH_summary_data_1stock,G_summary_data_1stock
from data_TTFT import FH_RL_data_1stock,R_T4,R_T3,G_T3
from data_intermediate_result import FH_addon_data_1stock
import config as sc
import os,sys
import numpy as np

class R_T5(R_T4):
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
        [lv, sv], support_view_dic= R_T4.read_one_day_data(self, date_s)
        return [lv, np.expand_dims(sv, axis=0)], support_view_dic

    def read_one_day_data_by_index(self, period_idx, idx):
        [lv, sv], support_view_dic = R_T4.read_one_day_data_by_index(self, period_idx, idx)
        return [lv, np.expand_dims(sv, axis=0)], support_view_dic

    def BT_read_one_day_data(self, date_s):
        [lv, sv], support_view_dic = R_T4.BT_read_one_day_data(self, date_s)
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



class R_T5_scale(R_T5):
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
        [lv, sv], support_view_dic= R_T4.read_one_day_data(self, date_s)
        lv = self.lv_scale(lv)
        sv = self.sv_scale(sv)
        return [lv, np.expand_dims(sv, axis=0)], support_view_dic

    def read_one_day_data_by_index(self, period_idx, idx):
        [lv, sv], support_view_dic = R_T4.read_one_day_data_by_index(self, period_idx, idx)
        lv = self.lv_scale(lv)
        sv = self.sv_scale(sv)
        return [lv, np.expand_dims(sv, axis=0)], support_view_dic

    def BT_read_one_day_data(self, date_s):
        [lv, sv], support_view_dic = R_T4.BT_read_one_day_data(self, date_s)
        lv = self.lv_scale(lv)
        sv = self.sv_scale(sv)
        return [lv, np.expand_dims(sv, axis=0)], support_view_dic


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
        sv[:, 0:1][sv[:, 0:1] <=-4] =-4
        sv[:, 0:1][sv[:, 0:1] >= 4] = 4
        sv[:, 0:1]= (sv[:, 0:1] +4 ) * D_factors["f8"]

        sv[:, 1:2][sv[:, 1:2] <=-4] =-4
        sv[:, 1:2][sv[:, 1:2] >= 4] = 4
        sv[:, 1:2]= (sv[:, 1:2] +4 ) * D_factors["f8"]

        return sv

class R_T5_balance(R_T5_scale):
    def __init__(self,data_name,stock):
        R_T5.__init__(self, data_name,stock)

    def lv_scale(self, lv):
        lv[:, 5:6] = lv[:, 5:6] - 0.5

        lv[:, 8:10] = lv[:, 8:10] - 0.5

        lv[:, 12:14] = lv[:, 12:14] - 0.5
        return lv

    def sv_scale(self, sv):
        return sv

class R_T5_skipSwh(R_T5): #lv shape (20 *16)
    def __init__(self,data_name, stock):
        R_T5.__init__(self,data_name, stock)
        self.skip_idx=14   # yuan SwhV20


    def _skip_column(self, lv,cidx):
        #print lv.shape
        #assert False
        assert cidx>0 and cidx<lv.shape[2]
        return np.concatenate([lv[:,:,:cidx],lv[:,:,cidx+1:]],axis=2)

    def read_one_day_data(self, date_s):
        [lv, sv], support_view_dic= R_T4.read_one_day_data(self, date_s)
        lv=self._skip_column(lv, self.skip_idx)
        assert lv.shape[2]==16, "{0}".format(lv.shape)
        return [lv, np.expand_dims(sv, axis=0)], support_view_dic

    def read_one_day_data_by_index(self, period_idx, idx):
        [lv, sv], support_view_dic = R_T4.read_one_day_data_by_index(self, period_idx, idx)
        lv=self._skip_column(lv, self.skip_idx)
        assert lv.shape[2] == 16,"{0}".format(lv.shape)
        return [lv, np.expand_dims(sv, axis=0)], support_view_dic

    def BT_read_one_day_data(self, date_s):
        [lv, sv], support_view_dic = R_T4.BT_read_one_day_data(self, date_s)
        lv=self._skip_column(lv, self.skip_idx)
        assert lv.shape[2] == 16,"{0}".format(lv.shape)
        return [lv, np.expand_dims(sv, axis=0)], support_view_dic

class R_T5_skipSwh_balance(R_T5):
    def __init__(self,data_name, stock):
        R_T5.__init__(self,data_name, stock)
        self.skip_idx=14   # yuan SwhV20
    def _skip_column(self, lv,cidx):
        #print lv.shape
        #assert False
        assert cidx>0 and cidx<lv.shape[2]
        return np.concatenate([lv[:,:,:cidx],lv[:,:,cidx+1:]],axis=2)
    def lv_scale(self, lv):
        lv[:, 5:6] = lv[:, 5:6] - 0.5

        lv[:, 8:10] = lv[:, 8:10] - 0.5

        lv[:, 12:14] = lv[:, 12:14] - 0.5
        return lv

    def sv_scale(self, sv):
        return sv

    def read_one_day_data(self, date_s):
        [lv, sv], support_view_dic= R_T4.read_one_day_data(self, date_s)
        lv=self._skip_column(lv, self.skip_idx)
        lv = self.lv_scale(lv)
        sv = self.sv_scale(sv)
        assert lv.shape[2]==16, "{0}".format(lv.shape)
        return [lv, np.expand_dims(sv, axis=0)], support_view_dic

    def read_one_day_data_by_index(self, period_idx, idx):
        [lv, sv], support_view_dic = R_T4.read_one_day_data_by_index(self, period_idx, idx)
        lv=self._skip_column(lv, self.skip_idx)
        lv = self.lv_scale(lv)
        sv = self.sv_scale(sv)
        assert lv.shape[2] == 16,"{0}".format(lv.shape)
        return [lv, np.expand_dims(sv, axis=0)], support_view_dic

    def BT_read_one_day_data(self, date_s):
        [lv, sv], support_view_dic = R_T4.BT_read_one_day_data(self, date_s)
        lv=self._skip_column(lv, self.skip_idx)
        lv = self.lv_scale(lv)
        sv = self.sv_scale(sv)
        assert lv.shape[2] == 16,"{0}".format(lv.shape)
        return [lv, np.expand_dims(sv, axis=0)], support_view_dic


class G_T5(G_T3):
    def get_addon_data(self, addon_data, period_idx, date_s):
        l_np_date_s, l_np_stock_SwhV1, l_np_stock_S20V20, l_np_syuan_SwhV20, _, _ = addon_data
        date_s_idx = np.where(l_np_date_s[period_idx] == date_s)[0][0]
        a = l_np_syuan_SwhV20[period_idx][date_s_idx]
        b = l_np_stock_S20V20[period_idx][date_s_idx]
        np_result = np.concatenate([np.reshape(a, (20,1)), np.reshape(b, (20,2))], axis=1)
        assert np_result.shape==(20,3)
        c = l_np_stock_SwhV1[period_idx][date_s_idx]  # c should scalar
        return [np_result, c]

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



    def prepare_1stock(self, stock):
        i_summary=G_summary_data_1stock(self.data_name,stock)
        if not i_summary.flag_prepare_data_ready:
            exclude_stock_list(self.data_name).add_to_exlude_list(stock, reason="no_summary_data")
            print "Summary data not exists {0}".format(stock)
            return False,"" ,"","",""

        if not self.summary_data_sanity_check(stock,i_summary.data):
            print "Summary data not have enough lenth {0}".format(stock)
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
                print "\thandling {0} {1} RL data period {2}".format(stock, date_s,period_idx)
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

def main(argv):
    data_name, stock_type, date_i = keyboard_input()
    if "T5" in data_name: # this is to make data for T5 T5_V2_
        stock_list = API_G_IPO_sl(data_name, stock_type, str(date_i)).load_stock_list(1, 0)  # 1, 0 means all
        while True:
            choice=raw_input("Overwrite exits data for {0}? (Y)es or (N)o: ".format(data_name))
            if choice in ["Y", "N"]:
                break
        G_T5(data_name).prepare_data(stock_list,flag_overwrite= True if choice=="Y" else False)
    else:
        print "data_T5.py on;y support create T5 seriese data, means data name should start with T5 "

if __name__ == '__main__':
    main(sys.argv[1:])


