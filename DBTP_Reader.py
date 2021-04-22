from DBTP_Base import DBTP_Base
import os, random, pickle
import pandas as pd
import numpy as np

class DBTP_Reader(DBTP_Base):
    def __init__(self, DBTP_Name):
        DBTP_Base.__init__(self,DBTP_Name )

    def _get_ref_value_with_TitleD(self,ref, TitleD, MatchMood):
        assert MatchMood in ["Equal","Contain"]
        if MatchMood =="Equal":
            found_idxs=[idx for idx, FT_TitleD in enumerate(self.FT_TitlesD_Flat[2]) if TitleD == FT_TitleD]
        else:
            found_idxs = [idx for idx, FT_TitleD in enumerate(self.FT_TitlesD_Flat[2]) if TitleD in FT_TitleD]
        assert len(found_idxs)==1, "Fail Found TitleD={0} in self.FT_TitlesD_Flat[2] ={1}".format(TitleD,self.FT_TitlesD_Flat[2])
        return ref[found_idxs[0]]


    def Fill_support_view_with_ref(self, ref, Stock,DateI, Flag_last_day):
        assert self._get_ref_value_with_TitleD(ref[-1], "DateI", "Equal")==DateI, "ref={0} DateI={1}".format(ref,DateI)
        support_view_dic={"Flag_LastDay":   Flag_last_day,
                          "Nprice":         self._get_ref_value_with_TitleD(ref[-1], "Potential_NPrice","Contain"),
                          "HFQRatio":       self._get_ref_value_with_TitleD(ref[-1], "HFQ_Ratio","Equal"),
                          "Flag_Tradable":  True if self._get_ref_value_with_TitleD(ref[-1], "Flag_Tradable", "Equal")==1.0 else False,
                          "Stock":          Stock,
                          "DateI" :         DateI,
                          }
        return support_view_dic


    def read_1day_TP_Data(self, Stock, DateI):
        lv, sv, ref = pickle.load(open(self.get_DBTP_data_fnwp(Stock, DateI), "rb"))
        return np.expand_dims(lv, axis=0),np.expand_dims(sv, axis=0), ref
    #def data_reset(self):
    #    #keep for legacy
    #    return
class DBTP_Train_Reader(DBTP_Reader):
    def __init__(self, DBTP_Name, Stock, SDateI, EDateI,PLen=30):
        DBTP_Reader.__init__(self,DBTP_Name )
        self.Stock  =   Stock
        self.PLen   =   PLen
        DBTP_log_fnwp=self.get_DBTP_data_log_fnwp(Stock)
        assert os.path.exists(DBTP_log_fnwp),"File Not Exists {0}".format(DBTP_log_fnwp)
        dfSD=pd.read_csv(DBTP_log_fnwp, header=0, names=["Result", "Date", "Message"],
                    dtype={"Result": str, "Date": int, "Message": str})
        dfSD=dfSD[(dfSD["Date"]>=SDateI) & (dfSD["Date"]<=EDateI)]["Date"]
        SD=dfSD.values
        assert len(self.generate_TD_periods(SD[0],SD[-1]))==len(SD), \
            "{0}  len(self.generate_TD_periods(SD[0],SD[-1])) = {1} len(SD)= {2} ".format(
                Stock,len(self.generate_TD_periods(SD[0], SD[-1])),len(SD))
        self.SDTDSidx, _ = self.get_closest_TD(SD[0], True)
        self.SDTDEidx, _ = self.get_closest_TD(SD[-1], False)

        assert self.SDTDEidx-self.SDTDSidx>self.PLen, \
            "self.SDTDEidx-self.SDTDSidx={0}<=PLen={1}".format(self.SDTDEidx-self.SDTDSidx,self.PLen)

        self.CTDidx = self.SDTDSidx
        self.ETDidx = self.SDTDEidx

        flag,dfh,mess = self.get_hfq_df(self.get_DBI_hfq_fnwp(Stock))
        assert flag, "Fail in read HFQ for {0} {1}".format(Stock, mess)
        dfh["CloseNPrice"] = dfh["close_price"] / dfh["coefficient_fq"]
        dfh["date"]=dfh["date"].astype(int)
        dfh=dfh[(dfh["date"]>= SDateI) & (dfh["date"] <= EDateI)][["date","CloseNPrice"]]
        dfh.reset_index(inplace=True)
        df = pd.merge(dfSD, dfh, left_on="Date", right_on="date", how="outer")
        # this first day if tinpai keep NaN , as it do not have holding the eval result is invalidate
        if pd.isna(df["CloseNPrice"].loc[0]):
            df.loc[0,"CloseNPrice"]=0
        df["CloseNPrice"].ffill(inplace=True)  # if it is tinpai no hfq, use previous close price and Eval purpose.

        assert not df["CloseNPrice"].isnull().any(), " Should not have nan any more {0}".format(df)
        self.npCloseNPrice=df["CloseNPrice"].values

    def reset_get_data(self):
        self.CTDidx = random.randrange(self.SDTDSidx,self.SDTDEidx-self.PLen)
        self.ETDidx = self.CTDidx + self.PLen
        lv, sv, ref=self.read_1day_TP_Data(self.Stock, self.nptd[self.CTDidx])
        support_view_dic=self.Fill_support_view_with_ref(ref,self.Stock, self.nptd[self.CTDidx],self.CTDidx==self.ETDidx )
        support_view_dic["flag_all_period_explored"]=False
        return [lv,sv],support_view_dic

    def next_get_data(self):
        self.CTDidx+=1
        Flag_Done=True if self.CTDidx==self.ETDidx else False
        lv, sv, ref = self.read_1day_TP_Data(self.Stock, self.nptd[self.CTDidx])
        support_view_dic=self.Fill_support_view_with_ref(ref,self.Stock,self.nptd[self.CTDidx],self.CTDidx==self.ETDidx )
        support_view_dic["flag_all_period_explored"]=False
        return [lv,sv], support_view_dic, Flag_Done

    def get_CloseNPrice(self):
        return self.npCloseNPrice[self.CTDidx-self.SDTDSidx]

    def get_the_dateI(self):
        return self.nptd[self.CTDidx]
class DBTP_Eval_Reader(DBTP_Train_Reader):
    def __init__(self, DBTP_Name, Stock, SDateI, EDateI,PLen=30,eval_reset_total_times=5 ):
        DBTP_Train_Reader.__init__(self, DBTP_Name, Stock, SDateI, EDateI,PLen)
        self.eval_reset_count=0
        self.eval_reset_total_times=eval_reset_total_times

    def reset_get_data(self):
        state, support_view_dic=DBTP_Train_Reader.reset_get_data(self)
        self.eval_reset_count +=1
        if self.eval_reset_count > self.eval_reset_total_times:
            support_view_dic["flag_all_period_explored"] = True
            self.eval_reset_count = 0
        else:
            support_view_dic["flag_all_period_explored"] = False
        return state, support_view_dic

class DBTP_DayByDay_reader(DBTP_Train_Reader):
    def __init__(self, DBTP_Name, Stock, SDateI, EDateI,PLen=30,eval_reset_total_times=0):
        DBTP_Train_Reader.__init__(self, DBTP_Name, Stock, SDateI, EDateI,PLen)
        self.EvalTillTDidx= self.SDTDSidx

    def reset_get_data(self):
        self.CTDidx = self.EvalTillTDidx
        self.EvalTillTDidx+=1
        self.ETDidx = self.CTDidx + self.PLen
        lv, sv, ref=self.read_1day_TP_Data(self.Stock, self.nptd[self.CTDidx])
        support_view_dic=self.Fill_support_view_with_ref(ref,self.Stock, self.nptd[self.CTDidx],self.CTDidx==self.ETDidx )
        if self.EvalTillTDidx <= self.SDTDEidx - self.PLen:
            support_view_dic["flag_all_period_explored"] = False
        else:
            support_view_dic["flag_all_period_explored"] = True
            self.EvalTillTDidx = self.SDTDSidx
        return [lv,sv],support_view_dic



class DBTP_Eval_CC_Reader(DBTP_Train_Reader):
    def __init__(self, DBTP_Name, Stock, SDateI, EDateI, PLen=30,eval_reset_total_times=0):
        DBTP_Train_Reader.__init__(self, DBTP_Name, Stock, SDateI, EDateI,PLen)
        self.flag_init=True

    def reset_get_data(self):
        if self.flag_init:
            self.flag_init=False
        else:
            self.CTDidx += 1
        lv, sv, ref=self.read_1day_TP_Data(self.Stock, self.nptd[self.CTDidx])
        support_view_dic=self.Fill_support_view_with_ref(ref,self.Stock, self.nptd[self.CTDidx],self.CTDidx==self.ETDidx )
        if self.CTDidx <= self.SDTDEidx - self.PLen:
            support_view_dic["flag_all_period_explored"] = False
        else:
            support_view_dic["flag_all_period_explored"] = True
            self.flag_init = True
            self.CTDidx = self.SDTDSidx
        if self.CTDidx == self.SDTDEidx - self.PLen:  #this is to synchronize the stop for CC Eval
            support_view_dic["Flag_Force_Next_Reset"] = True
        else:
            support_view_dic["Flag_Force_Next_Reset"] = False
        return [lv,sv],support_view_dic


    def next_get_data(self):
        self.CTDidx+=1
        Flag_Done=True if self.CTDidx==self.ETDidx else False
        lv, sv, ref = self.read_1day_TP_Data(self.Stock, self.nptd[self.CTDidx])
        support_view_dic=self.Fill_support_view_with_ref(ref,self.Stock,self.nptd[self.CTDidx],self.CTDidx==self.ETDidx )
        support_view_dic["flag_all_period_explored"]=False
        if self.CTDidx == self.SDTDEidx - self.PLen:  #this is to synchronize the stop for CC Eval
            support_view_dic["Flag_Force_Next_Reset"] = True
        else:
            support_view_dic["Flag_Force_Next_Reset"] = False
        return [lv,sv], support_view_dic, Flag_Done

class DBTP_Eval_WR_Reader(DBTP_Eval_CC_Reader):
    pass