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

        #print (self.FT_TitlesD_Flat[2])
        #print (ref)
        #print (ref[-1, found_idxs[0]])
        #assert False

        #return ref[-1,found_idxs[0]]

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
        SD=pd.read_csv(DBTP_log_fnwp, header=0, names=["Result", "Date", "Message"],
                            dtype={"Result":str,"Date":int,"Message":str})["Date"].values
        SD.sort()
        SD= SD[(SD>=SDateI) & (SD<=EDateI)]

        assert len(self.generate_TD_periods(SD[0],SD[-1]))==len(SD), \
            "{0}  len(self.generate_TD_periods(self.SD[0],self.SD[-1])) = {1} len(self.SD)= {2} ".format(
                Stock,len(self.generate_TD_periods(SD[0], SD[-1])),len(SD))
        self.SDTDSidx, _ = self.get_closest_TD(SD[0], True)
        self.SDTDEidx, _ = self.get_closest_TD(SD[-1], False)

        assert self.SDTDEidx-self.SDTDSidx>self.PLen, \
            "self.SDTDEidx-self.SDTDSidx={0}<=PLen={1}".format(self.SDTDEidx-self.SDTDSidx,self.PLen)

        self.CTDidx = self.SDTDSidx
        self.ETDidx = self.SDTDEidx


    def reset_get_data(self):
        self.CTDidx = random.randrange(self.SDTDSidx,self.SDTDEidx-self.PLen)
        self.ETDidx = self.CTDidx + self.PLen
        lv, sv, ref=self.read_1day_TP_Data(self.Stock, self.nptd[self.CTDidx])
        #lv, sv, ref=pickle.load(open(self.get_DBTP_data_fnwp(self.Stock, self.nptd[self.CTDidx]), "rb"))
        support_view_dic=self.Fill_support_view_with_ref(ref,self.Stock, self.nptd[self.CTDidx],self.CTDidx==self.ETDidx )
        support_view_dic["flag_all_period_explored"]=False
        return [lv,sv],support_view_dic

    def next_get_data(self):
        self.CTDidx+=1
        Flag_Done=True if self.CTDidx==self.ETDidx else False
        lv, sv, ref = self.read_1day_TP_Data(self.Stock, self.nptd[self.CTDidx])
        #lv, sv, ref=pickle.load(open(self.get_DBTP_data_fnwp(self.Stock, self.nptd[self.CTDidx]), "rb"))
        support_view_dic=self.Fill_support_view_with_ref(ref,self.Stock,self.nptd[self.CTDidx],self.CTDidx==self.ETDidx )
        support_view_dic["flag_all_period_explored"]=False
        return [lv,sv], support_view_dic, Flag_Done

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

class DBTP_BT_Reader(DBTP_Train_Reader):
    def __init__(self, DBTP_Name, Stock, SDateI, EDateI,PLen=30):
        DBTP_Train_Reader.__init__(self, DBTP_Name, Stock, SDateI, EDateI,PLen)

    def reset_get_data(self):
        self.CTDidx += 1
        self.ETDidx = self.CTDidx + self.PLen
        lv, sv, ref = self.read_1day_TP_Data(self.Stock, self.nptd[self.CTDidx])
        #lv, sv, ref=pickle.load(open(self.get_DBTP_data_fnwp(self.Stock, self.nptd[self.CTDidx]), "rb"))
        support_view_dic=self.Fill_support_view_with_ref(ref,self.Stock,self.nptd[self.CTDidx],self.CTDidx==self.ETDidx)
        support_view_dic["flag_all_period_explored"] = True if self.CTDidx>=self.SDTDEidx else False
        return [lv, sv], support_view_dic
