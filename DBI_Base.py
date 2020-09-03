import os,json,random
from DB_Base import DB_Base
from DBR_Reader import RawData, Raw_HFQ_Index
from collections import OrderedDict
import pandas as pd
import numpy as np
'''

DBI data structure

/home/rdchujf/n_workspace/data/RL_data/I_DB
    Index/index_code.csv
    HFQ/stock.csv
    Stock_List/XXXXXXXX-XXXXXXXX/
    HFQ_Index_Update_log/lumpsum_HFQ_Inited_log.csv
              /addon_HFQ_log_xxxxxxxx.csv
    
    
    V1/StockID/YYYYMM/YYMMDD.pickle
    V1/Addonlog/YYYYMMDD.csv
    V1/Type_Definition.json
    

    V2/StockID/YYYYMM/YYMMDD.pickle
    V2/Addonlog/YYYYMMDD.csv
    V2/Type_Definition.json
    

   *sanitycheck time line base on last lumpsumdate from DBR_reader and files under Addonlog
'''
{
   "Elements":["DateI","Price_VS_Mount","Sell_Dan","Buy_Dan","Norm_Average_Nprice_And_Mount_Whole_Day",
               "Potential_Nprice_930","HFQ_Ratio","Exchange_Ratios"]
}

{
    "Elements":["Norm_Average_Nprice_And_Mount_Half_Day","Potential_Nprice_1300"]
}

class DBI_init(DB_Base):
    def __init__(self):
        DB_Base.__init__(self)
        self.IRD=RawData()
        self.IRHFQ=Raw_HFQ_Index("HFQ")
        self.IRIdx = Raw_HFQ_Index("Index")
        flag, mess=self.init_DBI_lumpsum_Indexes()
        assert flag,("inital lumpsum index in DBI fail with {0}".format(mess))
        self.init_DBI_lumpsum_HFQs()
        flag, self.nptd, mess=self.generate_TD()
        if not flag:
            raise  ValueError("Fail to generate TD with mess {0}".format(mess))

    def get_DBI_index_fnwp(self, index_code):
        assert index_code in self.DBI_Index_Code_List
        return os.path.join(self.Dir_DBI_index,"{0}.csv".format(index_code))

    def get_DBI_hfq_fnwp(self, stock):
        return os.path.join(self.Dir_DBI_HFQ, "{0}.csv".format(stock))

    def get_DBI_Update_Log_HFQ_Index_fnwp(self,DayI):
        return os.path.join(self.Dir_DBI_Update_Log_HFQ_Index, "{0}.csv".format(DayI))
    def get_DBI_Lumpsum_Log_HFQ_Index_fnwp(self):
        return os.path.join(self.Dir_DBI_Update_Log_HFQ_Index, self.DBI_HFQ_Inited_flag)

    def check_DBI_lumpsum_inited_Indexes(self):
        for index_code in self.DBI_Index_Code_List:
            fnwp=self.get_DBI_index_fnwp(index_code)
            if not os.path.exists(fnwp):
                return False, "File Not Found**** {0}".format(fnwp)
            flag, df, mess=self.get_index_df(fnwp)
            if not flag:
                return False, mess
            if len(df[df["date"]>=str(self.Raw_Normal_Lumpsum_EndDayI)])<1:
                return False, "Index Not Have Lumpsum End**** {0} does not exists in {1}".format(self.Raw_Normal_Lumpsum_EndDayI,fnwp)
        return True,"Success"

    def init_DBI_lumpsum_Indexes(self):
        flag,mess=self.check_DBI_lumpsum_inited_Indexes()
        if flag:
            return True, "Success"
        for index_code in self.DBI_Index_Code_List:
            flag,df,mess=self.IRIdx.get_lumpsum_df(index_code)
            if not flag: return False, mess
            df=df[df["date"] <= str(self.Raw_Normal_Lumpsum_EndDayI)]
            fnwp=self.get_DBI_index_fnwp(index_code)
            df.to_csv(fnwp,index=False)
        return True, "Success"

    def _check_1stock_HFQ_inited(self,stock):
        fnwp=self.get_DBI_hfq_fnwp(stock)
        if not os.path.exists(fnwp):
            return False, "File Not Found**** {0}".format(fnwp)
        flag,df, mess=self.get_hfq_df(fnwp)
        if not flag:
            return False, mess
        if len(df[df["date"]>=str(self.Raw_Normal_Lumpsum_EndDayI)])<1:
            return False, "HFQ Not Have Lumpsum End****{0} does not exists in {1}".format(self.Raw_Normal_Lumpsum_EndDayI,fnwp)
        return True,"Success"

    def init_lumpsum_1stock_HFQ(self,stock):
        flag, mess = self._check_1stock_HFQ_inited(stock)
        if flag:
            return True, "Success"
        flag,df,mess=self.IRHFQ.get_lumpsum_df(stock)
        if not flag: return False, mess
        df=df[df["date"] <= str(self.Raw_Normal_Lumpsum_EndDayI)]
        fnwp=self.get_DBI_hfq_fnwp(stock)
        df.to_csv(fnwp,index=False)
        return True, "Success"

    def init_DBI_lumpsum_HFQs(self):
        flag_HFQ_inited_fnwp =self.get_DBI_Lumpsum_Log_HFQ_Index_fnwp()
        if os.path.exists(flag_HFQ_inited_fnwp): return
        print ("Start Init lumpsum_HFQs for DBI")
        compressed_fnwp, decompressed_fnwp = self.IRD.get_normal_addon_raw_fnwp(self.Raw_Normal_Lumpsum_EndDayI, "XXXXXXXX", True)
        flag,mess=self.IRD.decompress_normal_addon_qz(self.Raw_Normal_Lumpsum_EndDayI, compressed_fnwp, os.path.dirname(decompressed_fnwp))
        if not flag:
            raise ValueError("Init all HFQ on {0} failed for {1}".format(self.Raw_Normal_Lumpsum_EndDayI,mess))
        logs=[]
        fns=os.listdir(os.path.dirname(decompressed_fnwp))
        fns.sort()
        for fn in fns:
            code= "{0}{1}".format("SH" if int(fn[:6])>=600000 else "SZ", fn[:6])
            flag,mess=self.init_lumpsum_1stock_HFQ(code)
            logs.append(["init HFQ",code, flag, mess])
            print ("\t",logs[-1])

        pd.DataFrame(logs, columns=self.title_DBI_HFQ_Inited_flag).to_csv(flag_HFQ_inited_fnwp, index=False)
        print("Finish Init lumpsum_HFQs for DBI. Log file in {0}".format(flag_HFQ_inited_fnwp))
        return

    def Reset_Init_Index_HFQ(self):
        for index_code in self.DBI_Index_Code_List:
            fnwp=self.get_DBI_index_fnwp(index_code)
            if os.path.exists(fnwp): os.remove(fnwp)
        flag_HFQ_inited_fnwp=os.path.join(self.Dir_DBI_Update_Log_HFQ_Index,self.DBI_HFQ_Inited_flag)
        if os.path.exists(flag_HFQ_inited_fnwp): os.remove(flag_HFQ_inited_fnwp)
        for fn in os.listdir(self.Dir_DBI_HFQ):
            os.remove(os.path.join(self.Dir_DBI_HFQ,fn))
        return

    def Update_DBI_addon_HFQ_Index(self, DayI):
        assert DayI >self.Raw_Normal_Lumpsum_EndDayI
        logfnwp=self.get_DBI_Update_Log_HFQ_Index_fnwp(DayI)
        if os.path.exists(logfnwp):
            return True, "Already Update DBI*****  {0} logfile exsists {1}".format(DayI,logfnwp)
        # check existance for raw files(fnwp)
        addon_HFQ_fnwp,_=self.IRHFQ.get_addon_fnwp(DayI)
        addon_Index_fnwp,_=self.IRIdx.get_addon_fnwp(DayI)
        addon_qz_compress_fnwp,addon_qz_decompress_fnwp=self.IRD.get_normal_addon_raw_fnwp(DayI,"XXXXXXXX",False)
        for fnwp in [addon_HFQ_fnwp,addon_Index_fnwp,addon_qz_compress_fnwp]:
            if not os.path.exists(fnwp):
                return False, "File Not Found**** {0}".format(fnwp)

        idx_flag,idx_rawdf,idx_mess=self.IRIdx.get_addon_df(DayI)
        if not idx_flag:
            Error_mess="{0} {1} {2}".format(idx_mess,DayI, idx_flag)
            print (Error_mess)
            return False,Error_mess
        hfq_flag, hfq_rawdf, hfq_mess = self.IRHFQ.get_addon_df(DayI)
        if not hfq_flag:
            Error_mess = "{0} {1} {2}".format(hfq_mess,DayI, hfq_mess)
            print(Error_mess)
            return False, Error_mess
        decompress_flag, decompress_mess=self.IRD.decompress_normal_addon_qz(DayI, addon_qz_compress_fnwp,os.path.dirname(addon_qz_decompress_fnwp))
        if not decompress_flag:
            Error_mess = "{0} {1}".format(decompress_mess,DayI)
            print(Error_mess)
            return False, Error_mess

        for index_code in self.DBI_Index_Code_List:
            fnwp=self.get_DBI_index_fnwp(index_code)
            flag,DBIdf, mess=self.get_index_df(fnwp)
            if not flag:
                return False, mess
            if DBIdf[DBIdf["date"] == str(DayI)].empty:
                input=idx_rawdf[idx_rawdf["code"]==index_code].iloc[0].values
                if input.size==0:
                    return False, "No Data for {0} at {1} in addon Index file".format(index_code, DayI)
                DBIdf.loc[len(DBIdf)]=input
                DBIdf.sort_values(by="date",inplace=True)
                DBIdf.to_csv(fnwp,index=False)
                print (["Addon indexes Update",index_code,True, "Success"])

        set_HFQ_StockL=set(hfq_rawdf["code"].tolist())
        set_qz_StockL=set(["{0}{1}".format("SH" if int(fn[:6]) >= 600000 else "SZ", fn[:6])
                   for fn in  os.listdir(os.path.dirname(addon_qz_decompress_fnwp))])
        HFQ_no_QZ_L=list(set_HFQ_StockL.difference(set_qz_StockL))
        QZ_no_HFQ_L=list(set_qz_StockL.difference(set_HFQ_StockL))
        HFQ_and_QZ_L=list(set_HFQ_StockL &set_qz_StockL)
        Log_List = []
        for ll in [[HFQ_no_QZ_L,"In raw HFQ not in raw QZ"],[QZ_no_HFQ_L,"In raw Qz not in raw HFQ"]]:
            for code in ll[0]:
                Log_List.append([code, False, ll[1]])
                print(Log_List[-1])
        for stock in HFQ_and_QZ_L:
            fnwp = self.get_DBI_hfq_fnwp(stock)
            if os.path.exists(fnwp):
                flag, DBIdf, mess = self.get_hfq_df(fnwp)
                if not flag:
                    Log_List.append([stock, False, "In raw HFQ in raw QZ****"+mess])
                    print(Log_List[-1])
                    continue
            else:
                DBIdf= pd.DataFrame(columns=self.title_hfq)
            if DBIdf[DBIdf["date"] == str(DayI)].empty:
                input=hfq_rawdf[hfq_rawdf["code"]==stock].values.tolist()
                assert  input.size!=0, "stock in raw addon HFQ and in raw QZ can not in this situation"
                DBIdf.loc[len(DBIdf)]=input[0]
                DBIdf.sort_values(by="date", inplace=True)
                DBIdf.to_csv(fnwp, index=False)
                Log_List.append([stock, True, "In raw HFQ in raw QZ****Success update"])
            else:
                Log_List.append([stock, True, "In raw HFQ in raw QZ****Date Already exisit"])
            print(Log_List[-1])
            continue

        pd.DataFrame(Log_List, columns=["stock", "status","mess"]).to_csv(logfnwp, index=False)
        print ("Finish Update_DBI_addon_HFQ_Index {0}. Log file in {1}".format(DayI,logfnwp))
        return True, "Success"

    ##TD related
    def generate_TD(self, input_index="SH000001"):
        fnwp=self.get_DBI_index_fnwp(input_index)
        if not os.path.exists((fnwp)): #raise ValueError("Not Found {0} in generating TD".format(fnwp))
            return False, "", "Index Not Found**** {0}".format(fnwp)
        flag, df, mess=self.get_index_df(fnwp)
        if not flag:
            return False, "", mess
        df["TD"]=df["date"].apply(lambda x: int(x))
        return True, df[df["TD"]>=self.TD_StartI]["TD"].values, "Success"

    def generate_TD_periods(self, StartI, EndI):
        return self.nptd[(self.nptd>=StartI) & (self.nptd<=EndI)]

    def get_closest_TD(self, DayI, Flag_Greater_or_Less):
        if Flag_Greater_or_Less:
            lidx=np.where(self.nptd>=DayI)
            assert len(lidx[0])>=1
            idx=lidx[0][0]
        else:
            lidx= np.where(self.nptd <= DayI)
            assert len(lidx[0]) >= 1
            idx=lidx[0][-1]
        return idx, self.nptd[idx]

class hfq_toolbox:
    def get_Nprice_from_hfq_price(self, hfq_price, hfq_ratio):
        return hfq_price/hfq_ratio
    def get_update_volume_on_hfq_ratio_change(self, old_hfq_ratio, new_hfq_ratio, old_volume):
        #return old_volume*old_hfq_ratio/new_hfq_ratio
        return old_volume *  new_hfq_ratio/ old_hfq_ratio
    def get_hfqprice_from_Nprice(self, Nprice, hfq_ratio):
        return Nprice * hfq_ratio


{
    "TLStartI":20180101,
    "TLEndI":20200531,
    "TrainNums": [600],
    "EvalNums":[150,150],
    "IPO_threadhold":100,
    "SL_Filter": "SH",
    "DBTP_Generator":
        [["Train", 0, 20180101, 20191231],
        ["Eval", 0, 20180101, 20191231],
        ["Eval", 1, 20200101, 20200531]]
}
class StockList(DBI_init):
    def __init__(self, SLName):
        DBI_init.__init__(self)
        self.SLName=SLName
        self.SL_wdn=os.path.join(self.Dir_DBI_SL,self.SLName)
        assert os.path.exists(self.SL_wdn), self.SL_wdn
        SL_Def_fnwp=os.path.join(self.SL_wdn,"SL_Definition.json")
        self.SLDef = json.load(open(SL_Def_fnwp, "r"), object_pairs_hook=OrderedDict)

    def TL_fnwp(self):
        return os.path.join(self.SL_wdn,"Total.csv")

    def Sub_fnwp(self, tag, idx):
        assert tag in ["Train", "Eval"]
        return os.path.join(self.SL_wdn, "{0}_{1}.csv".format( tag, idx))


    def get_total_stock_list(self):
        StartI          =   self.SLDef["TLStartI"]
        EndI            =   self.SLDef["TLEndI"]
        IPO_threadhold  =   self.SLDef["IPO_threadhold"]

        slfnwp=self.TL_fnwp()
        if os.path.exists(slfnwp):
            df = pd.read_csv(slfnwp, header=0, names=["stock"])
            return df["stock"].tolist()
        adjust_StartI=self.nptd[np.where(self.nptd>=StartI)[0][0]-IPO_threadhold]
        adjust_EndI=self.nptd[np.where(self.nptd<=EndI)[0][-1]]
        print (adjust_StartI, adjust_EndI)
        HFQlog_fnwp = os.path.join(self.Dir_DBI_Update_Log_HFQ_Index, self.DBI_HFQ_Inited_flag)
        df=pd.read_csv(HFQlog_fnwp, header=0,names=self.title_DBI_HFQ_Inited_flag)
        sl_source=df[df["result"]==True]["code"].tolist()
        sl_result=[]
        for code in sl_source:
            flag, df, mess = self.IRHFQ.get_lumpsum_df(code)
            if not flag:
                print (code,flag,mess  )
                print("skip {0}".format(code))
                continue
            if  (not df[df["date"]==str(adjust_StartI)].empty  and not df[df["date"]==str(adjust_EndI)].empty):
                print ("add {0}".format(code))
                sl_result.append(code)
            else:
                print("skip {0}".format(code))
        assert len(sl_result)>=1
        df=pd.DataFrame(sl_result,columns=["stock"])
        df.to_csv(slfnwp, index=False)
        print ("Create Total Stock List in {0}".format(slfnwp))
        return df["stock"].tolist()

    def generate_Train_Eval_stock_list(self):
        l_Train_num = self.SLDef["TrainNums"]
        l_Eval_num  = self.SLDef["EvalNums"]
        filter      = self.SLDef["SL_Filter"]
        Total_list=self.get_total_stock_list()
        if filter not in ["SH", "SZ"]:
            return False, "Filter Not Support****{0}".format(filter)
        adj_Total_List=[stock for stock in Total_list if filter in stock]
        if  len(adj_Total_List)<sum(l_Train_num)+sum(l_Eval_num):
            return False, "Numb Stock Not Enought****adj total list len {0} less than the needed Train and Eval toal stock num {1}".\
                format(len(adj_Total_List),sum(l_Train_num)+sum(l_Eval_num))
        random.shuffle(adj_Total_List)
        for l_num, tag in zip([l_Train_num,l_Eval_num],["Train", "Eval"]):
            for id,num in enumerate(l_num):
                extract_list=adj_Total_List[:num]
                fnwp=self.Sub_fnwp(tag,id)
                pd.DataFrame(extract_list,columns=["stock"]).to_csv(fnwp, index=False)
                print("Create Sub Stock List in {0}".format(fnwp))
                adj_Total_List=adj_Total_List[num:]
        return True, "Success"

    def get_sub_sl(self, tag, index):
        fnwp=self.Sub_fnwp(tag, index)
        if not os.path.exists(fnwp):
            return False, ""
        else:
            return True, pd.read_csv(fnwp, header=0, names=["stock"])["stock"].tolist()

class DBI_Base(DBI_init):
    def __init__(self, DBI_name):
        DBI_init.__init__(self)
        self.DBI_name=DBI_name
        self.Dir_DBI_WP=os.path.join(self.Dir_IDB,self.DBI_name)
        self.Dir_DBI_WP_Data=os.path.join(self.Dir_DBI_WP,"Data")
        self.Dir_DBI_WP_Log=os.path.join(self.Dir_DBI_WP,"Log")

        for dir in [self.Dir_DBI_WP]:
            assert os.path.exists(dir),"DBI name folder with its Type_Definition.json should be ready before run the program"
        for dir in [self.Dir_DBI_WP_Data,self.Dir_DBI_WP_Log]:
            if not os.path.exists(dir): os.makedirs(dir)
        Type_fnwp=os.path.join(self.Dir_DBI_WP,"DBI_Definition.json")
        self.TypeDefinition = json.load(open(Type_fnwp, "r"), object_pairs_hook=OrderedDict)

    def get_DBI_data_fnwp(self, stock, dayI):
        dn=self.Dir_DBI_WP_Data
        for subdir in [stock, str(dayI//100)]:
            dn = os.path.join(dn,subdir)
            if not os.path.exists(dn): os.mkdir (dn)
        return os.path.join(dn,"{0}.pickle".format(dayI))

    def get_DBI_log_fnwp(self, stock):
        return os.path.join(self.Dir_DBI_WP_Log,"{0}.csv".format(stock))


'''
class ginfo_one_stock:
    def __init__(self, stock):
        self.stock = stock

        self.i_tdate=API_trade_date()
        self.df_hfq=API_HFQ_from_file().get_df_HFQ(self.stock)
    def check_not_tinpai(self,date_s):
        assert self.i_tdate.check_trading_date(date_s)
        #return np.any( self.df_hfq["date"] == date_s)
        return date_s in self.df_hfq["date"].values
    def check_after_IPO(self, date_s):
        assert self.i_tdate.check_trading_date(date_s)
        IPO_date_s=self.df_hfq["date"][0]
        return date_s>=IPO_date_s

    def hfq_amount(self, date_s):  #this is called after check_not_tinpai
        return self.df_hfq[self.df_hfq["date"] == date_s].iloc[0]["amount_gu"]
    def hfq_ratio(self,date_s): #this is called after check_not_tinpai
        return self.df_hfq[self.df_hfq["date"]==date_s].iloc[0]["coefficient_fq"]

    def get_closest_close_Nprice(self, date_s):
        df=self.df_hfq[self.df_hfq["date"] <= date_s]
        assert len(df)>0, "no_price_before_that_day {0} {1}".format(self.stock,date_s)
        hfq_price = df.iloc[-1]["close_price"]
        hfq_ratio = df.iloc[-1]["coefficient_fq"]
        Nprice=hfq_toolbox().get_Nprice_from_hfq_price(hfq_price,hfq_ratio )

        return Nprice,hfq_ratio

    def get_open_price(self,date_s): #this is called after check_not_tinpai
        df=self.df_hfq[self.df_hfq["date"] == date_s]
        assert len(df)>0
        hfq_price = df.iloc[0]["open_price"]
        hfq_ratio = df.iloc[0]["coefficient_fq"]
        Nprice = hfq_toolbox().get_Nprice_from_hfq_price(hfq_price, hfq_ratio)
        return Nprice, hfq_ratio

    def get_exchange_ratio_for_tradable_part(self, date_s):
        return self.df_hfq[self.df_hfq["date"] == date_s].iloc[0]["exchange_ratio_for_tradable_part"]
'''
