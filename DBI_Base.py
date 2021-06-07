import os,json,random,pickle
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
               "Potential_Nprice_930","HFQ_Ratio","Exchange_Ratios"],
    "TinpaiFun":"TinpaiZero"
}

{
    "Elements":["Norm_Average_Nprice_And_Mount_Half_Day","Potential_Nprice_1300"],
    "TinpaiFun": "TinpaiNAN"
}

class DBI_init(DB_Base):
    def __init__(self):
        DB_Base.__init__(self)
        self.IRD=RawData()
        self.IRHFQ=Raw_HFQ_Index("HFQ")
        self.IRIdx = Raw_HFQ_Index("Index")


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
        # Following is to ensure all the raw hfq file has lower sh or sz name
        for fn in self.Dir_raw_Index_base:
            if fn.startswith("SH") or fn.startswith("SZ"):
                os.rename(os.path.join(self.Dir_raw_Index_base, fn), os.path.join(self.Dir_raw_Index_base, fn.lower()))

        for index_code in self.DBI_Index_Code_List:
            flag,df,mess=self.IRIdx.get_lumpsum_df(index_code)
            if not flag: return False, mess
            df=df[df["date"] <= str(self.Raw_Normal_Lumpsum_EndDayI)]
            df.drop_duplicates(subset=["date"],inplace=True)  # should set subset as duplicate row has minor difference in float
            assert not any(df.duplicated(["date"])),"these are duplicated rows {0}".format(df[df.duplicated(["date"])])
            df.reset_index(inplace=True)
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
        df.drop_duplicates(subset=["date"],inplace=True)  #should set subset as duplicate row has minor difference in float
        assert not any(df.duplicated(["date"])),"these are duplicated rows {0}".format(df[df.duplicated(["date"])])
        df.reset_index(inplace=True)
        fnwp=self.get_DBI_hfq_fnwp(stock)
        df.to_csv(fnwp,index=False)
        return True, "Success"

    def init_DBI_lumpsum_HFQs(self):
        flag_HFQ_inited_fnwp =self.get_DBI_Lumpsum_Log_HFQ_Index_fnwp()
        if os.path.exists(flag_HFQ_inited_fnwp): return
        print ("Start Init lumpsum_HFQs for DBI")

        #Following is to ensure all the raw hfq file has uppder SH or SZ name
        for fn in self.Dir_raw_HFQ_base:
            if fn.startswith("sh") or fn.startswith("sz"):
                os.rename(os.path.join(self.Dir_raw_HFQ_base, fn), os.path.join(self.Dir_raw_HFQ_base, fn.upper()))

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
        #flag_HFQ_inited_fnwp=os.path.join(self.Dir_DBI_Update_Log_HFQ_Index,self.DBI_HFQ_Inited_flag)
        flag_HFQ_inited_fnwp=self.get_DBI_Lumpsum_Log_HFQ_Index_fnwp()
        if os.path.exists(flag_HFQ_inited_fnwp): os.remove(flag_HFQ_inited_fnwp)
        for fn in os.listdir(self.Dir_DBI_HFQ):
            os.remove(os.path.join(self.Dir_DBI_HFQ,fn))
        return


    def Update_DBI_addon(self, DayI):
        assert DayI > self.Raw_Normal_Lumpsum_EndDayI
        logfnwp=self.get_DBI_Update_Log_HFQ_Index_fnwp(DayI)
        if os.path.exists(logfnwp):
            return True, "Already Update DBI*****  {0} logfile exsists {1}".format(DayI,logfnwp)

        addon_HFQ_fnwp, _ = self.IRHFQ.get_addon_fnwp(DayI)
        addon_qz_compress_fnwp, addon_qz_decompress_fnwp = self.IRD.get_normal_addon_raw_fnwp(DayI, "XXXXXXXX", False)
        for fnwp in [addon_HFQ_fnwp,addon_qz_compress_fnwp]:
            if not os.path.exists(fnwp):
                return False, "File Not Found**** {0}".format(fnwp)

        hfq_flag, hfq_rawdf, hfq_mess = self.IRHFQ.get_addon_df(DayI)
        if not hfq_flag:
            Error_mess = "{0} {1} {2}".format(hfq_mess,DayI, hfq_mess)
            return False, Error_mess
        else:
            print (hfq_mess)
        decompress_flag, decompress_mess=self.IRD.decompress_normal_addon_qz(DayI, addon_qz_compress_fnwp,os.path.dirname(addon_qz_decompress_fnwp))
        if not decompress_flag:
            Error_mess = "{0} {1}".format(decompress_mess,DayI)
            return False, Error_mess
        else:
            print (decompress_mess)

        for index_code in self.DBI_Index_Code_List:
            fnwp=self.get_DBI_index_fnwp(index_code)
            flag,DBIdf, mess=self.get_index_df(fnwp)
            if not flag:
                return False, mess
            if DBIdf[DBIdf["date"] == str(DayI)].empty:
                df_found=hfq_rawdf[hfq_rawdf["code"] == index_code]
                assert len(df_found)==1, "{0}raw HFQ_index file {1} record {2}".format(DayI, index_code,len(df_found) )
                input=df_found.iloc[0].values
                if input.size==0:
                    return False, "No Data for {0} at {1} in addon HFQ_index file".format(index_code, DayI)
                DBIdf.loc[len(DBIdf)]=input[:-3]  #the last three column is HFQ related inform
                DBIdf.sort_values(by="date",inplace=True)
                DBIdf.to_csv(fnwp,index=False)
                print (["Addon indexes Update",index_code,True, "Success"])
            else:
                print(["Addon indexes Update", index_code, True, "Already updated"])


        qz_StockL=["{0}{1}".format("SH" if int(fn[:6]) >= 600000 else "SZ", fn[:6]) for fn in os.listdir(os.path.dirname(addon_qz_decompress_fnwp))]
        Log_List = []
        CSuccess, CSuccessNew, CAlredayExist,CError=0,0,0,0
        for stock in qz_StockL:
            fnwp = self.get_DBI_hfq_fnwp(stock)
            if os.path.exists(fnwp):
                flag, DBIdf, mess = self.get_hfq_df(fnwp)
                if not flag:
                    Log_List.append([stock, False, "In raw QZ, DBI HFQ found but fail to read****"+mess])
                    print(Log_List[-1])
                    CError+=1
                    continue
            else:
                DBIdf= pd.DataFrame(columns=self.title_hfq)
            if DBIdf[DBIdf["date"] == str(DayI)].empty:
                df_found=hfq_rawdf[hfq_rawdf["code"] == stock]
                assert len(df_found)==1, "{0}raw HFQ_index file {1} record {2}".format(DayI, stock,len(df_found) )
                input=df_found.iloc[0].values
                assert  input.size!=0, "No Data for {0} at {1} in addon HFQ_index file".format(stock, DayI)
                DBIdf.loc[len(DBIdf)]=input
                DBIdf.sort_values(by="date", inplace=True)
                DBIdf.to_csv(fnwp, index=False)
                if len(DBIdf)==1:
                    Log_List.append([stock, True, "In raw HFQ in raw QZ****Generate New HFQ and Success update"])
                    CSuccessNew+=1
                else:
                    Log_List.append([stock, True, "In raw HFQ in raw QZ****Success update"])
                    CSuccess+=1
            else:
                Log_List.append([stock, True, "In raw HFQ in raw QZ****Date Already exists"])
                CAlredayExist+=1
            print(Log_List[-1])
        dflog=pd.DataFrame(Log_List, columns=["stock", "status","mess"])
        dflog.to_csv(logfnwp, index=False)
        print ("Finish Update_DBI_addon_HFQ_Index {0}. Log file in {1}".format(DayI,logfnwp))
        print ("Success Update: {0} Success Update Generate New HFQ: {1} Already Updated before:{2} Error: {3}".
               format(CSuccess, CSuccessNew, CAlredayExist,CError))
        assert len(dflog)== CSuccess+ CSuccessNew+ CAlredayExist+CError
        return True, "Success"


class DBI_init_with_TD(DBI_init):
    def __init__(self):
        DBI_init.__init__(self)
        flag, mess=self.check_DBI_lumpsum_inited_Indexes()
        if not flag:
            assert False,"lumpsum idexes not inited yet {0}".format(mess)
        flag_HFQ_inited_fnwp =self.get_DBI_Lumpsum_Log_HFQ_Index_fnwp()
        if not os.path.exists(flag_HFQ_inited_fnwp):
            assert False,"lumpsum HFQs not inited yet {0} not exist".format(flag_HFQ_inited_fnwp)

        flag, self.nptd, mess=self.generate_TD()
        if not flag:
            raise  ValueError("Fail to generate TD with mess {0}".format(mess))

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
'''
class stock_code:
    "SH":[600,601,603,605]  #沪A
    "SZ":[000]          #深A
    "SZ":[002]          #中小板
    "SZ":[300]          #创业板
    "SH":[688]          #科创板
'''

{
    "FunGenTSL":"TSL_from_caculate",
    "ParamGenSL":{
        "TLStartI":         20180101,
        "TLEndI":           20200531,
        "IPO_threadhold":   100,
        "SL_Filter":        "SH",
        "TrainNums":        [600],
        "EvalNums": [150, 150],
    },
    "DBTP_Generator":
        [["Train", 0, 20180101, 20191231],
        ["Eval", 0, 20180101, 20191231],
        ["Eval", 1, 20200101, 20200531]]
}

{
    "FunGenTSL":"TSL_from_300",
    "ParamGenTSL":{
    },
    "DBTP_Generator":
        [["Train", 0, 20180101, 20210226]]
}


class StockList(DBI_init_with_TD):
    def __init__(self, SLName):
        DBI_init_with_TD.__init__(self)
        self.SLName=SLName
        self.SL_wdn=os.path.join(self.Dir_DBI_SL,self.SLName)
        assert os.path.exists(self.SL_wdn), self.SL_wdn
        SL_Def_fnwp=os.path.join(self.SL_wdn,"SL_Definition.json")
        self.SLDef = json.load(open(SL_Def_fnwp, "r"), object_pairs_hook=OrderedDict)
        assert self.SLDef["FunGenTSL"] in ["TSL_from_caculate","TSL_from_50", "TSL_from_300","TSL_from_500","TSL_from_try"]
        if self.SLDef["FunGenTSL"] == "TSL_from_caculate":
            self.generate_Train_Eval_SL=self.generate_Train_Eval_SL_for_TSL_from_caculate
        else:
            self.generate_Train_Eval_SL = self.generate_Train_Eval_SL_for_TSL_from_src
            assert len(self.SLDef["ParamGenTSL"])==0

    #common
    def TL_fnwp(self):
        return os.path.join(self.SL_wdn,"Total.csv")

    def Sub_fnwp(self, tag, idx):
        assert tag in ["Train", "Eval"]
        return os.path.join(self.SL_wdn, "{0}_{1}.csv".format( tag, idx))

    def Sanity_Check_SL(self, sl, flag_remove_adj, flag_remove_price):
        adj_fnwp=os.path.join(self.SL_wdn,"Adj_to_Remove.csv")
        if flag_remove_adj and os.path.exists(adj_fnwp):
            sl_remove=pd.read_csv(adj_fnwp,header=0, names=["stock"])["stock"].tolist()
            sl= list(set(sl) - set(sl_remove))
        Price_adj_fnwp=os.path.join(self.SL_wdn,"Price_to_Remove.csv")
        if flag_remove_price and os.path.exists(Price_adj_fnwp):
            sl_remove=pd.read_csv(Price_adj_fnwp,header=0, names=["Stock","Reason"])["Stock"].tolist()
            sl= list(set(sl) - set(sl_remove))
        return sl

    def Get_Total_SL(self):
        slfnwp=self.TL_fnwp()
        if os.path.exists(slfnwp):
            df = pd.read_csv(slfnwp, header=0, names=["stock"])
            return df["stock"].tolist()

        df=getattr(self,self.SLDef["FunGenTSL"])()
        df.to_csv(slfnwp, index=False, header=["stock"])
        return df["stock"].tolist()

    def get_sub_sl(self, tag, index):
        fnwp=self.Sub_fnwp(tag, index)
        if not os.path.exists(fnwp):
            return False, ""
        else:
            return True, self.Sanity_Check_SL(pd.read_csv(fnwp, header=0, names=["stock"])["stock"].tolist(),
                                              flag_remove_adj=True, flag_remove_price=True)
    #legacy
    def TSL_from_caculate(self):
        try:
            StartI = self.SLDef["ParamGenSL"]["TLStartI"]
            EndI = self.SLDef["ParamGenSL"]["TLEndI"]
            IPO_threadhold = self.SLDef["ParamGenSL"]["IPO_threadhold"]
            filter = self.SLDef["ParamGenSL"]["SL_Filter"]
        except:
            assert False, "TSL_from_caculate missing param"
        assert filter in ["SH", "SZ"], "Filter Not Support****{0}".format(filter)
        adjust_StartI = self.nptd[np.where(self.nptd >= StartI)[0][0] - IPO_threadhold]
        adjust_EndI = self.nptd[np.where(self.nptd <= EndI)[0][-1]]
        #HFQlog_fnwp = os.path.join(self.Dir_DBI_Update_Log_HFQ_Index, self.DBI_HFQ_Inited_flag)
        HFQlog_fnwp =self.get_DBI_Lumpsum_Log_HFQ_Index_fnwp()
        df = pd.read_csv(HFQlog_fnwp, header=0, names=self.title_DBI_HFQ_Inited_flag)
        sl_source = df[df["result"] == True]["code"].tolist()
        sl_result = []
        for code in sl_source:
            flag, df, mess = self.IRHFQ.get_lumpsum_df(code)
            if not flag:
                print(code, flag, mess)
                print("skip {0} due to no HFQ data".format(code))
                continue
            if (not df[df["date"] == str(adjust_StartI)].empty and not df[df["date"] == str(adjust_EndI)].empty):
                print("add {0}".format(code))
                sl_result.append(code)
            else:
                print("skip {0} due to data not exists for {1} {2} ".format(code, adjust_StartI,adjust_EndI))
        assert len(sl_result) >= 1

        adj_Total_List = [stock for stock in sl_result if filter in stock]
        assert len(adj_Total_List) >= 1
        return pd.DataFrame(adj_Total_List, columns=["stock"])
    def generate_Train_Eval_SL_for_TSL_from_caculate(self):
        l_Train_num = self.SLDef["ParamGenSL"]["TrainNums"]
        l_Eval_num  = self.SLDef["ParamGenSL"]["EvalNums"]

        Total_list=self.Get_Total_SL()
        if  len(Total_list)<sum(l_Train_num)+sum(l_Eval_num):
            print ("Numb Stock Not Enought****adj total list len {0} less than the needed Train and Eval toal stock num {1}".\
                format(len(Total_list),sum(l_Train_num)+sum(l_Eval_num)))
            return False
        random.shuffle(Total_list)
        for l_num, tag in zip([l_Train_num,l_Eval_num],["Train", "Eval"]):
            for id,num in enumerate(l_num):
                extract_list=Total_list[:num]
                fnwp=self.Sub_fnwp(tag,id)
                pd.DataFrame(extract_list,columns=["stock"]).to_csv(fnwp, index=False)
                print("Create Sub Stock List in {0}".format(fnwp))
                Total_list=Total_list[num:]
        return True


    # 50, 300, 500
    def TSL_from_50(self):
        assert len(self.SLDef["ParamGenTSL"])==0, "TSL_from_50 does not need param"
        return pd.read_csv(os.path.join(self.Dir_DBI_SL, "Stock50.csv"))[["stock"]]
    def TSL_from_300(self):
        assert len(self.SLDef["ParamGenTSL"]) == 0, "TSL_from_300 does not need param"
        return pd.read_csv(os.path.join(self.Dir_DBI_SL, "Stock300.csv"))[["stock"]]

    def TSL_from_500(self):
        assert len(self.SLDef["ParamGenTSL"]) == 0, "TSL_from_300 does not need param"
        return pd.read_csv(os.path.join(self.Dir_DBI_SL, "Stock500.csv"))[["stock"]]

    def generate_Train_Eval_SL_for_TSL_from_src(self):
        Total_list = self.Get_Total_SL()
        for tag in ["Train", "Eval"]:
            fnwp=self.Sub_fnwp(tag,0)
            pd.DataFrame(Total_list,columns=["stock"]).to_csv(fnwp, index=False)
            print("Create Sub Stock List in {0}".format(fnwp))
        random.shuffle(Total_list)
        total_num=len(Total_list)
        fnwp = self.Sub_fnwp("Train", 1)
        pd.DataFrame(Total_list[:int(total_num*0.7)], columns=["stock"]).to_csv(fnwp, index=False)
        print("Create Sub Stock List in {0}".format(fnwp))
        fnwp = self.Sub_fnwp("Eval", 1)
        pd.DataFrame(Total_list[int(total_num*0.7):], columns=["stock"]).to_csv(fnwp, index=False)
        print("Create Sub Stock List in {0}".format(fnwp))
        return True

    def Get_Stocks_Error_Generate_DBTP(self):
        logdn=os.path.join(self.SL_wdn,"CreateLog")
        l_fnwp = [os.path.join(logdn, fn) for fn in os.listdir(logdn) if
                  "Error" in fn and os.path.getsize(os.path.join(logdn, fn)) != 0]
        l_stock_to_remove = []
        for fnwp in l_fnwp:
            df = pd.read_csv(fnwp, names=["stock", "day", "mess"])
            l_stock_to_remove.extend(list(set(df["stock"].tolist())))
        if len(l_stock_to_remove)!=0:
            adj_fnwp=os.path.join(self.SL_wdn,"Adj_to_Remove.csv")
            pd.DataFrame(l_stock_to_remove, columns=["stock"]).to_csv(adj_fnwp, index=False)
            print ("TPDB Error generate Stocks saved to {0}".format(adj_fnwp))
        else:
            print("All Stocks succesfully generate TPDB")
        return

    def Get_Eval_SubProcess_SL(self, lc,process_group_idx,process_idx):
        process_idx_left = process_idx % lc.eval_num_process_each_group
        SL_idx, self.SL_StartI, self.SL_EndI = lc.l_eval_SL_param[process_group_idx]
        flag, group_stock_list = self.get_sub_sl("Eval", SL_idx)

        assert flag, "Get Stock list {0} tag=\"Eval\" index={1}".format(self.SLName, process_group_idx)
        mod=len(group_stock_list)//lc.eval_num_process_each_group
        left=len(group_stock_list)%lc.eval_num_process_each_group
        stock_list = group_stock_list[process_idx_left * mod:(process_idx_left + 1) * mod]
        if process_idx_left<left:
            stock_list.append(group_stock_list[-(process_idx_left+1)])
        return stock_list

    #todo move get_exceed_max_price_sl to Stock list
    def get_exceed_max_price_sl(self,set_max_price):
        flag, sl = self.get_sub_sl("Train", 0)
        assert flag
        esl = []
        for stock in sl:
            flag, df, mess = self.get_hfq_df(self.get_DBI_hfq_fnwp(stock))
            assert flag
            max_price = df["open_price"].max()
            if max_price >= set_max_price:
                print("{0} should exclude for exceeding {1}".format(stock, set_max_price))
                esl.append([stock, "price_too_high"])
            else:
                print(stock, " ok")

        df = pd.DataFrame(esl, columns=["Stock", "Reason"])
        #fnwp = os.path.join("/home/rdchujf/n_workspace/data/RL_data/I_DB/Stock_List", sl_name, "Price_to_Remove.csv")
        fnwp = os.path.join(self.Dir_DBI_SL, self.SLName, "Price_to_Remove.csv")
        df.to_csv(fnwp, index=False)
        print ("stock has max price exceed {0} be stored in {1}".format(set_max_price,fnwp))


class DBI_Base(DBI_init_with_TD):
    def __init__(self, DBI_name):
        DBI_init_with_TD.__init__(self)
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
        if "TinpaiFun" not in self.TypeDefinition.keys():
            self.TypeDefinition["TinpaiFun"]="TinpaiZero"
        else:
            assert self.TypeDefinition["TinpaiFun"] in ["TinpaiNAN","TinpaiZero"]

    def get_DBI_data_fnwp(self, stock, dayI):
        dn=self.Dir_DBI_WP_Data
        for subdir in [stock, str(dayI//100)]:
            dn = os.path.join(dn,subdir)
            if not os.path.exists(dn): os.mkdir (dn)
        return os.path.join(dn,"{0}.pickle".format(dayI))

    def get_DBI_log_fnwp(self, stock):
        return os.path.join(self.Dir_DBI_WP_Log,"{0}.csv".format(stock))

    def Is_DBI_Oneday_exists(self, stock, dayI):
        return os.path.exists(self.get_DBI_data_fnwp(stock, dayI))

    def load_DBI(self, stock, dayI):
        return pickle.load(open(self.get_DBI_data_fnwp(stock, dayI), "rb"))

    def dump_DBI(self, stock, dayI, result_L):
        pickle.dump(result_L, open(self.get_DBI_data_fnwp(stock, dayI), 'wb'))

    def print_DBI(self,stock, dayI):
        data_L=self.load_DBI(stock, dayI)
        for data, title in zip(data_L,self.TypeDefinition["Elements"]):
            print (title, data)
