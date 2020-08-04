import os
import pandas as pd
class DB_Base:
    #general param
    SDir_IDB="I_DB"
    SDir_TPDB="TP_DB"

    Dir_DB_Base = "/home/rdchujf/n_workspace/data/RL_data"

    ##raw lumpsum addon data param
    Dir_raw_legacy_1        =   "/home/rdchujf/Stk_qz"
    Dir_raw_legacy_2        =   "/home/rdchujf/Stk_qz_2"
    Dir_raw_normal          = "/media/rdchujf/2G"
    Dir_raw_normal_addon    ="/media/rdchujf/2G"

    Dir_raw_normal_decompressed="/media/rdchujf/2G/decompress"
    Dir_raw_normal_addon_decompressed = "/media/rdchujf/2G/decompress"

    Dir_raw_HFQ_base        =   "/home/rdchujf/Stk_qz_3_support/Stk_Day_FQ_WithHS"
    Dir_raw_HFQ_base_addon = "/home/rdchujf/Stk_qz_3_support/Stk_Day_FQ_WithHS_addon"
    # "/home/rdchujf/Stk_qz_3_support/Stk_Day_FQ_WithHS_addon/YYYYMM/YYYYMMDD.rar"
    Dir_raw_HFQ_base_addon_decompressed = "/home/rdchujf/Stk_qz_3_support/Stk_Day_FQ_WithHS_addon_decompress"

    Dir_raw_Index_base      =   "/home/rdchujf/Stk_qz_3_support/Stk_Day_Idx"
    Dir_raw_Index_base_addon = "/home/rdchujf/Stk_qz_3_support/Stk_Day_Idx_addon"
    #"/home/rdchujf/Stk_qz_3_support/Stk_Day_Idx_addon/YYYYMM/YYYYMMDD.rar
    Dir_raw_Index_base_addon_decompressed= "/home/rdchujf/Stk_qz_3_support/Stk_Day_Idx_addon_decompress"


    #df structure
    title_qz = ["TranID", "Time", "Price", "Volume", "SaleOrderVolume", "BuyOrderVolume", "Type",
                     "SaleOrderID","SaleOrderPrice", "BuyOrderID", "BuyOrderPrice"]
    dtype_qz = {
        "TranID": int, "Time": str, "Price": float,
        "Volume": int, "SaleOrderVolume": int, "BuyOrderVolume": int, "Type": str,
        "SaleOrderID": int, "SaleOrderPrice": float, "BuyOrderID": int, "BuyOrderPrice": float}

    title_hfq = ["code", "date", "open_price", "highest_price", "lowest_price", "close_price", "amount_gu",
                      "amount_yuan", "exchange_ratio_for_tradable_part", "exchange_ratio_for_whole",
                      "coefficient_fq"]
    dtype_hfq = {"code": str, "date": str, "open_price": float, "highest_price": float, "lowest_price": float,
                      "close_price": float, "amount_gu": int, "amount_yuan": float,
                      "exchange_ratio_for_tradable_part": float, "exchange_ratio_for_whole": float,
                      "coefficient_fq": float}
    title_index = ["code", "date", "open_price", "highest_price", "lowest_price", "close_price", "amount_gu",
                        "amount_yuan"]
    dtype_index = {"code": str, "date": str, "open_price": float, "highest_price": float, "lowest_price": float,
                        "close_price": float, "amount_gu": int, "amount_yuan": float}

    ##DBI related
    DBI_Index_Code_List=["SH000001","SZ399001"]
    DBI_HFQ_Inited_flag="lumpsum_HFQ_Inited_log.csv"  #should under addonlog dir
    title_DBI_HFQ_Inited_flag=    ["operation", "code", "result", "message"]


    def __init__(self,Raw_lumpsum_End_DayI=20200529):
        self.TD_StartI = 20100101
        self.Raw_legacy_Lumpsum_StartDayI = 20130415
        self.Raw_Legacy_Lumpsum_EndDayI = 20171229
        assert Raw_lumpsum_End_DayI>self.Raw_Legacy_Lumpsum_EndDayI
        self.Raw_Normal_Lumpsum_EndDayI = Raw_lumpsum_End_DayI#20200529

        self.Dir_IDB=os.path.join(self.Dir_DB_Base,self.SDir_IDB)
        self.Dir_DBI_index = os.path.join(self.Dir_IDB, "index")
        self.Dir_DBI_HFQ = os.path.join(self.Dir_IDB, "HFQ")
        self.Dir_DBI_Update_Log_HFQ_Index=os.path.join(self.Dir_IDB, "Update_Log_HFQ_Index")
        self.Dir_TPDB=os.path.join(self.Dir_DB_Base,self.SDir_TPDB)

        for dir in [self.Dir_DB_Base,self.Dir_IDB,self.Dir_TPDB,
                    self.Dir_DBI_index,self.Dir_DBI_HFQ,self.Dir_DBI_Update_Log_HFQ_Index,
                    self.Dir_raw_HFQ_base_addon,
                    self.Dir_raw_Index_base_addon,self.Dir_raw_HFQ_base_addon_decompressed,
                    self.Dir_raw_Index_base_addon_decompressed]:
            if not os.path.exists(dir):os.makedirs(dir)

    def get_qz_df(self, fnwp):
        try:
            df = pd.read_csv(fnwp, header=0, names=self.title_qz, dtype=self.dtype_qz)
        except ValueError as e:
            # this is to handle 13 in file as 12.99999997 situation
            if "cannot safely convert passed user dtype of int64 for float64 dtyped data" in str(e):
                df = pd.read_csv(fnwp, header=0, names=self.title_qz)
                for item in list(self.dtype_qz.keys()):
                    df[item] = df[item].astype(self.dtype_qz[item])
            # 6883,undefined,undefined,NaN,NaN,NaN,undefined,undefined,undefined,undefined,undefined
            # error message could not convert string to float: undefined
            elif "undefined" in str(e):
                df = pd.read_csv(fnwp, header=0, names=self.title_qz)
                df.dropna(inplace=True)
                for item in list(self.dtype_qz.keys()):
                    df[item] = df[item].astype(self.dtype_qz[item])
            else:
                return False, "", "Read QZ DF Error**** {0} {1}".format(fnwp,str(e))

        df.drop(["TranID"], axis=1, inplace=True)
        if df.empty:
            return False,"", "QZ DF Empty****{0}".format(fnwp)
            #raise ValueError("{0} is empty".format(fnwp))

        df["Time"] = df["Time"].apply(lambda x: int(x.replace(":", "")))
        df.at[df[df["Time"] >= 150000].index, "Time"] = 145959
        df.at[df[df["Time"] < 92500].index, "Time"] = 92500
        df.at[df[(df["Time"] < 130000) & (df["Time"] >= 113000)].index, "Time"] = 112959

        df["Money"] = df["Volume"] * df["Price"]

        return True, df, "Success"

    def get_hfq_df(self,fnwp):
        try:
            df = pd.read_csv(fnwp, encoding="gb18030", header=0, names=self.title_hfq, dtype=self.dtype_hfq)
            df["date"] = df["date"].apply(lambda x: x.replace("-", ""))
        except ValueError as e:
            return False, "", "Read HFQ DF Error****{0} {1}".format(fnwp, str(e))
        return True, df, "Success"

    def get_index_df(self,fnwp):
        try:
            df = pd.read_csv(fnwp, encoding="gb18030", header=0, names=self.title_index, dtype=self.dtype_index)
            df["date"] = df["date"].apply(lambda date_s: date_s.replace("-", ""))
            df["code"]=df["code"].apply(lambda x: x.upper())
        except ValueError as e:
            return False, "", "Read Index DF Error****{0} {1}".format(fnwp, str(e))
        return True, df, "Success"
    ##appended log file
    def log_append_keep_old(self,loglist, logfnwp,log_titles, unique_check_title="Date"):
        if os.path.exists(logfnwp):
            dfo=pd.read_csv(logfnwp, header=0, names=log_titles)
            dfl=pd.DataFrame(loglist,columns=log_titles)
            dfl=dfl[dfl[unique_check_title].isin(dfo[unique_check_title]).apply(lambda x: not x)]

            dfo=dfo.append(dfl, ignore_index=True)
            dfo.sort_values(by=[unique_check_title],inplace=True)
        else:
            dfo=pd.DataFrame(loglist,columns=log_titles)
        dfo.to_csv(logfnwp,index=False)

    def log_append_keep_new(self,loglist, logfnwp,log_titles, unique_check_title="Date"):
        if os.path.exists(logfnwp):
            dfo=pd.read_csv(logfnwp, header=0, names=log_titles)
            dfl=pd.DataFrame(loglist,columns=log_titles)
            dfo=dfo[dfo[unique_check_title].isin(dfl[unique_check_title]).apply(lambda x: not x)]

            dfo=dfo.append(dfl, ignore_index=True)
            dfo.sort_values(by=[unique_check_title],inplace=True)
        else:
            dfo=pd.DataFrame(loglist,columns=log_titles)
        dfo.to_csv(logfnwp,index=False)