import os
class DBR_base:
    #general param
    SDir_IDB="I_DB"
    SDir_TPDB="TP_DB"
    Dir_Tmp="/home/rdchujf/tmp"
    Dir_DB_Base = "/home/rdchujf/n_workspace/data/RL_data"
    FN_Raw_Source = "Raw_Source.json"

    #raw lumpsum addon data param

    Dir_raw_legacy_1        =   "/home/rdchujf/Stk_qz"
    Dir_raw_legacy_2        =   "/home/rdchujf/Stk_qz_2"
    Dir_raw_normal          = "/media/rdchujf/2G"
    Dir_raw_normal_addon    ="/media/rdchujf/2G"


    Dir_raw_HFQ_base        =   "/home/rdchujf/Stk_qz_3_support/Stk_Day_FQ_WithHS"
    Dir_raw_Index_base      =   "/home/rdchujf/Stk_qz_3_support/Stk_Day_Idx"
    Dir_raw_HFQ_base_addon="/home/rdchujf/Stk_qz_3_support/Stk_Day_FQ_WithHS_addon"
    #"/home/rdchujf/Stk_qz_3_support/Stk_Day_FQ_WithHS_addon/YYYYMM/YYYYMMDD.rar"
    Dir_raw_Index_base_addon = "/home/rdchujf/Stk_qz_3_support/Stk_Day_Idx_addon"
    #"/home/rdchujf/Stk_qz_3_support/Stk_Day_Idx_addon/YYYYMM/YYYYMMDD.rar

    #df structure
    title_qz = ["TranID", "Time", "Price", "Volume", "SaleOrderVolume", "BuyOrderVolume", "Type",
                     "SaleOrderID",
                     "SaleOrderPrice", "BuyOrderID", "BuyOrderPrice"]
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

    def __init__(self,Raw_lumpsum_End_DayI=20200529):
        self.Raw_legacy_Lumpsum_StartDayI = 20130415
        self.Raw_Legacy_Lumpsum_EndDayI = 20171229
        assert Raw_lumpsum_End_DayI>self.Raw_Legacy_Lumpsum_EndDayI
        self.Raw_Normal_Lumpsum_EndDayI = Raw_lumpsum_End_DayI#20200529

        self.Dir_IDB=os.path.join(self.Dir_DB_Base,self.SDir_IDB)
        self.Dir_TPDB=os.path.join(self.Dir_DB_Base,self.SDir_TPDB)

        for dir in [self.Dir_DB_Base,self.Dir_IDB,self.Dir_TPDB,self.Dir_Tmp,self.Dir_raw_HFQ_base_addon,self.Dir_raw_Index_base_addon]:
            if not os.path.exists(dir):os.makedirs(dir)