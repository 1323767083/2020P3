from DBR_base import DBR_base
import os,random
import subprocess
import pandas as pd
class RawData(DBR_base):
    def __init__(self,Raw_lumpsum_End_DayI):
        DBR_base.__init__(self,Raw_lumpsum_End_DayI)

    #********************Internal API**************************
    def _get_qz_df(self, fnwp):
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
                return False, "", "read {0} to df get error {1}".format(fnwp,str(e))
                #raise ValueError(str(e))
        # df = pd.read_csv(fnwp, header=0, names=self.title_qz, dtype=self.dtype_qz)
        df.drop(["TranID"], axis=1, inplace=True)
        if df.empty:
            return False,"", "read {0} get empty df".format(fnwp)
            #raise ValueError("{0} is empty".format(fnwp))

        df["Time"] = df["Time"].apply(lambda x: int(x.replace(":", "")))
        df.at[df[df["Time"] >= 150000].index, "Time"] = 145959
        df.at[df[df["Time"] < 92500].index, "Time"] = 92500
        df.at[df[(df["Time"] < 130000) & (df["Time"] >= 113000)].index, "Time"] = 112959

        df["Money"] = df["Volume"] * df["Price"]

        return True, df, "Success"

    # ********************Normal Data API**************************
    def get_normal_addon_raw_fnwp(self, dayI, stock,flag): #flag=True mens normal / flagg=False means addon
        if dayI >= 20180716 and dayI <= 20180719:
            decompressed_fn = "{0}{1}.csv".format(stock[0:2].upper(), stock[2:])
        else:
            decompressed_fn = "{0}.csv".format(stock[2:])
        if flag:
            compressed_fnwp = os.path.join(self.Dir_raw_normal, str(dayI//100), "{0:8d}.7z".format(dayI))
        else:
            compressed_fnwp = os.path.join(self.Dir_raw_normal_addon, str(dayI // 100), "{0:8d}.7z".format(dayI))
        return compressed_fnwp, decompressed_fn

    def get_normal_addon_qz_df(self,dayI, stock,flag): #flag=True mens normal / flagg=False means addon
        compressed_fnwp, decompressed_fn=self.get_normal_addon_raw_fnwp(dayI,stock,flag)
        #res = subprocess.call(["7z","e", "/home/rdchujf/n_workspace/tmp/20180102.7z", "-o/home/rdchujf/n_workspace/tmp","2018-01-02/000001.csv","-r","-y"]
        if not os.path.exists(compressed_fnwp):
            return False, "", "Failed found compress file {0}".format(compressed_fnwp)
        lcmd=["7z","e", compressed_fnwp,
              "{0}-{1:02}-{2:02}/{3}".format(dayI//10000, dayI%10000//100,dayI%100,decompressed_fn),
              "-o%s" % self.Dir_Tmp,
              "-r","-y","-bd"]
        with open(os.devnull, 'w')  as FNULL: #hide terminal output
            res = subprocess.call(lcmd,stdout=FNULL)
        if res==0: # success
            out_fnwp=os.path.join(self.Dir_Tmp,decompressed_fn)
            if os.path.exists(out_fnwp):
                flag,df, message=self._get_qz_df(out_fnwp)
                if flag:
                    os.remove(out_fnwp)
                return flag,df, message
            else:
                return False,"","Failed found output file {0}".format(out_fnwp)
        else: # fail log
            return False, "", "Failed decompress {0} {1} raw qz data".format(dayI,stock)

    # ********************Legacy Data API**************************
    def get_legacy_fnwp(self,dayI, stock):
        year=dayI//10000
        date_with_dash = "{0}-{1:02d}-{2:02d}".format(dayI//10000, dayI%10000//100, dayI%100)
        if year in [2013, 2014]:
            return os.path.join(self.Dir_raw_legacy_1,str(year),"{0}{1}".format(dayI//100,stock[0:2].upper()),date_with_dash,"{0}.csv".format(stock[2:]))
        elif year ==2016:
            return os.path.join(self.Dir_raw_legacy_1, str(year), date_with_dash,"{0}.csv".format(stock[2:]))
        elif year ==2015:
            return os.path.join(self.Dir_raw_legacy_2, str(year), date_with_dash,"{0}.csv".format(stock[2:]))
        elif year ==2017:
            return os.path.join(self.Dir_raw_legacy_2, str(year), date_with_dash,"{0}.csv".format(stock[2:]))
        else:
            raise ValueError( "Does not has data for {0} {1}".format(dayI,stock))


    def get_legacy_qz_df(self,dayI, stock):
        fnwp=self.get_legacy_fnwp(dayI, stock)
        if os.path.exists(fnwp):
            flag, df, message = self._get_qz_df(fnwp)
            if flag and dayI<20170801:
                sampled_index = random.sample(list(df.index), int(len(df) * 0.995))
                df = df.loc[sampled_index]
                df.reset_index(inplace=True, drop=True)
            return flag, df, message
        else:
            return False, "", "Failed found file {0}".format(fnwp)

    # ********************HFQ Data API**************************
    def get_lumpsum_HFQ_fnwp(self, stock):
        assert len(stock) == 8
        fnwp = os.path.join(self.Dir_raw_HFQ_base, stock.upper() + ".csv")
        return fnwp

    def get_HFQ_lumpsum_df(self, stock):
        assert len(stock) == 8
        fnwp = self.get_lumpsum_HFQ_fnwp(stock)
        if not os.path.exists(fnwp):
            return False, "" ,"{0} does not exists".format(fnwp)
        else:
            df = pd.read_csv(fnwp, encoding="gb18030", header=0, names=self.title_hfq, dtype=self.dtype_hfq)
            df["date"] = df["date"].apply(lambda x: x.replace("-", ""))
            if len(df[df["date"]>=str(self.Raw_Normal_Lumpsum_EndDayI)])<1:
                return False,"", "{0} does ot have HFQ data at {1} or later".format(stock,self.Raw_Normal_Lumpsum_EndDayI)
            else:
                return True, df, "Success"

    def get_HFQ_addon_fnwp(self, dayI):
        #Dir_raw_HFQ_base_addon = "/home/rdchujf/Stk_qz_3_support/Stk_Day_FQ_WithHS_addon"
        # "/home/rdchujf/Stk_qz_3_support/Stk_Day_FQ_WithHS_addon/YYYYMM/YYYYMMDD.rar"
        compressed_fnwp = os.path.join(self.Dir_raw_HFQ_base_addon, str(dayI//100),str(dayI) + ".rar")
        decompressed_fn=str(dayI) + ".csv"
        return compressed_fnwp,decompressed_fn

    def get_HFQ_addon_df(self, dayI):

        compressed_fnwp,decompressed_fn=self.get_HFQ_addon_fnwp(dayI)
        if not os.path.exists(compressed_fnwp):
            return False,"","Fail found compress HFQ addon file {0}".format(compressed_fnwp)
        #" rar e 20200624.rar ProcessFile/Stk_Day/Stk_Day_FQ_WithHS_Daily/20200624.csv /home/rdchujf/tmp -y -inul"
        lcmd = ["rar", "e", compressed_fnwp,
                "ProcessFile/Stk_Day/Stk_Day_FQ_WithHS_Daily/{0}".format(decompressed_fn),
                self.Dir_Tmp,
                "-y","-inul"]
        res = subprocess.call(lcmd)
        if res ==0:
            out_fnwp=os.path.join(self.Dir_Tmp,decompressed_fn)
            if os.path.exists(out_fnwp):
                df = pd.read_csv(out_fnwp, encoding="gb18030", header=0, names=self.title_hfq, dtype=self.dtype_hfq)
                df["date"] = df["date"].apply(lambda x: x.replace("-", ""))
                return True, df, "Success"
            else:
                return False, "", "Fail to find decompress HFQ addon file {0}".format(out_fnwp)
        else:
            return False, "", "Failed decompress {0} HFQ addon data".format(dayI)

    # ********************Index Data API**************************
    def get_index_lumpsum_fnwp(self, index):
        assert len(index) == 8
        fnwp=os.path.join(self.Dir_raw_Index_base,index.lower()+".csv")
        return fnwp

    def get_index_lumpsum_df(self, index):
        assert len(index)==8
        fnwp=self.get_index_lumpsum_fnwp(index)
        if not os.path.exists(fnwp):
            return False, "" ,"{0} does not exists".format(fnwp)
        else:
            df=pd.read_csv(fnwp,encoding="gb18030", header=0, names=self.title_index,dtype=self.dtype_index)
            df["date"] = df["date"].apply(lambda date_s: date_s.replace("-", ""))
            if len(df[df["date"]>=str(self.Raw_Normal_Lumpsum_EndDayI)])<1:
                return False, "Index {0} does ot have data at {1} or later".format(index,self.Raw_Normal_Lumpsum_EndDayI)
            else:
                return True, df, "Success"

    def get_index_addon_fnwp(self, dayI):

        #Dir_raw_Index_base_addon = "/home/rdchujf/Stk_qz_3_support/Stk_Day_Idx_addon"
        # "/home/rdchujf/Stk_qz_3_support/Stk_Day_Idx_addon/YYYYMM/YYYYMMDD.rar
        compressed_fnwp = os.path.join(self.Dir_raw_Index_base_addon, str(dayI // 100), str(dayI) + ".rar")
        decompressed_fn = str(dayI) + ".csv"
        return compressed_fnwp, decompressed_fn

    def get_index_addon_df(self, dayI):
        compressed_fnwp,decompressed_fn=self.get_index_addon_fnwp(dayI)
        if not os.path.exists(compressed_fnwp):
            return False,"","Fail found compress index add on file {0}".format(compressed_fnwp)
        #" rar e 20200624.rar ProcessFile/Stk_Day/Stk_Day_FQ_WithHS_Daily/20200624.csv /home/rdchujf/tmp -y -inul"
        lcmd = ["rar", "e", compressed_fnwp,
                "ProcessFile/Stk_Day/Stk_Day_Idx_Daily/{0}".format(decompressed_fn),
                self.Dir_Tmp,
                "-y","-inul"]
        res = subprocess.call(lcmd)
        if res ==0:
            out_fnwp=os.path.join(self.Dir_Tmp,decompressed_fn)
            if os.path.exists(out_fnwp):
                df = pd.read_csv(out_fnwp, encoding="gb18030", header=0, names=self.title_index, dtype=self.dtype_index)
                df["date"] = df["date"].apply(lambda date_s: date_s.replace("-", ""))
                return True, df, "Success"
            else:
                return False, "", "Fail to find decompress index addon file {0}".format(out_fnwp)
        else:
            return False, "", "Failed decompress {0} index addon data".format(dayI)

    # ********************Interface API**************************
    def get_qz_df(self, dayI, stock):
        if dayI>=self.Raw_legacy_Lumpsum_StartDayI and dayI<=self.Raw_Legacy_Lumpsum_EndDayI:
            flag,df,message=self.get_legacy_qz_df(dayI, stock)
        elif dayI<=self.Raw_Normal_Lumpsum_EndDayI:
            flag,df,message=self.get_normal_addon_qz_df(dayI, stock,True)
        else:
            flag, df, message = self.get_normal_addon_qz_df(dayI, stock,False)
        return flag, df, message


    def get_hfq_df(self, dayI, stock):
        #if is lumpsum get one stock all day df
        #else is addon get all stock one day df
        if dayI<=self.Raw_Normal_Lumpsum_EndDayI:
            flag, df, message=self.get_HFQ_lumpsum_df(stock)
        else:
            flag, df, message=self.get_HFQ_addon_df(dayI)
        return flag, df, message

    def get_index_df(self, dayI, index):
        #if is lumpsum get one index all day df
        #else is addon get all index one day df
        if dayI<=self.Raw_Normal_Lumpsum_EndDayI:
            flag, df, message=self.get_index_lumpsum_df(index)
        else:
            flag, df, message=self.get_index_addon_df(dayI)
        return flag, df, message


    def test(self):
        #import DBR_Reader as i
        #r = i.RawData(20200529)

        flag, df, message=self.get_qz_df(20150203,"SH600000")

        flag, df, message = self.get_qz_df(20180205, "SH600000")

        #flag, df, message = self.get_qz_df(20200624, "SH600001") not have addon

        flag, df, message=self.get_hfq_df(20150203,"SH600000")

        flag, df, message = self.get_hfq_df(20180203, "SH600000")

        flag, df, message = self.get_hfq_df(20200624, "SH600000")

        flag, df, message = self.get_index_df(20150203, "SH000001")

        flag, df, message = self.get_index_df(20180203, "SH000001")

        flag, df, message = self.get_index_df(20200624, "SH000001")