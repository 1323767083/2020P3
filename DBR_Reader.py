from DB_Base import DB_Base
import os,random,sys
import subprocess
from datetime import datetime

class Decompressed_flag:
    def get_decompressed_flag_fnwp(self, compressed_fnwp):
        dn,fn=os.path.split(compressed_fnwp)
        return os.path.join(dn,"{0}.Decompressed".format(fn.split(".")[0]))

    def check_decompressed(self, compressed_fnwp):
        return os.path.exists(self.get_decompressed_flag_fnwp(compressed_fnwp))

    def set_Normal_decompressed(self, compressed_fnwp):
        with open(self.get_decompressed_flag_fnwp(compressed_fnwp),"w") as f:
            f.write("Success Decompressed")

    def set_HFQ_Index_decompressed(self, compressed_fnwp):
        with open(self.get_decompressed_flag_fnwp(compressed_fnwp),"w") as f:
            f.write("Success Decompressed")

class Raw_HFQ_Index(DB_Base,Decompressed_flag):
    def __init__(self,swith_HFQ__index):
        DB_Base.__init__(self)
        self.swith_HFQ__index=swith_HFQ__index
        assert self.swith_HFQ__index in ["HFQ","Index"]
        if self.swith_HFQ__index=="HFQ":
            self.Dir_base=self.Dir_raw_HFQ_base
            self.FUN_get_df=self.get_hfq_df
            self.Dir_addon_base=self.Dir_raw_HFQ_base_addon
            self.Dir_addon_decompress_base=self.Dir_raw_HFQ_base_addon_decompressed
        else:
            self.Dir_base=self.Dir_raw_Index_base
            self.FUN_get_df=self.get_index_df
            self.Dir_addon_base=self.Dir_raw_Index_base_addon
            self.Dir_addon_decompress_base=self.Dir_raw_Index_base_addon_decompressed


    # ********************HFQ Data API**************************
    def _get_lumpsum_fnwp(self, code):
        assert len(code) == 8
        fnwp = os.path.join(self.Dir_base,"{0}.csv".format(code.upper() if self.swith_HFQ__index=="HFQ" else code.lower()))
        return fnwp

    def get_lumpsum_df(self, code):
        assert len(code) == 8
        fnwp = self._get_lumpsum_fnwp(code)
        if not os.path.exists(fnwp):
            return False, "" ,"{0} Lumpsum DF File Not Found****,{1},{2} does not exists".format(self.swith_HFQ__index,code,fnwp)
        else:
            flag,df, mess=self.FUN_get_df(fnwp)
            if not flag:
                return False, "", "{0} {1} {2}".format(mess,self.swith_HFQ__index,code)
            if len(df[df["date"]>=str(self.Raw_Normal_Lumpsum_EndDayI)])<1:
                return False,"", "{0} Not Have Lumpsum End****{1} not have {2} data".format(self.swith_HFQ__index,code,self.Raw_Normal_Lumpsum_EndDayI)
            else:
                return True, df, "Success"

    def get_addon_fnwp(self, dayI):
        compressed_fnwp = os.path.join(self.Dir_addon_base, str(dayI//100),str(dayI) + ".rar")
        decompressed_fnwp=os.path.join(self.Dir_addon_decompress_base, str(dayI//100),str(dayI) + ".csv")
        return compressed_fnwp,decompressed_fnwp

    def decompress_normal_addon(self, compressed_fnwp, decompressed_dn):
        if not os.path.exists(compressed_fnwp):
            return False,"Compress File Not Found****{0}".format(compressed_fnwp)

        print ("\tStart Decompress at {0}  {1}".format(datetime.now().time(),compressed_fnwp))
        if not os.path.exists(decompressed_dn): os.makedirs(decompressed_dn)
        lcmd = ["rar", "e", compressed_fnwp, decompressed_dn,"-y","-inul"]
        with open(os.devnull, 'w')  as FNULL: #hide terminal output
            res = subprocess.call(lcmd,stdout=FNULL)
        if res==0: # success
            return True, "Success"
        else: # fail log
            return False,  "Decomprress File Error**** {0}".format(compressed_fnwp)

    def get_addon_df(self, dayI):
        compressed_fnwp,decompressed_fnwp=self.get_addon_fnwp(dayI)
        if os.path.exists(decompressed_fnwp):
            flag, df, mess = self.FUN_get_df(decompressed_fnwp)
            if not flag:
                return False, "", mess
            return True, df, "Success"
        flag, mess= self.decompress_normal_addon(compressed_fnwp, os.path.dirname(decompressed_fnwp))
        if not flag:
            return False, "", mess
        if os.path.exists(decompressed_fnwp):
            flag, df, mess = self.FUN_get_df(decompressed_fnwp)
            if not flag:
                return False, "", mess
            return True, df, "Success"
        else:
            return False, "", "Decompress File Not Found****{0} {1}".format(self.swith_HFQ__index,decompressed_fnwp)

class RawData(DB_Base,Decompressed_flag):
    def __init__(self):
        DB_Base.__init__(self)

    # ********************Normal Data API**************************
    def get_normal_addon_raw_fnwp(self, dayI, stock,flag): #flag=True mens normal / flagg=False means addon
        if flag:
            decompress_dn=os.path.join(self.Dir_raw_normal_decompressed, str(dayI//100),str(dayI))
            compress_dn=os.path.join(self.Dir_raw_normal, str(dayI//100))
        else:
            decompress_dn=os.path.join(self.Dir_raw_normal_addon_decompressed, str(dayI // 100),str(dayI))
            compress_dn=os.path.join(self.Dir_raw_normal_addon, str(dayI // 100))

        if dayI >= 20180716 and dayI <= 20180719:

            decompressed_fnwp =os.path.join(decompress_dn,"{0}{1}.csv".format(stock[0:2].upper(), stock[2:]))
        else:
            decompressed_fnwp = os.path.join(decompress_dn,"{0}.csv".format(stock[2:]))

        if dayI >= 20180701 and dayI <= 20181231:
            compressed_fnwp = os.path.join(compress_dn, "{0}-{1:02d}-{2:02d}.7z".format(dayI//10000, dayI//100%100, dayI%100))
        else:
            compressed_fnwp = os.path.join(compress_dn, "{0:8d}.7z".format(dayI))
        return compressed_fnwp, decompressed_fnwp


    def decompress_normal_addon_qz(self,dayI, compressed_fnwp,decompressed_dn):
        if self.check_decompressed(compressed_fnwp):
            return True, "Already Decompressed"
        if not os.path.exists(compressed_fnwp):
            return False, "Compress File Not Found**** Failed found qz compress file {0}".format(compressed_fnwp)
        lcmd=["7z","e", compressed_fnwp, "-o%s" %   decompressed_dn,"-r","-y","-bd"]
        print ("\tStart Decompress at {0}  {1}".format(datetime.now().time(),compressed_fnwp))
        with open(os.devnull, 'w')  as FNULL: #hide terminal output
            res = subprocess.call(lcmd,stdout=FNULL)
        if res==0: # success
            #clean an empty directory cause by extract e not x
            dir_to_remove=os.path.join(decompressed_dn,
                         "{0:4d}-{1:02d}-{2:02d}".format(dayI // 10000, dayI % 10000 // 100, dayI % 100))
            if os.path.exists(dir_to_remove):   os.rmdir(dir_to_remove)
            self.set_Normal_decompressed(compressed_fnwp)
            return True, "Success"
        else: # fail log
            return False,  "Decomprress File Error**** {0}".format(compressed_fnwp)

    def get_normal_addon_qz_df(self,dayI, stock,flag): #flag=True mens normal / flagg=False means addon
        compressed_fnwp, decompressed_fnwp=self.get_normal_addon_raw_fnwp(dayI,stock,flag)
        if os.path.exists(decompressed_fnwp):
            flag, df, message = self.get_qz_df(decompressed_fnwp)
            return flag, df, message
        decompress_flag, decompress_mess=self.decompress_normal_addon_qz(dayI, compressed_fnwp, os.path.dirname(decompressed_fnwp))
        if decompress_flag:
            if os.path.exists(decompressed_fnwp):
                flag,df, message=self.get_qz_df(decompressed_fnwp)
                return flag,df, message
            else:
                return False,"","Decompress File Not Found****{0}".format(decompressed_fnwp)
        else:
            return False, "", decompress_mess

    # ********************Legacy Data API**************************
    def _get_legacy_fnwp(self,dayI, stock):
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
        fnwp=self._get_legacy_fnwp(dayI, stock)
        if os.path.exists(fnwp):
            flag, df, message = self.get_qz_df(fnwp)
            if flag and dayI<20170801:
                sampled_index = random.sample(list(df.index), int(len(df) * 0.995))
                df = df.loc[sampled_index]
                df.reset_index(inplace=True, drop=True)
            return flag, df, message
        else:
            return False, "", "File Not Found**** {0}".format(fnwp)

    # ********************Interface API**************************

    def get_qz_df_inteface(self, stock,dayI):
        if dayI>=self.Raw_legacy_Lumpsum_StartDayI and dayI<=self.Raw_Legacy_Lumpsum_EndDayI:
            flag,df,message=self.get_legacy_qz_df(dayI, stock)
        elif dayI<=self.Raw_Normal_Lumpsum_EndDayI:
            flag,df,message=self.get_normal_addon_qz_df(dayI, stock,True)
        elif dayI>self.Raw_Normal_Lumpsum_EndDayI:
            flag, df, message = self.get_normal_addon_qz_df(dayI, stock, False)
        else:
            flag, df, message=False, "", "DayI {0} should >= {1}".format(dayI,self.Raw_legacy_Lumpsum_StartDayI)

        return flag, df, message
