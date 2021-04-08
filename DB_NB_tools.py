

########################add Depressed flag
import os
def add_depressed_flag():
    pwd="/home/rdchujf/DB_raw/Normal"
    startMI=201801
    EndMI=202011

    dnIs=[int(dn) for dn in os.listdir(pwd) if dn.isdigit()]
    adIs=[dnI for dnI in dnIs if dnI>=startMI and dnI <=EndMI ]
    adIs.sort()
    for dnI in adIs:
        dnwp=os.path.join(pwd,str(dnI))
        fns=["{0}.Decompressed".format(fn.split(".")[0]) for fn in os.listdir(dnwp) if fn.endswith(".7z")]
        for fn in fns:
            flag_fnwp=os.path.join(dnwp,fn)
            with open(flag_fnwp, "w") as f:
                f.write("Success Decompressed")
            print ("marked {0}".format(flag_fnwp))


######################Remove doulbe row in HFQ for 20200731
import pandas as pd
import os
#注意这是要改DBI里的HFQcopy "/home/rdchujf/n_workspace/data/RL_data/I_DB/HFQ/"
#raw 是在 /mnt/data_disk/DB_raw/HFQ_Index/Stk_Day_FQ_WithHS/ 这个改了要 reset_DBI
#remove_double_row("/home/rdchujf/n_workspace/data/RL_data/I_DB/HFQ/", ["SZ000008","SZ000009","SZ000012","SZ000021","SZ000027"],20200731):
def remove_double_row(dnwp, stock_list,dateI):
    #new_dnwp="/home/rdchujf/n_workspace/data/RL_data/I_DB/HFQ/"
    #for stock in ["SZ000008","SZ000009","SZ000012","SZ000021","SZ000027"]:
    for stock in stock_list:
        print (stock)
        fnwp=os.path.join(dnwp,"{0}.csv".format(stock))
        dfn=pd.read_csv(fnwp,encoding="gb18030")
        a = dfn[dfn["date"] == dateI]  # 如果只改raw  时间的格式是  “时间”   “2020-07-31”
        print("Check whether doulbe row exists")
        print (a.index)
        if len((a.index))==2:
            dfn.drop([a.index[0]], inplace=True)
            a = dfn[dfn["date"] == dateI]
            print ("double row removed from dataframe")
            print (a.index)
            dfn.to_csv(fnwp, index=False,encoding="gb18030")
            b=pd.read_csv(fnwp,encoding="gb18030")
            a = b[b["date"] == dateI]
            print ("double row removed from DBI HFQ file")
            print (a.index)
        else:
            print (stock, "not double line")

####################################################################################################
#generate sub links for I_DB and TP DB
# DBI_V2  DBI_V3  NPrice1300_I5  NPrice930_I5 SV1M  SV5M
# DBTP_10MV1  DBTP_1MV1  DBTP_5MV1  TPHFD  TPV3  TPV4_5M  TPV5_10M_5MNprice
import subprocess

# def generate_synbolic_link(src, des):
#    result = subprocess.run(['ln', '-s', src, des], stdout=subprocess.PIPE)
#    return result.stdout

src_dir_base = "/home/rdchujf/RL_data_additional"
des_dir_base = "/home/rdchujf/n_workspace/data/RL_data"
for sub_dir in ["I_DB", "TP_DB"]:
    src_dir = os.path.join(src_dir_base, sub_dir)
    des_dir = os.path.join(des_dir_base, sub_dir)
    for dn in os.listdir(src_dir):
        # des_dir_dnwp=os.path.join(des_dir,dn)
        src_dir_dnwp = os.path.join(src_dir, dn)
        result = subprocess.run(['ln', '-s', src_dir_dnwp, des_dir], stdout=subprocess.PIPE)
'''
also need change
lrwxrwxrwx  1 rdchujf rdchujf        21 8月  20  2020 DB_raw -> /mnt/data_disk/DB_raw
lrwxrwxrwx  1 rdchujf rdchujf        32 3月  11 08:51 DB_raw_addon -> /mnt/pdata_disk2Tw/DB_raw_addon/
lrwxrwxrwx  1 rdchujf rdchujf        38 3月  11 08:37 RL_data_additional -> /mnt/pdata_disk2Tw/RL_data_additional/
'''


###############################################################################################
#add addon index day data manually
import os

from DBI_Base import Raw_HFQ_Index, DBI_init

IRHFQ = Raw_HFQ_Index("HFQ")
IRIdx = Raw_HFQ_Index("Index")
DateIs= [20210301, 20210302, 20210303, 20210304, 20210305, 20210308, 20210309, 20210310, 20210311, 20210312,
                 20210315, 20210316, 20210317, 20210318, 20210319]
for index_code in ["SH000001","SZ399001"]:
    for DayI in DateIs:
        hfq_flag, hfq_rawdf, hfq_mess = IRHFQ.get_addon_df(DayI)
        if not hfq_flag:
            Error_mess = "{0} {1} {2}".format(hfq_mess, DayI, hfq_mess)
            assert False
            # return False, Error_mess
        else:
            print(hfq_mess)

        i = DBI_init()
        fnwp = i.get_DBI_index_fnwp(index_code)
        print(fnwp)
        flag, DBIdf, mess = i.get_index_df(fnwp)
        if not flag:
            assert False, mess
        if DBIdf[DBIdf["date"] == str(DayI)].empty:
            df_found = hfq_rawdf[hfq_rawdf["code"] == index_code]
            assert len(df_found) == 1, "{0}raw HFQ_index file {1} record {2}".format(DayI, index_code, len(df_found))
            input = df_found.iloc[0].values
            if input.size == 0:
                print(input)
                assert False
                # return False, "No Data for {0} at {1} in addon HFQ_index file".format(index_code, DayI)
            DBIdf.loc[len(DBIdf)] = input[:-3]  # the last three column is HFQ related inform
            DBIdf.sort_values(by="date", inplace=True)
            DBIdf.to_csv(fnwp, index=False)
            print(["Addon indexes Update", index_code, True, "Success"])
        else:
            print(["Addon indexes Update", index_code, True, "Already updated"])

