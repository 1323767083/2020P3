

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


from DBI_Base import StockList,DBI_init
import pandas as pd
import os
#get_exceed_max_price_sl("SLV500_10M", 500)

def get_exceed_max_price_sl(sl_name, max_price):
    i=StockList(sl_name)
    flag,sl=i.get_sub_sl("Train",0)
    assert flag
    j=DBI_init()
    threadhold=max_price
    esl=[]
    for stock in sl:
        flag,df, mess=j.get_hfq_df(j.get_DBI_hfq_fnwp(stock))
        assert flag
        max_price=df["open_price"].max()
        if max_price>=threadhold:
            print ("{0} should exclude for max {1}".format(stock,max_price))
            esl.append([stock,"price_too_high"])
        else:
            print (stock, " ok")

    df=pd.DataFrame(esl, columns=["Stock","Reason"])
    fnwp=os.path.join("/home/rdchujf/n_workspace/data/RL_data/I_DB/Stock_List",sl_name,"Price_to_Remove.csv")
    df.to_csv(fnwp,index=False)

