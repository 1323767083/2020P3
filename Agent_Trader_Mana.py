import os,json,re, pickle
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DBI_Base import DBI_init_with_TD
AT_dir="/mnt/data_disk2/n_workspace/AT"
def List_Strategies_Config(portfolio):
    dnwp=os.path.join(AT_dir,portfolio)
    dirs=[item for item in os.listdir(dnwp) if os.path.isdir(os.path.join(dnwp,item))]
    dfr=[]
    for dir in dirs:
        df=pd.read_csv(os.path.join(dnwp, dir,"strategy_config.csv"))
        df["Strategy"]=dir
        if len(dfr)==0:
            dfr=df
        else:
            dfr = pd.concat([dfr, df], axis=0, ignore_index=True)
    dfr.sort_values(["Strategy","RL_system_name"],inplace=True)
    dfr.reset_index(inplace=True, drop=True)
    dfr=dfr[["Strategy","RL_system_name","RL_Model_ET","GPU_idx","GPU_mem","TPDB_Name","SL_Name","SL_Tag","SL_Idx","strategy_fun"]]
    return dfr

def ana_earning(portfolio,strategy,experiment):
    account_detail_fnwp=os.path.join(AT_dir,portfolio,strategy,experiment,"AT_AccountDetail.csv")
    dfc = pd.read_csv(os.path.join(AT_dir, portfolio,strategy, "strategy_config.csv"))

    df=pd.read_csv(account_detail_fnwp)
    #df["Total"]=df["Cash_after_closing"]+df["MarketValue_after_closing"]
    df["Total"] = df["Cash_after_closing"]
    for i in list(range(len(dfc))):
        df["Total"]+=df["MarketValue_after_closing_M{0}".format(i)]
    df["MonthI"]=df["DateI"]//100
    dfm=df[["MonthI","Total"]].groupby(["MonthI"]).agg(monthly_start=pd.NamedAgg(column="Total", aggfunc="first"),
                                                 monthly_end=pd.NamedAgg(column="Total", aggfunc="last"))

    dfm["Month_earn"]=(dfm["monthly_end"]-dfm["monthly_start"])/dfm["monthly_start"]
    dfm.reset_index(inplace=True)

    plt.rcParams["figure.figsize"] = (20, 10)
    fig = plt.figure()
    fig.add_subplot(211)
    fig.add_subplot(212)
    allaxes = fig.get_axes()

    allaxes[0].set_title("Total, Cash and Market Value")
    allaxes[0].plot(df["Cash_after_closing"],label="Cash_after_closing",color='m')
    for i in list(range(len(dfc))):
        title="MarketValue_after_closing_M{0}".format(i)
        allaxes[0].plot(df[title],label=title)

    allaxes[0].legend(loc='upper left')
    ax2=allaxes[0].twinx()
    ax2.plot(df["Total"],label="Total",color='r')
    ax2.legend(loc='upper right')

    allaxes[1].set_title("Monthly Earning")
    allaxes[1].plot(dfm["Month_earn"],label="Month_earn")
    allaxes[1].plot([0,len(dfm)],[0,0])
    allaxes[1].set_xticks(list(range(len(dfm))))
    allaxes[1].set_xticklabels([str(MonthI) for MonthI in dfm["MonthI"].tolist()],rotation=90)
    allaxes[1].legend(loc='upper left')
    return df, dfm


def get_data_from_report(fnwp):
    one_time_patterns=[r'Today Sold with Earn (\d+)',
             r'Today Sold with Loss (\d+)',
             r'Today Sold with Balance (\d+)',
             r'Tommorow to Buy (\d+)',
             r'Tommorow to Sell (\d+)',
             r'Tommorow not_buy_due_limit (\d+)',
             r'Tommorow multibuy\(not sell due to multibuy\) (\d+)']
    multi_time_patterns= [r"Fail to buy due to ",r"Fail to sell due to "]

    with open(fnwp,"r") as f:
        flag_found=[False for _ in one_time_patterns]
        results=[np.NaN for _ in one_time_patterns]
        multi_time_counts = [0 for _ in multi_time_patterns]
        for line in f.readlines():
            for idx in list(range(len(one_time_patterns))):
                if flag_found[idx]:
                    continue
                else:
                    a=re.findall(one_time_patterns[idx],line)
                    if len(a)!=0:
                        assert results[idx]!=results[idx]
                        results[idx]=int(a[0])
                        flag_found[idx]=True
            for idx in list(range(len(multi_time_patterns))):
                a = re.findall(multi_time_patterns[idx], line)
                if len(a) != 0:
                    multi_time_counts[idx]+=1

        assert all(flag_found),flag_found
    return results+multi_time_counts
#get_data_from_report("/mnt/data_disk2/n_workspace/AT/p1/s17/eF1/201801/20180102/Report.txt")

def get_datas_for_experiment(portfolio, strategy, experiment):
    ednwp=os.path.join(AT_dir,portfolio, strategy, experiment)
    monthIs=[int(item) for item in os.listdir(ednwp) if os.path.isdir(os.path.join(ednwp,item))]
    monthIs.sort()
    DateIs=[]
    for monthI in monthIs:
        emdnwp=os.path.join(ednwp,str(monthI))
        mDateIs=[int(item) for item in os.listdir(emdnwp) if os.path.isdir(os.path.join(emdnwp,item))]
        mDateIs.sort()
        DateIs.extend(mDateIs)
    Datas=[]
    for DateI in DateIs:
        fnwp=os.path.join(ednwp,str(DateI//100), str(DateI),"Report.txt")
        Datas.append(get_data_from_report(fnwp)+[DateI])
    df=pd.DataFrame(Datas, columns=["SoldE","SoldL","SoldB","ToBuy","ToSell","NotToBuy_due_limit","Multibuy","Fail_Buy", "Fail_Sell","DateI"])

    df["MonthI"] = df["DateI"] // 100
    dfm = df[["MonthI", "SoldE", "SoldL", "SoldB", "ToBuy","ToSell","NotToBuy_due_limit","Multibuy","Fail_Buy", "Fail_Sell"]].groupby(["MonthI"]).agg(
        CEarn=pd.NamedAgg(column="SoldE", aggfunc=sum),
        CLoss=pd.NamedAgg(column="SoldL", aggfunc=sum),
        CBalance=pd.NamedAgg(column="SoldB", aggfunc=sum),
        CToBuy=pd.NamedAgg(column="ToBuy", aggfunc=sum),
        CToSell=pd.NamedAgg(column="ToSell", aggfunc=sum),
        CNotToBuy_due_limit=pd.NamedAgg(column="NotToBuy_due_limit", aggfunc=sum),
        CMultibuy=pd.NamedAgg(column="Multibuy", aggfunc=sum),
        CFailBuy=pd.NamedAgg(column="Fail_Buy", aggfunc=sum),
        CFailSell=pd.NamedAgg(column="Fail_Sell", aggfunc=sum)
    )

    dfm["CTotal"] = dfm.CEarn + dfm.CLoss + dfm.CBalance

    dfm["REarn"] = dfm.CEarn / dfm.CTotal
    dfm["RLoss"] = dfm.CLoss / dfm.CTotal
    dfm["RBalance"] = dfm.CBalance / dfm.CTotal
    pd.options.display.float_format = "{:,.2f}".format
    dfm.fillna(0, inplace=True)
    return df,dfm





def get_per_tran_result(portfolio, strategy, experiment):
    dfc = pd.read_csv(os.path.join(AT_dir, portfolio, strategy, "strategy_config.csv"))

    dnwp = os.path.join(AT_dir, portfolio, strategy, experiment)
    monthSs = [item for item in os.listdir(dnwp) if os.path.isdir(os.path.join(dnwp, item))]
    monthSs.sort()
    trans = []
    for monthS in monthSs:
        wdnwp = os.path.join(dnwp, monthS)
        DateSs = [item for item in os.listdir(wdnwp) if os.path.isdir(os.path.join(wdnwp, item))]
        DateSs.sort()
        for DateS in DateSs:
            pfnwp = os.path.join(wdnwp, DateS, "Report.pikle")
            datas = pickle.load(open(pfnwp, "rb"))
            Cash_afterclosing, l_MarketValue_afterclosing, mumber_of_stock_could_buy,\
            ll_log_bought, ll_log_Earnsold, ll_log_balancesold, ll_log_Losssold,\
            ll_log_fail_action, ll_log_holding_with_no_action,\
            ll_ADlog, ll_a, adj_ll_a = datas
            for Model_idx in list(range(len(dfc))):
                for logs in [ll_log_Earnsold[Model_idx], ll_log_Losssold[Model_idx]]:
                    if len(logs) != 0:
                        for log in logs:
                            a = re.findall(r'(\w+):([-+]?[0-9]+[.][0-9]*)', log)
                            if len(a) != 0:
                                trans.append([int(DateS), Model_idx,a[0][0], eval(a[0][1])])
    df = pd.DataFrame(trans, columns=["dateI","model", "Stock", "profit"])
    return df


def get_tran_detail(portfolio, strategy, experiment, threadhold):
    df = get_per_tran_result(portfolio, strategy, experiment)
    i = DBI_init_with_TD()
    nptd = i.nptd
    results = []
    if threadhold<0:
        dfr=df[df["profit"] < threadhold]
    else:
        dfr = df[df["profit"] >= threadhold]
    for _, row in dfr.iterrows():
        dateI, ModelI, stock, profit = row[0], row[1], row[2], row[3]
        B1dateI = nptd[nptd < dateI][-1]
        account_backup_fnwp = os.path.join(AT_dir, portfolio, strategy, experiment,
                                           str(B1dateI // 100), str(B1dateI), "AT{0}_account_afterday_backup.csv".format(ModelI))
        df = pd.read_csv(account_backup_fnwp)
        df.set_index(["Stock"], drop=True, inplace=True)
        buy_dateI = df.loc[stock]["HoldingStartDateI"]
        result = [buy_dateI, dateI, ModelI,len(nptd[(nptd >= buy_dateI) & (nptd < dateI)]), profit]
        print(result)
        results.append(result)
    return df, results


'''
from DBI_Base import DBI_init_with_TD
import pandas as pd
monthIs=[202101,202102,202103]
shidx_fnwp="/mnt/pdata_disk2Tw/RL_data_additional/index/SH000001.csv"
szidx_fnwp="/mnt/pdata_disk2Tw/RL_data_additional/index/SZ399001.csv"

i=DBI_init_with_TD()
i.nptd
MonthI=202101
_,SDateI=i.get_closest_TD(MonthI*100+1, True)
_,EDateI=i.get_closest_TD(MonthI*100+31, False)
dfi=pd.read_csv(shidx_fnwp)#, encoding="gb18030")
dfim=dfi[(dfi["date"]>=SDateI)&(dfi["date"]<=EDateI)]
dfim.reset_index(drop=True, inplace=True)
print(dfim.iloc[0]["open_price"],dfim.iloc[-1]["open_price"])
print(dfim["open_price"].max(), dfim["open_price"].min())

Sidx=dfim.iloc[0]["open_price"]

[dfim.iloc[-1]["open_price"]/Sidx-1,dfim["open_price"].max()/Sidx-1, dfim["open_price"].min()/Sidx-1]
'''