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
    dfr=dfr[["Strategy","RL_system_name","RL_Model_ET","GPU_idx","GPU_mem","TPDB_Name","SL_Name"]]
    return dfr


def get_months_statics_Index(index_code, MonthIs):
    lr = []
    for MonthI in MonthIs:
        i = DBI_init_with_TD()
        _, SDateI = i.get_closest_TD(MonthI * 100 + 1, True)
        _, EDateI = i.get_closest_TD(MonthI * 100 + 31, False)
        fnwp = i.get_DBI_index_fnwp(index_code)
        dfi = pd.read_csv(fnwp)
        dfim = dfi[(dfi["date"] >= SDateI) & (dfi["date"] <= EDateI)]
        ops = dfim["open_price"].values
        op25 = np.percentile(ops, 25, axis=0)
        op50 = np.percentile(ops, 50, axis=0)
        op75 = np.percentile(ops, 75, axis=0)
        MStart = ops[0]
        MEnd = ops[-1]
        lr.append([op25 / MStart - 1, op50 / MStart - 1, op75 / MStart - 1, MEnd / MStart - 1])
    df = pd.DataFrame(lr, columns=["R25M1", "R50M1", "R75M1", "REndM1"])
    return df


def get_MonthIs(YearIs, StartMonthI=1, EndMonthI=12):
    LastYearidx = len(YearIs) - 1
    lr = []
    for idx, YearI in enumerate(YearIs):
        lr.extend([YearI * 100 + MonthI for MonthI in
                   list(range(StartMonthI if idx == 0 else 1, EndMonthI + 1 if idx == LastYearidx else 13, 1))])
    return lr


def ana_earning(portfolio,strategy,experiment,ref_index=""):
    account_detail_fnwp=os.path.join(AT_dir,portfolio,strategy,experiment,"AT_AccountDetail.csv")
    dfec = pd.read_csv(os.path.join(AT_dir, portfolio,strategy, experiment,"experiment_config.csv"))

    df=pd.read_csv(account_detail_fnwp)
    #df["Total"]=df["Cash_after_closing"]+df["MarketValue_after_closing"]
    df["Total"] = df["Cash_after_closing"]
    for i in list(range(len(dfec))):
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
    for i in list(range(len(dfec))):
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
    if len(ref_index)!=0:
        dfrefidx=get_months_statics_Index(ref_index, dfm["MonthI"].values)
        ax3=allaxes[1].twinx()
        for title in dfrefidx.columns:
            ax3.plot(dfrefidx[title],label=title)
        ax3.legend(loc='upper right')

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
    dfec = pd.read_csv(os.path.join(AT_dir, portfolio, strategy,experiment, "experiment_config.csv"))

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
            for emidx in list(range(len(dfec))):
                for logs in [ll_log_Earnsold[emidx], ll_log_Losssold[emidx]]:
                    if len(logs) != 0:
                        for log in logs:
                            a = re.findall(r'(\w+):([-+]?[0-9]+[.][0-9]*)', log)
                            if len(a) != 0:
                                trans.append([int(DateS), emidx,a[0][0], eval(a[0][1])])
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


