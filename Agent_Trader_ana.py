import os,json,re
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def List_Strategies_Config(portfolio):
    dnwp=os.path.join("/mnt/data_disk2/n_workspace/AT",portfolio)
    dirs=[item for item in os.listdir(dnwp) if os.path.isdir(os.path.join(dnwp,item))]
    df=[]
    titles=[]
    for dir in dirs:
        param=json.load(open(os.path.join(dnwp, dir,"config.json"),"r"),object_pairs_hook=OrderedDict)
        if len(titles)==0:
            titles=list(param.keys())
            titles.remove("strategy_fun")
            titles.remove("GPU_mem")
        item =[param[key] for key in titles]+[dir]
        if len(df)==0:
            df=pd.DataFrame([item],columns=titles+["strategy_name"])
        else:
            df.loc[len(df)]=item
    df.sort_values(["RL_system_name"],inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df

def ana_earning(portfolio,strategy,experiment):
    base_dir="/mnt/data_disk2/n_workspace/AT/"
    account_detail_fnwp=os.path.join(base_dir,portfolio,strategy,experiment,"AT_AccountDetail.csv")

    df=pd.read_csv(account_detail_fnwp)
    df["Total"]=df["Cash_after_closing"]+df["MarketValue_after_closing"]
    df["MonthI"]=df["DateI"]//100
    dfm=df[["MonthI","Total"]].groupby(["MonthI"]).agg(monthly_start=pd.NamedAgg(column="Total", aggfunc="first"),
                                                 monthly_end=pd.NamedAgg(column="Total", aggfunc="last"))

    dfm["Month_earn"]=dfm["monthly_end"]-dfm["monthly_start"]
    dfm.reset_index(inplace=True)

    plt.rcParams["figure.figsize"] = (20, 10)
    fig = plt.figure()
    fig.add_subplot(211)
    fig.add_subplot(212)
    allaxes = fig.get_axes()

    allaxes[0].set_title("Total, Cash and Market Value")
    allaxes[0].plot(df["Cash_after_closing"],label="Cash_after_closing",color='m')
    allaxes[0].plot(df["MarketValue_after_closing"],label="MarketValue_after_closing",color='r')

    allaxes[0].legend(loc='upper left')
    ax2=allaxes[0].twinx()
    ax2.plot(df["Total"],label="Total")
    ax2.legend(loc='upper right')

    allaxes[1].set_title("Monthly Earning")
    allaxes[1].plot(dfm["Month_earn"],label="Month_earn")
    allaxes[1].plot([0,len(dfm)],[0,0])
    allaxes[1].set_xticks(list(range(len(dfm))))
    allaxes[1].set_xticklabels([str(MonthI) for MonthI in dfm["MonthI"].tolist()],rotation=90)
    allaxes[1].legend(loc='upper left')
    return df, dfm


def get_data_from_report(fnwp):
    patterns=[r'Today Sold with Earn (\d+)',
             r'Today Sold with Loss (\d+)',
             r'Today Sold with Balance (\d+)',
             r'Tommorow to Buy (\d+)',
             r'Tommorow to Sell (\d+)',
             r'Tommorow not_buy_due_limit (\d+)',
             r'Tommorow multibuy\(not sell due to multibuy\) (\d+)']
    with open(fnwp,"r") as f:
        flag_found=[False for _ in patterns]
        results=[np.NaN for _ in patterns]
        for line in f.readlines():
            for idx in list(range(len(patterns))):
                if flag_found[idx]:
                    continue
                else:
                    a=re.findall(patterns[idx],line)
                    if len(a)!=0:
                        assert results[idx]!=results[idx]
                        results[idx]=int(a[0])
                        flag_found[idx]=True
        assert all(flag_found),flag_found
    return results
#get_data_from_report("/mnt/data_disk2/n_workspace/AT/p1/s17/eF1/201801/20180102/Report.txt")

def get_datas_for_experiment(portfolio, strategy, experiment):
    dnwp="/mnt/data_disk2/n_workspace/AT/"
    ednwp=os.path.join(dnwp,portfolio, strategy, experiment)
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
    df=pd.DataFrame(Datas, columns=["SoldE","SoldL","SoldB","ToBuy","ToSell","NotToBuy_due_limit","Multibuy","DateI"])

    df["MonthI"] = df["DateI"] // 100
    dfm = df[["MonthI", "SoldE", "SoldL", "SoldB", "ToBuy","ToSell","NotToBuy_due_limit","Multibuy"]].groupby(["MonthI"]).agg(
        CEarn=pd.NamedAgg(column="SoldE", aggfunc=sum),
        CLoss=pd.NamedAgg(column="SoldL", aggfunc=sum),
        CBalance=pd.NamedAgg(column="SoldB", aggfunc=sum),
        CToBuy=pd.NamedAgg(column="ToBuy", aggfunc=sum),
        CToSell=pd.NamedAgg(column="ToSell", aggfunc=sum),
        CNotToBuy_due_limit=pd.NamedAgg(column="NotToBuy_due_limit", aggfunc=sum),
        CMultibuy=pd.NamedAgg(column="Multibuy", aggfunc=sum)
    )

    dfm["CTotal"] = dfm.CEarn + dfm.CLoss + dfm.CBalance

    dfm["REarn"] = dfm.CEarn / dfm.CTotal
    dfm["RLoss"] = dfm.CLoss / dfm.CTotal
    dfm["RBalance"] = dfm.CBalance / dfm.CTotal
    pd.options.display.float_format = "{:,.2f}".format
    dfm.fillna(0, inplace=True)
    return df,dfm
