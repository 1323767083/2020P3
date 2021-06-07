import os
import pandas as pd
import numpy as np

#sample process_names=["EvalAgent_4","EvalAgent_5"]
def ET_WR_detial(system_name, ET,process_names ):
    df=[]
    for process_name in process_names:
        dnwp=os.path.join("/mnt/data_disk2/n_workspace/RL/",system_name,"Classifiaction","WR",process_name)
        fwnp=os.path.join(dnwp,"ET{0}.csv".format(ET))
        dft=pd.read_csv(fwnp)
        if len(df)==0:
            df=dft
        else:
            df= pd.concat([df, dft], axis=0,ignore_index=True)

    df["buy"]=df["BW"]+df["BZ"]+df["BR"]
    df["all"]=df["BW"]+df["BZ"]+df["BR"]+df["NW"]+df["NZ"]+df["NR"]+df["NA"]
    assert len(set(df["all"].to_list()))==1, df["all"]
    print ("{0} {1:.2f} {2:.2f} {3:.2f} {4:.2f} {5}".format(ET,df["BW"].sum()/df["buy"].sum(),
                                                        df["BR"].sum()/df["buy"].sum(),
                                                        (df["BR"].sum()+df["BZ"].sum())/df["buy"].sum(),
                                                        df["buy"].sum()/ df["all"].sum(),df["buy"].sum()), end=" ")
    return df

def ET_PA_detial(system_name, ET,process_names):
    df=pd.DataFrame()
    for process_name in process_names:
        dnwp=os.path.join("/mnt/data_disk2/n_workspace/RL/",system_name,"Classifiaction","PA",process_name)
        fwnp=os.path.join(dnwp,"ET{0}.csv".format(ET))
        dft=pd.read_csv(fwnp)
        if len(df)==0:
            df=dft
        else:
            df= pd.concat([df, dft], axis=0,ignore_index=True)
    npa = df.values
    if npa[0,0]>20000000: # first column is dateI:
        npa=npa[:,1:]
    npr=npa[(npa<5)&(npa>-5)]
    p25=np.percentile(npr, 25, axis=0)
    p50=np.percentile(npr, 50, axis=0)
    p75=np.percentile(npr, 75, axis=0)
    total=npr.sum()-len(npr)*0.0016
    print ("{0} {1:.2f} {2:.2f} {3:.2f} {4:.2f}".format(ET, p25,p50,p75,total), end=" ")
    return df


def ET_WR_detial_old(system_name, ET):
    process_name1="EvalAgent_4"
    dnwp1=os.path.join("/mnt/data_disk2/n_workspace/RL/",system_name,"Classifiaction","WR",process_name1)

    process_name2="EvalAgent_5"
    dnwp2=os.path.join("/mnt/data_disk2/n_workspace/RL/",system_name,"Classifiaction","WR",process_name2)
    fwnp1=os.path.join(dnwp1,"ET{0}.csv".format(ET))
    df1=pd.read_csv(fwnp1)

    fwnp2=os.path.join(dnwp2,"ET{0}.csv".format(ET))
    df2=pd.read_csv(fwnp2)

    df= pd.concat([df1, df2], axis=0,ignore_index=True)

    df["buy"]=df["BW"]+df["BZ"]+df["BR"]
    df["all"]=df["BW"]+df["BZ"]+df["BR"]+df["NW"]+df["NZ"]+df["NR"]+df["NA"]
    assert len(set(df["all"].to_list()))==1, df["all"]
    print ("{0} {1:.2f} {2:.2f} {3:.2f} {4:.2f} {5}".format(ET,df["BW"].sum()/df["buy"].sum(),
                                                        df["BR"].sum()/df["buy"].sum(),
                                                        (df["BR"].sum()+df["BZ"].sum())/df["buy"].sum(),
                                                        df["buy"].sum()/ df["all"].sum(),df["buy"].sum()), end=" ")
    return df

def ET_PA_detial_old(system_name, ET):
    process_name1="EvalAgent_4"
    dnwp1=os.path.join("/mnt/data_disk2/n_workspace/RL/",system_name,"Classifiaction","PA",process_name1)

    process_name2="EvalAgent_5"
    dnwp2=os.path.join("/mnt/data_disk2/n_workspace/RL/",system_name,"Classifiaction","PA",process_name2)

    fwnp1=os.path.join(dnwp1,"ET{0}.csv".format(ET))
    df1=pd.read_csv(fwnp1)

    fwnp2=os.path.join(dnwp2,"ET{0}.csv".format(ET))
    df2=pd.read_csv(fwnp2)

    df= pd.concat([df1, df2], axis=0,ignore_index=True)

    npa=df.values
    npr=npa[(npa<5)&(npa>-5)]
    p25=np.percentile(npr, 25, axis=0)
    p50=np.percentile(npr, 50, axis=0)
    p75=np.percentile(npr, 75, axis=0)
    total=npr.sum()-len(npr)*0.0016
    print ("{0} {1:.2f} {2:.2f} {3:.2f} {4:.2f}".format(ET, p25,p50,p75,total), end=" ")
    return df

def get_summery_per_month_df(dnwp,lsdn):
    ETs=[int (fn[2:-4]) for fn in os.listdir(os.path.join(dnwp,lsdn[0]))]
    ETs.sort()
    df=[]
    for lsdnidx, sdn in enumerate(lsdn):
        dft=pd.read_csv(os.path.join(dnwp,sdn,"ET{0}.csv".format(ETs[0])))
        if lsdnidx==0:
            df=dft
        else:
            df=pd.concat([dft, df], axis=0, ignore_index=True)
    df["MonthI"]=df["DateI"].apply(lambda x: int(x/100))
    MonthIs=list(set(df["MonthI"].tolist()))
    MonthIs.sort()

    dfBRRatio=pd.DataFrame(columns=["ET"]+[str(MonthI) for MonthI in MonthIs])
    dfBRatio=pd.DataFrame(columns=["ET"]+[str(MonthI) for MonthI in MonthIs])
    dfTotalC=pd.DataFrame(columns=["ET"]+[str(MonthI) for MonthI in MonthIs])
    for ETidx,ET in enumerate(ETs):
        for lsdnidx, sdn in enumerate(lsdn):
            dft=pd.read_csv(os.path.join(dnwp,sdn,"ET{0}.csv".format(ET)))
            if lsdnidx==0:
                df=dft
            else:
                df=pd.concat([dft, df], axis=0, ignore_index=True)
        df["MonthI"]=df["DateI"].apply(lambda x: int(x/100))

        dfg=df[["BW","BZ","BR","NW","NZ","NR","NA","MonthI"]].groupby(["MonthI"]).agg(
        BR=pd.NamedAgg(column="BR", aggfunc="sum"),
        BZ=pd.NamedAgg(column="BZ", aggfunc="sum"),
        BW=pd.NamedAgg(column="BW", aggfunc="sum"),
        NW=pd.NamedAgg(column="NW", aggfunc="sum"),
        NZ=pd.NamedAgg(column="NZ", aggfunc="sum"),
        NR=pd.NamedAgg(column="NR", aggfunc="sum"),
        NA=pd.NamedAgg(column="NA", aggfunc="sum")
           )
        dfg["BRRatio"]=dfg["BR"]/(dfg["BR"]+dfg["BZ"]+dfg["BW"])
        dfg["BRatio"]=(dfg["BR"]+dfg["BZ"]+dfg["BW"])/(dfg["BR"]+dfg["BZ"]+dfg["BW"]+dfg["NW"]+dfg["NZ"]+dfg["NR"]+dfg["NA"])
        dfg["TotalC"]=dfg["BR"]+dfg["BZ"]+dfg["BW"]+dfg["NW"]+dfg["NZ"]+dfg["NR"]+dfg["NA"]

        dfBRRatio.loc[len(dfBRRatio)]=[ET]+dfg["BRRatio"].tolist()
        dfBRatio.loc[len(dfBRatio)]=[ET]+dfg["BRatio"].tolist()
        dfTotalC.loc[len(dfTotalC)]=[ET]+dfg["TotalC"].tolist()
    return dfBRRatio,dfBRatio,dfTotalC
