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
