import numpy as np
import pandas as pd
def cal_r(input_1, output, num):
    return np.exp(np.log(output/input_1)/(num+num-1+num-2))-1

def cal_r2(input_1, deduct_1, rr, num_year):

    last_year_value=input_1
    result = [last_year_value]
    for idx in range(num_year):
        last_year_value=last_year_value*(1+rr)-deduct_1
        result.append(last_year_value)
    return result

def sim(input_1, deduct_1, num_year, fnwp="/home/rdchujf/result.csv"):

    df=pd.DataFrame()
    for rr in np.arange(0.001, 0.01,0.0005 ):
        print "handling {0}".format(rr)
        a=cal_r2(input_1, deduct_1, rr, num_year)
        df["{0}".format(rr)]=a
    df.to_csv(fnwp, index=False,float_format='%.2f')
    return df

def cr(invest, result, ny):
    return np.exp(np.log(result*1.0/invest)/ny)-1


def j2(invest,r, deduct, ny):
    r1=1 + r
    sum=invest * (np.power(r1, 5) + np.power(r1, 4) + np.power(r1, 3) + np.power(r1, 2) + np.power(r1, 1))
    result=[]
    for _ in range(ny):
        sum=(sum-deduct)*(r1)
        result.append(sum)
    return result


def j3(r):
    r1=1 + r
    invest=400000.0
    deduct1=22952.63
    deduct2=45905.25
    ny2=52
    sum=0
    result = []
    for idx in range(5):
        sum=invest+sum*r1
        result.append(sum)
    for _ in range(12-5):
        sum=(sum-deduct1)*r1
        result.append(sum)
    for _ in range(ny2-11):
        sum=(sum-deduct2)*r1
        result.append(sum)
    return result

def simj():
    #fnwp = "/home/rdchujf/Downloads/jiaohanginsurance.csv"
    #df = pd.read_csv(fnwp)
    df=pd.DataFrame()
    for rr in np.arange(0.01, 0.06,0.001 ):
        print "handling {0}".format(rr)
        a=j3(rr)
        df["{0}".format(rr)]=a
    df.to_csv("/home/rdchujf/Downloads/jiaoresult.csv",index=False, float_format='%.2f')
    return df

def jhbx():
    bx_fnwp = "/home/rdchujf/Downloads/jiaohanginsurance.csv"
    wn_fnwp = "/home/rdchujf/Downloads/jiaohangwenneng.csv"
    dfbx = pd.read_csv(bx_fnwp)
    dfwn = pd.read_csv(wn_fnwp)
    dfr = pd.DataFrame()
    dfr["baoxian"] = dfbx["baoxian"]
    dfr["waneng"] = dfwn["waneng"]
    dfr["result"] = dfr["baoxian"] + dfr["waneng"]
    dfr["result_shift"] = dfr["result"].shift(1)
    dfr.loc[0, "result_shift"] = 1834260.60
    dfr["rate"] = dfr["result"] / dfr["result_shift"]
    fuli = []
    for idx, row in dfr[["result"]].iterrows():
        fuli.append(cr(2000000.00, row.result, idx))
    return fuli

dfc=pd.read_csv("/home/rdchujf/Documents/insurance/Shanghai_insurance/cash_value.csv")
dfs=pd.read_csv("/home/rdchujf/Documents/insurance/Shanghai_insurance/accumulate_shencun.csv")
df=pd.DataFrame()
df["cash"]=dfc["cash_value"]
df["shencun"]=dfs["accumulate_shencun"]
df["total"]=df["cash"]+df["shencun"]



invest=500000.0

def cal_result(invest, r, n):
    return invest*(np.power(1+r,n)+np.power(1+r,n-1)+np.power(1+r,n-2))

def find_fuli(table_value, ny):
    l_result=[]
    l_rate=[]
    invest=500000.0
    for rate in np.arange(0.01, 0.06,0.001):
        l_result.append(cal_result(invest,rate,ny))
        l_rate.append(rate)
    for idx, item in enumerate(l_result):
        if item >table_value:
            return l_rate[idx]

l_fuli=[]
for idx, row in df.iterrows():
    l_fuli.append(find_fuli(row.total,idx+1))
df["fuli"]=l_fuli

invest=2400000.0
def cal_r(invest, result,ny):
    return np.exp(np.log(result/invest)/ny)-1

l_fuli=[]
for idx, row in df.iterrows():
    l_fuli.append(cal_r(invest, row.total,idx+1))
df["fuli"]=l_fuli


