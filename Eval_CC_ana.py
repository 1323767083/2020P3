import pandas as pd
import numpy as np
import prettytable as pt
import os,re
import config as sc
import matplotlib
import matplotlib.pyplot as plt


def get_CCET_fnwp(system_name, Eval_Group_idx,ET, flag_old):
    dnwp=os.path.join(sc.base_dir_RL_system,system_name,"CC")
    if flag_old:
        fnwp=os.path.join(dnwp,"ET{0}.csv".format(ET))
    else:
        fnwp=os.path.join(dnwp,"EvalGroup{0}".format(Eval_Group_idx),"ET{0}.csv".format(ET))
    return fnwp

def get_money_in_hand_fnwp(system_name, Eval_Group_idx,ET, flag_old):
    dnwp=os.path.join(sc.base_dir_RL_system,system_name,"CC")
    if flag_old:
        fnwp=os.path.join(dnwp,"ET{0}_money_in_hand.csv".format(ET))
    else:
        fnwp=os.path.join(dnwp,"EvalGroup{0}".format(Eval_Group_idx),"ET{0}_money_in_hand.csv".format(ET))
    return fnwp

def get_action_decison_fnwp(system_name, Eval_Group_idx,ET, flag_old):
    dnwp=os.path.join(sc.base_dir_RL_system,system_name,"CC")
    if flag_old:
        fnwp=os.path.join(dnwp,"ET{0}_action_decision.csv".format(ET))
    else:
        fnwp=os.path.join(dnwp,"EvalGroup{0}".format(Eval_Group_idx),"ET{0}_action_decision.csv".format(ET))
    return fnwp

def get_CC_ET_list_and_CC_type(system_name,Eval_Group_idx):
    dnwp=os.path.join(sc.base_dir_RL_system,system_name,"CC")
    l_sub_dir=[fn for fn in os.listdir(dnwp) if "EvalGroup" in fn]
    flag_old=True if len(l_sub_dir)==0 else False
    if flag_old:
        ETs=[int(re.findall(r'ET(\d+).csv',fn)[0]) for fn in os.listdir(dnwp) if "_" not in fn]
        ETs.sort()
    else:
        ETs=[int(re.findall(r'ET(\d+).csv',fn)[0]) for fn in os.listdir(os.path.join(dnwp,"EvalGroup{0}".format(Eval_Group_idx))) if "_" not in fn]
        ETs.sort()
    return flag_old,ETs

def get_acc_earn(system_name,Eval_Group_idx):
    flag_old,ETs=get_CC_ET_list_and_CC_type(system_name,Eval_Group_idx)
    acc_earn=[]
    for ET in ETs:
        fnwp=get_CCET_fnwp(system_name, Eval_Group_idx,ET,flag_old)
        df=pd.read_csv(fnwp)
        df["StockS"]=df["StockI"].apply(lambda x: "SH{0:06d}".format(int(x)) if x>=600000 else "SZ{0:06d}".format(int(x)))
        dfr=df[["StockS","Sell_Earn"]].groupby(["StockS"]).agg(totalEarn=pd.NamedAgg(column="Sell_Earn", aggfunc="sum"))
        acc_earn.append(dfr["totalEarn"].sum())
    return ETs,acc_earn

def get_stock_from_idx(dfsl, idx):
    return dfsl.loc[idx]["Stock"]

def plot_1_experiment(system_name, Eval_Group_idx):
    flag_old, ETs = get_CC_ET_list_and_CC_type(system_name, Eval_Group_idx)
    ETs, acc_earn = get_acc_earn(system_name, Eval_Group_idx)
    plt.rcParams["figure.figsize"] = (20, 10)
    plt.plot(ETs, acc_earn)
    #plt.xticks(np.arange(len(ETs)), ETs)
    plt.title("Accumulate Earning {0} group {1}".format(system_name, Eval_Group_idx))

def plot_multi_experiments(system_names__group_idxs,ET_tb_entropy_threadhold):
    plt.rcParams["figure.figsize"] = (20, 10)
    fig = plt.figure()
    for idx, _ in enumerate(system_names__group_idxs):
        fig.add_subplot(len(system_names__group_idxs) * 100 + 10 + idx + 1)

    allaxes = fig.get_axes()
    l_acc_earn = []
    l_ETs = []
    for idx, [system_name, group_idx] in enumerate(system_names__group_idxs):
        print(system_name, group_idx, end=' ')
        ETs, acc_earn = get_acc_earn(system_name, group_idx)
        l_acc_earn.append(acc_earn)
        l_ETs.append(ETs)
        allaxes[idx].plot(l_ETs[idx], l_acc_earn[idx])
        allaxes[idx].set_title("Accumulate Earning {0} group {1}".format(system_name, group_idx))
        allaxes[idx].plot([l_ETs[idx][0], l_ETs[idx][-1]], [0, 0])
        reverse_sorted_idxes = sorted(list(range(len(l_acc_earn[idx]))), key=lambda x: l_acc_earn[idx][x], reverse=True)
        print('max 5 ET and acc_earn:')
        for a in reverse_sorted_idxes[:5]:
            print("{0} {1:.2f}".format(l_ETs[idx][a], l_acc_earn[idx][a]), end='######')
        print("\n")

    for [_, i] in system_names__group_idxs:
        print ("location gruop idx ", i)
        reverse_sorted_idxes=sorted(list(range(len(l_acc_earn[i]))), key=lambda x:l_acc_earn[i][x],reverse=True)
        for a in reverse_sorted_idxes[:30]:
            if l_ETs[i][a]<ET_tb_entropy_threadhold:
                print ("{0} {1:.2f}".format(l_ETs[i][a],l_acc_earn[i][a]))
    #plt.show()

def ana_transaction_1ET_on_count(system_name,Eval_Group_idx,ET,flag_old):
    plt.rcParams["figure.figsize"] = (20, 10)

    dfl = pd.read_csv(get_CCET_fnwp(system_name, Eval_Group_idx, ET, flag_old))
    dfl["StockS"] = dfl["StockI"].apply(
        lambda x: "SH{0:06d}".format(int(x)) if x >= 600000 else "SZ{0:06d}".format(int(x)))
    dfl.drop("StockI", axis=1, inplace=True)
    dfl["TransIDI"] = dfl["TransIDI"].astype(int)
    dfl["DateI"] = dfl["DateI"].astype(int)

    dfd = pd.read_csv(get_action_decison_fnwp(system_name, Eval_Group_idx, ET, flag_old))
    dfm = pd.read_csv(get_money_in_hand_fnwp(system_name, Eval_Group_idx, ET, flag_old))

    dflr = dfl[['StockS', 'TransIDI', 'DateI', 'Sell_Earn', 'AcutalAction']].groupby(['StockS', 'TransIDI']). \
        agg(StartDateI=pd.NamedAgg(column='DateI', aggfunc=lambda x: [xi for xi in x][0]),
            EndDateI=pd.NamedAgg(column='DateI', aggfunc=lambda x: [xi for xi in x][-1]),
            ResultM=pd.NamedAgg(column='Sell_Earn', aggfunc="sum"),
            F_Buyed=pd.NamedAgg(column='AcutalAction', aggfunc=lambda x: any([True if xi == 0 else False for xi in x])),
            F_Selled=pd.NamedAgg(column='AcutalAction',
                                 aggfunc=lambda x: any([True if xi == 2 else False for xi in x])))
    dflr = dflr[dflr["F_Buyed"] & dflr["F_Selled"]][["StartDateI", "EndDateI", "ResultM"]]
    dflr.reset_index(inplace=True)
    print("sell_earn statis:")
    print(dflr.ResultM.describe())
    dflrc = dflr.groupby(["EndDateI"]).agg(
        CPr=pd.NamedAgg(column='ResultM', aggfunc=lambda x: len([xi for xi in x if xi > 0])),
        CNr=pd.NamedAgg(column='ResultM', aggfunc=lambda x: len([xi for xi in x if xi < 0])),
        CZr=pd.NamedAgg(column='ResultM', aggfunc=lambda x: len([xi for xi in x if xi == 0])))

    print("Pluse Count {0} Minus count {1} Zero count {2}".format(dflrc.CPr.sum(), dflrc.CNr.sum(), dflrc.CZr.sum()))
    fig = plt.figure()
    fig.add_subplot(211)
    fig.add_subplot(212)
    allaxes = fig.get_axes()
    allaxes[0].hist(dflr.ResultM, bins=np.arange(-5000, 20000, 1000))
    allaxes[0].set_title("Trans Count Statics {0} group {1}".format(system_name, Eval_Group_idx))
    allaxes[1].stackplot(list(range(len(dflrc))), dflrc.CPr, dflrc.CNr, dflrc.CZr, labels=['Pluse', 'Minus', 'Zero'])
    allaxes[1].set_xticks(list(range(len(dflrc))))
    allaxes[1].set_xticklabels([str(dayI) for dayI in dflrc.index.tolist()],rotation=90)
    allaxes[1].set_title("Trans Count (Plus Minus Zero) {0} group {1}".format(system_name, Eval_Group_idx))
    allaxes[1].legend(loc='upper left')

def ana_transaction_1ET_on_stock(system_name,Eval_Group_idx,ET,flag_old):

    pd.options.display.float_format = "{:,.2f}".format


    #flag_old, ETs = get_CC_ET_list_and_CC_type(system_name, Eval_Group_idx)
    fnwp = get_CCET_fnwp(system_name, Eval_Group_idx, ET, flag_old)
    df = pd.read_csv(fnwp)
    df["StockS"] = df["StockI"].apply(
        lambda x: "SH{0:06d}".format(int(x)) if x >= 600000 else "SZ{0:06d}".format(int(x)))
    dfr = df[["StockS", "Sell_Earn", "AcutalAction"]].groupby(["StockS"]).agg(
        totalEarn=pd.NamedAgg(column="Sell_Earn", aggfunc="sum"),
        Sell_count=pd.NamedAgg("AcutalAction", lambda x: len([True for xi in x if xi == 2])))
    dfr["Stock"] = dfr.index
    dfr.sort_values(by=["totalEarn"], inplace=True)
    tb = pt.PrettyTable()
    tb.field_names = dfr.columns
    tb.float_format = ".2"
    for idx, row in dfr.iterrows():
        tb.add_row(row)
    print(tb)
    print(dfr["totalEarn"].sum())

def ana_trans_1ET_1Stock(system_name,Eval_Group_idx,ET,Stock,flag_old):
    pd.options.display.float_format = "{:,.2f}".format

    fnwp = get_CCET_fnwp(system_name, Eval_Group_idx, ET, flag_old)

    df = pd.read_csv(fnwp)
    df["StockS"] = df["StockI"].apply(
        lambda x: "SH{0:06d}".format(int(x)) if x >= 600000 else "SZ{0:06d}".format(int(x)))

    df = df[["AcutalAction", "Holding_Invest", "Buy_Invest", "Sell_Earn", "Sell_Return", "StockS", "TransIDI", "DateI",
             "Eval_Profit"]]
    dfp = pd.pivot_table(df, index=["StockS", "TransIDI", "DateI"])
    #dfp.loc[Stock]

    tb = pt.PrettyTable()
    tb.field_names = dfp.columns
    tb.float_format = ".2"
    for idx, row in dfp.loc[Stock].iterrows():
        tb.add_row(row)
    print(tb)

def ana_earn(system_name, Eval_Group_idx, ET, flag_old):
    fnwp = get_money_in_hand_fnwp(system_name, Eval_Group_idx, ET, flag_old)
    df = pd.read_csv(fnwp)

    plt.rcParams["figure.figsize"] = (20, 10)
    num_fig = 4
    fig = plt.figure()
    for idx, _ in enumerate(range(num_fig)):
        fig.add_subplot(num_fig * 100 + 10 + idx + 1)

    allaxes = fig.get_axes()

    for title in ["Money_in_hand", "Eval_holding", "Eval_Ttotal"]:
        allaxes[0].plot(df[title], label=title)
    allaxes[0].plot([0, len(df)], [1000000, 1000000])
    allaxes[0].legend()
    allaxes[0].set_title("Account trend {0} group {1}".format(system_name, Eval_Group_idx))

    earn_loss = np.diff(df["Eval_Ttotal"].values, axis=0)
    earn_loss = np.delete(earn_loss, -1, 0)
    allaxes[1].plot(earn_loss)
    allaxes[1].plot([0, len(earn_loss)], [0, 0])
    allaxes[1].set_title("Daily Earning {0} group {1}".format(system_name, Eval_Group_idx))

    print("sum {0:.2f}".format(sum(earn_loss)))

    allaxes[2].plot(df["Tinpai_huaizhang"], label="Tinpai_huaizhang")
    allaxes[2].legend()
    allaxes[2].set_title("Tinpai_huaizhang {0} group {1}".format(system_name, Eval_Group_idx))

    PN_count = [1 if a > 0 else -1 if a < 0 else 0 for a in earn_loss]
    allaxes[3].plot(PN_count)

    period_len = 5
    period_earn_loss = np.zeros(len(earn_loss))
    for i in list(range(len(earn_loss) // period_len + 1)):
        period_sum = sum(earn_loss[i * period_len:(i + 1) * period_len])
        period_earn_loss[i * period_len:(i + 1) * period_len] = 1 if period_sum > 0 else -1 if period_sum < 0 else 0
        print('{0} {1:.2f}'.format(i, period_sum))

    allaxes[3].plot(period_earn_loss)
    allaxes[3].set_title("Daily/Biweekly Win loss {0} group {1}".format(system_name, Eval_Group_idx))
    allaxes[3].set_xticks(list(range(len(df))))
    allaxes[3].set_xticklabels([str(int(dayI)) for dayI in df["DateI"]],rotation=90)

    print("earn day {0} loss day {1}  balance day {2}".format(PN_count.count(1), PN_count.count(-1), PN_count.count(0)))

def ana_action(system_name, Eval_Group_idx, ET, flag_old):
    df = pd.read_csv(get_CCET_fnwp(system_name, Eval_Group_idx, ET, flag_old))
    df.dropna(subset=["AcutalAction"], inplace=True)
    df["StockS"] = df["StockI"].apply(
        lambda x: "SH{0:06d}".format(int(x)) if x >= 600000 else "SZ{0:06d}".format(int(x)))
    df.drop(labels=["StockI"], axis=1, inplace=True)

    dfc = df.groupby(["DateI"]).agg(
        buy_count=pd.NamedAgg(column="AcutalAction", aggfunc=lambda x: sum([True for xi in x if xi == 0])),
        sell_count=pd.NamedAgg(column="AcutalAction", aggfunc=lambda x: sum([True for xi in x if xi == 2])))
    dfc.reset_index(inplace=True)
    assert len(dfc) == len(set(df["DateI"].to_list()))


    plt.rcParams["figure.figsize"] = (20, 10)
    num_fig = 4
    fig = plt.figure()
    for idx, _ in enumerate(range(num_fig)):
        fig.add_subplot(num_fig * 100 + 10 + idx + 1)

    allaxes = fig.get_axes()

    dfad = pd.read_csv(get_action_decison_fnwp(system_name, Eval_Group_idx, ET, flag_old))
    cols = dfad.columns.to_list()
    cols.remove("DateI")
    for col in cols:
        dfad["C_" + col] = dfad[col].apply(lambda x: len(x.split("_")) if not x != x else 0)
        dfad.drop(labels=[col], axis=1, inplace=True)
    dfad.reset_index(inplace=True)


    allaxes[0].plot(dfc["buy_count"])
    allaxes[0].plot(dfc["sell_count"])

    for col in ["not_buy_due_low_profit", "sell_due_low_profit"]:
        allaxes[1].plot(dfad["C_" + col], label=col)
    allaxes[1].legend(loc="upper left")

    col = "not_buy_due_limit"
    allaxes[2].plot(dfad["C_" + col], label=col, color='m')
    allaxes[2].legend(loc="upper right")

    col = 'multibuy'
    allaxes[3].plot(dfad["C_" + col], label=col, color='m')
    allaxes[3].legend(loc="upper right")

    allaxes[3].set_xticks(list(range(len(dfad))))
    allaxes[3].set_xticklabels([str(int(dayI)) for dayI in dfad["DateI"]],rotation=90)

    return dfad