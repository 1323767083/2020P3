import os, sys, pickle
import config as sc
import numpy as np
from nets import lconfig
from nets_agent import init_agent_config
from nets_trainer import init_trainer_config, PG_trainer
from data_T5 import R_T5,R_T5_scale,R_T5_balance,R_T5_skipSwh,R_T5_skipSwh_balance
from data_common import API_trade_date
from prettytable import PrettyTable
from recorder import record_variable
import matplotlib.pyplot as plt
import pandas as pd


cn_lv = ["hp0", "hp25", "hp5", "hpr75", "hp1",
         "tradable_per",
         "Sdd_mdn_hp", "Sdd_avrg_hp", "Sxd_per", "Sdd_per",
         "Bdd_mdn_hp", "Bdd_avrg_hp", "Bxd_per", "Bdd_per",
         "stock_S20V20High", "stock_S20V20Low",
         "syuan_SwhV20"
         ]

###################################################################################################
#Ana record variable
###################################################################################################
RL_base_dir="/home/rdchujf/n_workspace/RL"
Ana_sub_dir="analysis_result"
fn_match_fn_tc="match_fn_tc.csv"
fn_cal_advent="cal_advent.csv"
fn_lv_normal="fn_lv_normal.pickle"
fn_lv_normal_mean_std="fn_lv_normal_mean_std.csv"
#####Ana record variable common
def get_tc(argv):
    assert len(argv) == 2
    system_name=argv[0]
    fn=argv[1]
    input_fnwp=os.path.join(RL_base_dir, system_name,Ana_sub_dir,fn_match_fn_tc)
    if not os.path.exists(input_fnwp):
        match_fn_tc([system_name])
    df=pd.read_csv(input_fnwp)
    for _, row in df.iterrows():
        if fn>=row["fn_start"] and fn <=row["fn_end"]:
            return row["Train_count"]
    return None

def read_advent(argv):
    assert len(argv)==1
    system_name = argv[0]
    cal_advent_fnwp=os.path.join(RL_base_dir, system_name,Ana_sub_dir,fn_cal_advent)
    if not os.path.exists(cal_advent_fnwp):
        match_fn_tc_fnwp = os.path.join(RL_base_dir, system_name, Ana_sub_dir, fn_match_fn_tc)
        if not os.path.exists(match_fn_tc_fnwp):
            match_fn_tc([system_name])
        cal_advent([system_name])
    df=pd.read_csv(cal_advent_fnwp)
    return df

#####Ana record variable step 1
def match_fn_tc(argv):
    assert len(argv)==1
    system_name=argv[0]
    #source_dir="/home/rdchujf/n_workspace/RL/htryc/record_variables"
    source_dir = os.path.join(RL_base_dir,system_name,"record_state")
    result_dir=os.path.join(RL_base_dir,system_name,Ana_sub_dir)
    if not os.path.exists(result_dir): os.makedirs(result_dir)
    result_fnwp=os.path.join(result_dir, fn_match_fn_tc)
    i=record_variable(source_dir)
    a3r = []
    #lfn=[fn for fn in os.listdir(source_dir) if fn>="2019-04-07_08:09:13_123.pkl" and fn<="2019-04-07_08:12:35_907.pkl"]
    lfn = [fn for fn in os.listdir(source_dir)]
    lfn.sort()
    for fn in lfn:
        print("handling ", fn)
        fnwp=os.path.join(source_dir,fn)
        _,_,a3=i.loader(fnwp)
        a3r.append(a3)
    result=[]
    flag_first=True
    result_count ,result_start_fn,  tem_end_fn = None, None,None
    for fn, a3i in zip(lfn,a3r):
        if flag_first:
            flag_first=False
            result_count=a3i[1]
            result_start_fn=fn
            tem_end_fn=fn
            continue
        if result_count!=a3i[1]:
            result.append([result_count,result_start_fn, tem_end_fn])
            result_count = a3i[1]
            result_start_fn=fn
            tem_end_fn=fn
        else:
            tem_end_fn = fn
    df=pd.DataFrame(result, columns=["Train_count", "fn_start", "fn_end"])
    df.to_csv(result_fnwp, index=False)
    print("result store in {0}".format(result_fnwp))

#####Ana record variable step 2
def cal_advent(argv):
    assert len(argv)==1
    system_name=argv[0]
    config_fnwp=os.path.join(RL_base_dir,system_name,"config.json")
    if not os.path.exists(config_fnwp):
        raise ValueError("{0} does not exisit".format(config_fnwp))
    lgc = sc.gconfig()
    lgc.read_from_json(config_fnwp)

    global Ctrainer,Cagent, lc 
    lc=lconfig()
    for key in list(lc.__dict__.keys()):
        lc.__dict__[key] = lgc.__dict__[key]
    init_agent_config(lc)
    init_trainer_config(lc)
    Ctrainer = globals()[lc.CLN_trainer]
    Cagent = globals()[lc.system_type]

    trainer = Ctrainer()
    rv = record_variable(lgc.record_variable_dir)

    input_fnwp=os.path.join(RL_base_dir, system_name,Ana_sub_dir,fn_match_fn_tc)
    if not os.path.exists(input_fnwp):
        match_fn_tc([system_name])
    df=pd.read_csv(input_fnwp)

    model_config_fnwp=os.path.join(lgc.brain_model_dir, "config.json")
    result = []
    for idx, row in df.iterrows():
        svc         =   row["Train_count"]
        start_fn    =   row["fn_start"]
        end_fn      =   row["fn_end"]
        print("handling {0}".format(svc))
        #get aio fn
        lAIO=[fn for fn in os.listdir(lgc.brain_model_dir) if lgc.actor_model_AIO_fn_seed in fn and "T{0}.h5py".format(svc) in fn]
        assert len(lAIO)==1
        AIO_fnwp=os.path.join(lgc.brain_model_dir, lAIO[0])
        Tmodel, Pmodel=trainer.load_train_model([AIO_fnwp,model_config_fnwp,""])

        lfn = [fn for fn in os.listdir(lgc.record_variable_dir) if fn >= start_fn and fn <= end_fn]

        for fn in lfn:
            print("\t handling {0}".format(fn))
            #fnwp=os.path.join(lgc.record_variable_dir,fn)
            #a1, a2, a3=rv.loader(fnwp)
            a1, _, _ = rv.loader(fn)

            #flag_got, s_lv, s_sv, s_av, a, r, s__lv, s__sv, s__av, done_flag, l_support_view = a1
            s_lv, s_sv, s_av, a, r, s__lv, s__sv, s__av, done_flag, l_support_view = a1
            n_s_lv = np.vstack(s_lv)
            n_s_sv = np.vstack(s_sv)
            n_s_av = np.vstack(s_av)
            n_a = np.vstack(a)
            n_r = np.vstack(r)
            n_s__lv = np.vstack(s__lv)
            n_s__sv = np.vstack(s__sv)
            n_s__av = np.vstack(s__av)

            num_record_to_train = len(n_s_lv)
            assert num_record_to_train == lgc.batch_size
            _, v = Pmodel.predict({'P_input_lv': n_s__lv, 'P_input_sv': n_s__sv, 'P_input_av': n_s__av})


            r = n_r + lgc.Brain_gamma**lgc.TDn * v

            y_pred = Tmodel.predict_on_batch({'input_l_view': n_s_lv, 'input_s_view': n_s_sv, 'input_account': n_s_av,
                                              'input_action': n_a, 'input_reward': r})

            prob,input_a,advents = y_pred[:, :3], y_pred[:, 3:6],y_pred[:, -1:]

            for idx, advent, support_view in zip(list(range(len(advents))), advents, l_support_view):
                #print support_view
                #print support_view[0]
                #if advent[0] > record_checked_threahold:
                #if advent[0] > 50:
                result.append([svc,fn,idx, support_view[0][0]["stock"], support_view[0][0]["date"], advent[0]])

    df=pd.DataFrame(result,columns=["train_count","fn","idx_in_fn","stock", "date", "advent"])
    result_fnwp=os.path.join(lgc.system_working_dir,Ana_sub_dir,fn_cal_advent)
    df.to_csv(result_fnwp, index=False)
    print("result store in {0}".format(result_fnwp))

#####Ana record variable step 3
def plot_loss_occurance(argv):
    assert len(argv)==2,"system_name  threadhold"
    system_name=argv[0]
    threadhold=int(argv[1])
    df=read_advent([system_name])
    dfr=df[df["advent"]>=threadhold]
    c = dfr[["stock", "advent"]].groupby("stock").mean()
    d = dfr.groupby('stock').count()

    dfp=pd.merge(d[["train_count"]], c, left_index=True, right_index=True)

    fig = plt.figure()
    ax=fig.add_subplot(1,1,1)

    sx=[]
    sy=[]
    for idx, row in dfp.iterrows():
        sx.append(row["train_count"])
        sy.append(row["advent"])
        ax.annotate(row.name, xy=(row["train_count"], row["advent"]), textcoords='data', fontsize=9, color='k')
    ax.scatter(sx, sy, color='r')
    ax.set_title("loss_mean vs occurance", fontdict={"fontsize": 10}, pad=2)
    ax.set_xlabel("occurance", fontdict={"fontsize": 10})
    ax.set_ylabel("loss mean", fontdict={"fontsize": 10})
    plt.show()

#####Ana record variable other functions
def relation_between_s_and_s__(argv):
    #system_name = "htryc"
    #stock = "SH600177"
    #threadhold=50
    #count = 200
    assert len(argv)==4, " system_name stock count threadhold"
    system_name = argv[0]
    stock = argv[1]
    count = int(argv[2])
    threadhold=int(argv[3])

    df=read_advent([system_name])
    dfr = df[(df["stock"] == stock) & (df["advent"] >= threadhold) & (df["train_count"] == count)]
    #dfr = df[(df["stock"] == "SH600177")  & (df["train_count"] == 100)]
    if len(dfr)==0:
        print("no data found for stock {0} threadhold {1} count {2}".format(stock,threadhold,count))
        return
    dfr=dfr.sort_values(by=["date"])
    dfr["date"] = dfr["date"].astype(str)
    dfr.reset_index(drop=True, inplace=True)

    td = API_trade_date().np_date_s
    period=td[(td>=dfr.iloc[0]["date"]) & (td<=dfr.iloc[-1]["date"])]
    if len(period) !=len(dfr):
        print("mismatch ", len(period), len(dfr))
        return
    else:
        print("match ", len(period) , len(dfr))

    sbs=step_by_step(system_name)
    for idx, row in dfr.iterrows():
        if idx ==len(dfr)-5-1:
            break
        s__date=row["date"]
        s__fn =row["fn"]
        #print s__date, s__fn


        sn_date=dfr.iloc[idx+5]["date"]
        sn__fn=dfr.iloc[idx+5]["fn"]
        #print sn_date, sn__fn
        _, _, _, state_f = sbs.load_data_from_record_variables(s__fn, stock, s__date)
        state_n, _, _, _ = sbs.load_data_from_record_variables(sn__fn, stock, sn_date)

        lv_f, sv_f, av_f, support_view_f=state_f
        lv_n, sv_n, av_n, support_view_n=state_n

        assert np.array_equal(lv_f,lv_n), "{0} {1} not ok".format(s__date, sn_date)
        assert (sv_f == sv_n).all(), "{0} {1} not ok".format(s__date, sn_date)
        assert av_f[0,7]==av_n[0,7]
        assert np.array_equal(av_f, av_n)
        print("{0} {1} ok".format(s__date, sn_date))


def compare_mean_prepare_data(argv):
    #V70
    #may_abnormal_stocks=["SH600177", "SH600639", "SH600063", "SH600213", "SH600614", "SH600519", "SH601328"]
    #server 100
    may_abnormal_stocks = ["SH600381", "SH600074", "SH600063", "SH600243"]
    assert len(argv)==1, "system_name"
    system_name=argv[0]

    source_dir = os.path.join(RL_base_dir,system_name,"record_state")
    result_dir=os.path.join(RL_base_dir,system_name,Ana_sub_dir)
    if not os.path.exists(result_dir): os.makedirs(result_dir)
    result_fnwp=os.path.join(result_dir, fn_lv_normal)
    #stock_result_fnwp = os.path.join(result_dir, "{0}_fn_lv_normal".format(stock))

    i=record_variable(source_dir)
    lfn = [fn for fn in os.listdir(source_dir)]
    lfn.sort()
    l_result= [[] for _ in range(17)]
    #l_stock_result= [[] for _ in range(17)]
    l_stock_result_fnwp=[]
    ll_stock_result=[]
    l_flag_need_handle_stock_lv=[]
    for stock in may_abnormal_stocks:
        fnwp=os.path.join(result_dir, "{0}_fn_lv_normal".format(stock))
        l_stock_result_fnwp.append(fnwp)
        ll_stock_result.append([[] for _ in range(17)])
        l_flag_need_handle_stock_lv.append(False if os.path.exists(fnwp) else True)

    flag_need_handle_normal_lv=not os.path.exists(result_fnwp)
    #flag_need_handle_stock_lv=not os.path.exists(stock_result_fnwp)
    #if flag_need_handle_normal_lv or flag_need_handle_stock_lv:
    if flag_need_handle_normal_lv or any(l_flag_need_handle_stock_lv):
        for fn in lfn:
            print("handling ", fn)
            fnwp=os.path.join(source_dir,fn)
            a1,_,_=i.loader(fnwp)
            llv,_,_,_,_,_,_,_,_,l_support_view=a1
            for lv, support_view in zip(llv,l_support_view):
                if not (support_view[0,0]["stock"] in may_abnormal_stocks) :
                    if flag_need_handle_normal_lv:
                        for idx in range(17):
                            l_result[idx].append(lv[0, 19,idx])
                else:
                    for idx, need_flag in enumerate(l_flag_need_handle_stock_lv):
                        if need_flag:
                            if support_view[0,0]["stock"]==may_abnormal_stocks[idx]:
                                for lv_idx in range(17):
                                    ll_stock_result[idx][lv_idx].append(lv[0, 19, idx])

        if flag_need_handle_normal_lv:
            pickle.dump(l_result, open(result_fnwp, 'wb'))
            print("result stored in {0}".format(result_fnwp))
        for idx, need_flag in enumerate(l_flag_need_handle_stock_lv):
            if need_flag:
                pickle.dump(ll_stock_result[idx], open(l_stock_result_fnwp[idx],"wb"))
                print("result stored in {0}".format(l_stock_result_fnwp[idx]))
    l_input_fnwp=[]
    l_output_fnwp=[]
    mean_std_result_fnwp = os.path.join(result_dir, fn_lv_normal_mean_std)
    if not os.path.exists(mean_std_result_fnwp):
        l_input_fnwp.append(result_fnwp)
        l_output_fnwp.append(mean_std_result_fnwp)
    for idx, stock in enumerate(may_abnormal_stocks):
        stock_result_fnwp=l_stock_result_fnwp[idx]
        stock_mean_std_result_fnwp = os.path.join(result_dir, "{0}_fn_lv_normal_mean_std".format(stock))
        if not os.path.exists(stock_mean_std_result_fnwp):
            l_input_fnwp.append(stock_result_fnwp)
            l_output_fnwp.append(stock_mean_std_result_fnwp)
    for inpout_file, output_file in zip(l_input_fnwp,l_output_fnwp):
        l_result=pickle.load(open(inpout_file, 'r'))
        l_mean_std=[]
        for idx in range(17):
            temp=np.array(l_result[idx])
            l_mean_std.append([temp.mean(),temp.std()])
        df=pd.DataFrame(l_mean_std, columns=["mean","std"])
        df.to_csv(output_file, index=False)
        print("result stored in {0}".format(output_file))


def compare_mean_plot(argv):
    assert len(argv)==2
    system_name=argv[0]
    stock = argv[1]
    result_dir=os.path.join(RL_base_dir,system_name,Ana_sub_dir)
    if not os.path.exists(result_dir): os.makedirs(result_dir)


    mean_std_result_fnwp = os.path.join(result_dir, fn_lv_normal_mean_std)
    stock_mean_std_result_fnwp = os.path.join(result_dir, "{0}_fn_lv_normal_mean_std".format(stock))

    dfn = pd.read_csv(mean_std_result_fnwp)
    dfs = pd.read_csv(stock_mean_std_result_fnwp)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(dfn["mean"], label="normal_lv_mean")
    ax.plot(dfn["mean"]+dfn["std"], dashes=[6, 2])
    ax.plot(dfn["mean"]-dfn["std"], dashes=[6, 2])
    ax.plot(dfs["mean"],label="{0}_lv_mean".format(stock))
    x_tick = list(range(17))
    x_label = cn_lv
    ax.set_xticks(x_tick)
    ax.set_xticklabels(x_label, fontsize=12)

    for tick in ax.get_xticklabels():
        tick.set_rotation(-90)

    ax.set_title("normal lv mean vs {0} lv mean".format(stock), fontdict={"fontsize": 10}, pad=2)
    ax.legend()
    plt.show()


def plot_state_get_y(plotchoice,inputs):
    if plotchoice=="holding":
        row, stock, sbs = inputs
        state, a, r, state_=sbs.load_data_from_record_variables(row["fn"],stock, str(row["date"]))
        return np.argmax(np.array(state[2][0, :6]))
    elif plotchoice=="action":
        row, stock, sbs = inputs
        state, a, r, state_=sbs.load_data_from_record_variables(row["fn"],stock, str(row["date"]))
        return np.argmax(a)
    elif plotchoice=="reward":
        row, stock, sbs = inputs
        state, a, r, state_=sbs.load_data_from_record_variables(row["fn"],stock, str(row["date"]))
        return r[0]
    elif plotchoice =="av_price":
        row, stock, sbs = inputs
        state, a, r, state_=sbs.load_data_from_record_variables(row["fn"],stock, str(row["date"]))
        return state[2][0, -1]
    elif plotchoice =="potential_profit":
        row, stock, sbs = inputs
        state, a, r, state_=sbs.load_data_from_record_variables(row["fn"],stock, str(row["date"]))
        return state[2][0, -2]
    elif plotchoice =="advent":
        row, _, _ = inputs
        return row["advent"]
    else:
        raise ValueError("not support {0}".format(plotchoice))

def plot_state_item(argv):
    assert len(argv)==5, " system_name stock count threadhold plotchoice"
    system_name = argv[0]
    stock = argv[1]
    train_count = int(argv[2])
    threadhold=int(argv[3])
    plotchoice=argv[4]

    #plotchoice "holding" "action", "av price","advent"

    df=read_advent([system_name])
    dff = df[(df["train_count"] == train_count) & (df["stock"] == stock)]
    sbs = step_by_step(system_name)
    l_Y=[]
    x_tick_label=[]
    for idx, row in dff.iterrows():
        #state, a, r, state_=sbs.load_data_from_record_variables(row["fn"],stock, str(row["date"]))

        #data=plot_state_get_y([state, a, r, state_], plotchoice)
        data = plot_state_get_y(plotchoice,[row, stock, sbs])
        l_Y.append(data)
        x_tick_label.append(str(row["date"]))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    x=list(range(len(l_Y)))
    ax.plot(x, l_Y, label=plotchoice)
    ax.set_xticks(x)
    ax.set_xticklabels(x_tick_label, fontsize=8)

    for tick in ax.get_xticklabels():
        tick.set_rotation(-90)

    ax.set_title("{0} {1}".format(stock,plotchoice), fontdict={"fontsize": 10}, pad=2)
    ax.legend()


    np_wrong_date = np.unique(dff[dff["advent"]>=threadhold]["date"].values)
    np_wrong_date.sort()

    sx=[]
    sy=[]

    for idx, Y in enumerate(l_Y):
        if int(x_tick_label[idx]) in np_wrong_date:
            sx.append(idx)
            sy.append(Y)
    ax.scatter(sx, sy, color='r')

    plt.show()

###################################################################################################
#Step by step test
###################################################################################################
######Step by step test common
class step_by_step:
    def __init__(self, system_name):
        config_fnwp=os.path.join("/home/rdchujf/n_workspace/RL", system_name,"config.json")
        if not os.path.exists(config_fnwp):
            raise ValueError("{0} does not exisit".format(config_fnwp))
        self.lgc = sc.gconfig()
        self.lgc.read_from_json(config_fnwp)


        self.lc = lconfig()
        for key in list(self.lc.__dict__.keys()):
            self.lc.__dict__[key] = self.lgc.__dict__[key]
        init_agent_config(self.lc)
        init_trainer_config(self.lc)
        Ctrainer = globals()[self.lc.CLN_trainer]
        Cagent = globals()[self.lc.system_type]
        self.trainer = Ctrainer()

    def load_model(self, train_count ):
        model_config_fnwp = os.path.join(self.lgc.brain_model_dir, "config.json")
        lAIO=[fn for fn in os.listdir(self.lgc.brain_model_dir)
              if self.lgc.actor_model_AIO_fn_seed in fn and "T{0}.h5py".format(train_count) in fn]
        assert len(lAIO)==1
        AIO_fnwp=os.path.join(self.lgc.brain_model_dir, lAIO[0])
        Tmodel, Pmodel=self.trainer.load_train_model([AIO_fnwp,model_config_fnwp,""])
        return Tmodel, Pmodel
    def T_predict(self, State, State_, a, r, Pmodel, Tmodel):
        s_lv, s_sv, s_av, l_support_view = State
        s__lv, s__sv, s__av, l_support_view_= State_

        n_s_lv = np.expand_dims(s_lv[0], axis=0)
        n_s_sv = np.expand_dims(s_sv[0], axis=0)
        n_s_av = np.expand_dims(s_av[0], axis=0)

        n_a = np.expand_dims(a[0], axis=0)
        n_r = np.expand_dims(r[0], axis=0)
        n_s__lv = np.expand_dims(s__lv[0], axis=0)
        n_s__sv = np.expand_dims(s__sv[0], axis=0)
        n_s__av = np.expand_dims(s__av[0], axis=0)

        #n_s_lv = np.vstack(s_lv)
        #n_s_sv = np.vstack(s_sv)
        #n_s_av = np.vstack(s_av)
        #n_a = np.vstack(a)
        #n_r = np.vstack(r)
        #n_s__lv = np.vstack(s__lv)
        #n_s__sv = np.vstack(s__sv)
        #n_s__av = np.vstack(s__av)
        #num_record_to_train = len(n_s_lv)
        #assert num_record_to_train == self.lgc.batch_size

        _, v = Pmodel.predict({'P_input_lv': n_s__lv, 'P_input_sv': n_s__sv, 'P_input_av': n_s__av})

        r = n_r + self.lgc.Brain_gamma ** self.lgc.TDn * v

        y_pred = Tmodel.predict_on_batch({'input_l_view': n_s_lv, 'input_s_view': n_s_sv, 'input_account': n_s_av,
                                          'input_action': n_a, 'input_reward': r})

        prob, input_a, advents = y_pred[:, :3], y_pred[:, 3:6], y_pred[:, -1:]

        return prob, input_a, advents


    def load_data_from_T5(self, stock, date_s):
        if not hasattr(self,"i_data"):
            #self.i_data = R_T5("T5", stock)
            self.i_data = globals()[self.lgc.CLN_env_read_data]("T5", stock)
        else:
            try:
                if self.i_data.stock!=stock:
                    #self.i_data=R_T5("T5",stock )
                    self.i_data = globals()[self.lgc.CLN_env_read_data]("T5", stock)
            except AttributeError:  #NameError:
                #self.i_data = R_T5("T5", stock)
                self.i_data = globals()[self.lgc.CLN_env_read_data]("T5", stock)
        try:
            [lv,sv], support_view=self.i_data.read_one_day_data(date_s)

        except:
            lv, sv, support_view = None, None, None
        return lv, sv, support_view

    def load_data_from_record_variables(self, fn, stock, date_s):
        from recorder import record_variable
        try:
            a1, a2, a3 = self.rv.loader(fn)
        except:
            #self.rv = record_variable(self.lgc.record_variable_dir)
            self.rv = record_variable(self.lgc)
            a1, a2, a3 = self.rv.loader(fn)

        # flag_got, s_lv, s_sv, s_av, a, r, s__lv, s__sv, s__av, done_flag, l_support_view = a1
        s_lv, s_sv, s_av, a, r, s__lv, s__sv, s__av, done_flag, l_support_view = a1
        for idx, support_view in enumerate(l_support_view):
            if (support_view[0][0]["stock"]==stock) and (support_view[0][0]["date"]==date_s):
                return [s_lv[idx], s_sv[idx], s_av[idx],support_view ], a[idx], r[idx], [s__lv[idx], s__sv[idx], s__av[idx],None]
        else:
            return None, None, None, None

    #how to implement av
    def fabricate_av(self, inputs, holding, potential_profit):
        #holding = 0,1,2,3,4,5
        lv, sv, support_view=inputs
        assert self.lgc.data_name == "T5"
        if not self.lgc.flag_multi_buy:
            av=np.array([1.0 if holding > 0.0 else 0.0, potential_profit, support_view["stock_SwhV1"]]).reshape(1, -1)
        else:
            idx = int(holding)
            l_av = [0 for _ in range(self.lgc.times_to_buy + 1)]
            l_av[idx] = 1
            l_av.append(potential_profit)
            l_av.append(support_view["stock_SwhV1"])
            av=np.array(l_av).reshape(1, -1)
        return [lv,sv,av,support_view]

######Step by step test procedures
def Predict_on_recorded_data(argv):

    #load data from rv and predict , compare with debug_result
    system_name="htryc"
    stock="SH600177"
    date_s="20170731"
    fn="2019-04-07_08:03:30_584.pkl"
    train_count=100

    sbs=step_by_step(system_name)
    Tmodel, Pmodel= sbs.load_model(train_count)
    state, a, r, state_=sbs.load_data_from_record_variables(fn,stock, date_s)
    prob, input_a, advents=sbs.T_predict(state, state_, a, r, Pmodel, Tmodel)

    print(prob, input_a, advents, "should advent is {0}".format(9694.671875))

def Predict_on_T5_data(argv):
    system_name="htryc"
    stock="SH600177"

    #date_s="20170731"
    date_s = "20170608"
    holding =0
    potential_profit=0
    holding_ = 0
    potential_profit_ =0
    a=2
    r=0

    td = API_trade_date().np_date_s
    train_count=100

    sbs = step_by_step(system_name)
    Tmodel, Pmodel = sbs.load_model(train_count)


    lv, sv, support_view=sbs.load_data_from_T5(stock, date_s)
    if lv is None:
        print("Can not get {0} {1} data".format(stock, date_s))
        return False
    else:
        state=sbs.fabricate_av([lv, sv, support_view], holding, potential_profit)


    period=td[td>=date_s]
    print(period[4])

    lv_, sv_, support_view_=sbs.load_data_from_T5(stock, period[4])
    if lv_ is None:
        print("Can not get {0} {1} data".format(stock, period[4]))
        return False
    else:
        state_ = sbs.fabricate_av([lv_, sv_, support_view_], holding_, potential_profit_)

    a_onehot = np.zeros([1, 3])
    a_onehot[0, a] = 1

    prob, input_a, advents=sbs.T_predict(state, state_, a_onehot, np.expand_dims(np.array([r]), axis=0), Pmodel, Tmodel)
    #print prob, input_a, advents, "should advent is {0}".format(9694.671875)
    print(prob, input_a, advents, "should advent is {0}".format(1452.085083))

######Step by step test plots
def view_record_data_AV(argv):
    system_name="htryc"
    stock="SH600177"
    date_s="20170731"
    fn="2019-04-07_08:03:30_584.pkl"

    sbs=step_by_step(system_name)
    state, a, r, state_=sbs.load_data_from_record_variables(fn,stock, date_s)

    print(state[2])
    print(state_[2])

class C_view_lv:
    def __init__(self, argv):
        system_name = "htryc"
        stock="SH600177"
        self.iter=self.iter_row(system_name,stock)

    def iter_row(self,system_name,stock ):
        system_dir = os.path.join("/home/rdchujf/n_workspace/RL/", system_name)
        fnwp = os.path.join(system_dir, "{0}_date.csv".format(stock))
        if not os.path.exists(fnwp):
            print("run python Ana_impluse_loss.py debug7 ")
            assert False
        df = pd.read_csv(fnwp)
        df["date"] = df["date"].astype(str)
        sbs = step_by_step(system_name)
        for idx, row in df.iterrows():
            fn = row["fn"]
            date_s = row["date"]
            state, _, _, _ = sbs.load_data_from_record_variables(fn, stock, date_s)
            dflv = pd.DataFrame(state[0].reshape(20, -1), columns=cn_lv)
            for col in dflv.columns:
                dflv[col] = dflv[col].map(('{:,.2f}'.format))
            yield dflv
        yield None


    def pretty_table_print(self, df):
        pt = PrettyTable()
        for column_name in  df.columns:
            pt.add_column(column_name, df[column_name])
        print(pt)

    def get_keys(self):
        while True:
            command = input("n for next; q for quit")
            if len(command)==1 and command in ["n","q"]:
                break
            else:
                print("invalid input {0}".format(command))
        return command

    def run(self):
        while True:
            dflv=next(self.iter)
            if dflv is None:
                break
            self.pretty_table_print(dflv)
            command=self.get_keys()
            if command=="q":
                break


if __name__ == '__main__':
    if sys.argv[1].startswith("C_"):
        globals()[sys.argv[1]](sys.argv[2:]).run()
    else:
        globals()[sys.argv[1]](sys.argv[2:])
