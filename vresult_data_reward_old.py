import os,re,collections
import pandas as pd
import numpy as np
from data_common import API_trade_date,API_HFQ_from_file,hfq_toolbox,API_qz_data_source_related
from vcomm import img_tool
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import progressbar
import pickle
from env import env_reward_basic#, env_reward_10,env_reward_100,env_reward_1000,env_reward_1000_shift10
from vresult_data_com import get_data_start_end,are_esi_reader
import config as sc
def natural_keys(text):
    return [ int(c) if c.isdigit() else c for c in re.split(r'(\d+)',text) ]



#interfaces
# self.get_reward_on_date
# self.get_are_summary
# self.get_stock_hprice_data
# self.get_tranaction_price_data
class ana_reward_data(are_esi_reader):
    def __init__(self, system_name, process_name,Lstock, LEvalT, LYM,lgc):
        are_esi_reader.__init__(self,system_name, process_name)
        self.Lstock, self.LEvalT, self.LYM,self.lgc=Lstock, LEvalT, LYM, lgc
        self.data_start_s,self.data_end_s=get_data_start_end(self.lgc,process_name)
        self.td = API_trade_date().np_date_s

    def _get_are_summary_1stock_1ET(self, stock, evalT):
        flag_file_found,df = self._read_stock_are(stock, evalT)
        if not flag_file_found:
            return False, "",df  # in False situation df is a string message

        # remove not in trans summary
        idx_Not_in_trans = df["trans_id"] == "Not_in_trans"
        df = df[~idx_Not_in_trans]
        idx_pre_fail_buy = (df["action"] == 0) & (df["action_result"] != "Success")
        if any(idx_pre_fail_buy):
            df.loc[idx_pre_fail_buy, "action"] = 1
            df.loc[idx_pre_fail_buy, "action_result"] = "No_action"
        idx_pre_fail_sell = (df["action"] == 2) & (df["action_result"] != "Success")
        if any(idx_pre_fail_sell):
            df.loc[idx_pre_fail_sell, "action"] = 3
            df.loc[idx_pre_fail_sell, "action_result"] = "No_action"

        df = df[["action", "reward", "day", "trans_id", "action_result", "trade_Nprice"]]

        df = df.groupby(["trans_id", "action", "action_result"]).agg({"reward": {"reward": "sum"},
                                                                       "day": {"duration": "count",
                                                                               "trans_start": "first",
                                                                               "trans_end": "last"},
                                                                       "trade_Nprice": {
                                                                           "price": "mean"}})  # mean for price because (buy success) (sell sucess) only have one record

        df.columns = df.columns.droplevel(0)
        df.reset_index(inplace=True)
        df.loc[df.action == 3, 'action'] = 1
        dfr = df.groupby(["trans_id"]).agg({
            "trans_start": {"trans_start": "first"},
            "trans_end": {"trans_end": "last"},
            "reward": {"reward": "sum"},
            "duration": {"duration": "sum"},
            "price": {"buy_price": "first", "sell_price": "last"},
            "action": {"valid_trans_kpi": "mean"}})

        dfr.columns = dfr.columns.droplevel(0)
        # action senario  0,2 valid_trans_cpi=1
        # action senario  0,1,2 valid_trans_cpi=1
        # action senario  0,1,1 valid_trans_cpi=0.666  # this is unfinished trans becouse originally has 3 "no_action  and 3 "Success" (forcesell)
        dfrr = dfr[dfr["valid_trans_kpi"] == 1]
        dfrr.reset_index(inplace=True)
        if len(dfrr)==0:
            return False, "", "no_valid_transaction"
        dfrr=pd.DataFrame(dfrr)
        dfrr["buy_count"]=1
        dfrr["flag_trans_valid"] = True
        dfrr["stock"] = stock
        dfrr["EvalT"] = evalT

        '''
        if not flag_opt:
            # flag_skip=True
            np_CSR_sum[ETidx, Stockidx] = 0
            np_CSR_mean[ETidx, Stockidx] = 0
            np_CSR_median[ETidx, Stockidx] = 0
            np_CSR_std[ETidx, Stockidx] = 0
            np_CSR_count[ETidx, Stockidx] = 0
            # break
        else:
            np_CSR_sum[ETidx, Stockidx] = df["reward"].sum()
            np_CSR_mean[ETidx, Stockidx] = df["reward"].mean()
            np_CSR_median[ETidx, Stockidx] = df["reward"].median()
            np_CSR_std[ETidx, Stockidx] = df["reward"].std()
            np_CSR_count[ETidx, Stockidx] = df["reward"].count()
        '''
        Dic_effective = {"effective_buy": len(dfr), "effective_sell": len(dfrr), "effective_trans": len(dfrr)}
        return True, dfrr, Dic_effective

    def _get_are_summary_1ET(self, evalT):
        des_dir=self.dir_analysis
        for sub_dir in [self.process_name,"per_ET"]:
            des_dir =os.path.join(des_dir,sub_dir)
            if not os.path.exists(des_dir): os.mkdir(des_dir)
        fnwp=os.path.join(des_dir, "ET{0}.csv".format(evalT))
        effective_count_fnwp=os.path.join(des_dir, "ET{0}_effective_count.csv".format(evalT))
        #fnwp = os.path.join(self.dir_analysis, "per_ET", "ET{0}.csv".format(evalT))
        if os.path.exists(fnwp) and os.path.exists(effective_count_fnwp):
            try:
                df=pd.read_csv(fnwp)
                dfe=pd.read_csv(effective_count_fnwp)
                assert len(dfe)==1
                Summery_effective_count={}
                Summery_effective_count["effective_buy"] =dfe.iloc[0]["effective_buy"]
                Summery_effective_count["effective_sell"] =dfe.iloc[0]["effective_sell"]
                Summery_effective_count["effective_trans"] =dfe.iloc[0]["effective_trans"]
                Summery_effective_count["EvalT"] = dfe.iloc[0]["EvalT"]
                return True, df,Summery_effective_count
            except Exception:
                print "{0} empty file".format(fnwp)
                return False, "","No_valid_transaction"
        df=pd.DataFrame()
        bar = progressbar.ProgressBar(maxval=len(self.Lstock),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        flag_file_not_fount = False
        Summery_effective_count={"EvalT":evalT}
        for stock_idx, stock in enumerate(self.Lstock):
            if stock_idx==0:
                print "preparing ", evalT
                flag_opt,df,Dic_effective=self._get_are_summary_1stock_1ET(stock, evalT)

                if not flag_opt:
                    if Dic_effective in ["File_Not_Found","File_is_empty"]:
                        flag_file_not_fount=True
                        break
                    else:
                        assert Dic_effective=="no_valid_transaction"
                        df=pd.DataFrame()
                        continue
                else:
                    Summery_effective_count["effective_buy"] = Dic_effective["effective_buy"]
                    Summery_effective_count["effective_sell"] = Dic_effective["effective_sell"]
                    Summery_effective_count["effective_trans"] = Dic_effective["effective_trans"]

            else:
                flag_opt,dfi,Dic_effective=self._get_are_summary_1stock_1ET(stock, evalT)
                if not flag_opt:
                    if Dic_effective in ["File_Not_Found","File_is_empty"]:
                        flag_file_not_fount=True
                        break
                    else:
                        assert Dic_effective=="no_valid_transaction"
                        continue
                else:
                    df=df.append(dfi, ignore_index=True)
                    Summery_effective_count["effective_buy"]+=Dic_effective["effective_buy"]
                    Summery_effective_count["effective_sell"]+=Dic_effective["effective_sell"]
                    Summery_effective_count["effective_trans"]+=Dic_effective["effective_trans"]
            bar.update(stock_idx)
        bar.finish()
        if flag_file_not_fount:
            return False, "","File_Not_Found"
        else:
            df.to_csv(fnwp, index=False)
            pd.DataFrame([Summery_effective_count.values()], columns=Summery_effective_count.keys()).\
                to_csv(effective_count_fnwp, index=False)
            if len(df)==0:
                return False, "", "No_valid_transaction"
            return True,df,Summery_effective_count

    def get_are_summary(self):
        df = pd.DataFrame()
        dfse=pd.DataFrame()
        for ET_idx, EvalT in enumerate(self.LEvalT):
            #print "EvalT",EvalT
            if ET_idx==0:
                flag_opt,df,Summery_effective_count=self._get_are_summary_1ET(EvalT)
                if not flag_opt:
                    if Summery_effective_count=="File_Not_Found":
                        #return False, "File_Not_Found",""
                        assert False,"File_Not_Found"
                    else:
                        assert Summery_effective_count=="No_valid_transaction"
                        df=pd.DataFrame()
                        dfse = pd.DataFrame()
                        continue
                else:
                    dfse = pd.DataFrame([Summery_effective_count.values()], columns=Summery_effective_count.keys())
            else:
                flag_opt,dfi,Summery_effective_count=self._get_are_summary_1ET(EvalT)
                if not flag_opt:
                    if Summery_effective_count=="File_Not_Found":
                        #return False, "File_Not_Found",""
                        #assert False, "File_Not_Found"
                        dfse.reset_index(inplace=True)
                        return df, dfse
                    else:
                        assert Summery_effective_count=="No_valid_transaction"
                        continue
                else:
                    df=df.append(dfi, ignore_index=True)
                    dfse = dfse.append(pd.DataFrame([Summery_effective_count.values()], columns=Summery_effective_count.keys()),ignore_index=True)
        dfse.reset_index(inplace=True)
        return df, dfse

    #def get_tranaction_price_data(self, stock, evalT ):
    #    #dfare = self._read_stock_are(stock, evalT)
    #    df,_ = self.get_verified_are_data(stock, evalT)
    #    return df

    def get_stock_hprice_data(self, stock):
        def priceH2N(row_name):
            def convert_row(row):
                return row[row_name] / row["coefficient_fq"]
            return convert_row

        dfh=API_HFQ_from_file().get_df_HFQ(stock)
        #dfh["date"] = dfh["date"].astype(str)
        #dfh = dfh[(dfh.date >= self.data_start_s) & (dfh.date <= self.data_end_s)]
        dfh["date"] = dfh["date"].astype(int)
        dfh = dfh[(dfh.date >= int(self.data_start_s)) & (dfh.date <= int(self.data_end_s))]

        dfh["NpriceHighest"] = dfh.apply(priceH2N("highest_price"), axis=1)
        dfh["NpriceLowest"] = dfh.apply(priceH2N("lowest_price"), axis=1)
        dfh["NpriceOpen"] = dfh.apply(priceH2N("open_price"), axis=1)
        dfh=dfh[["date","NpriceHighest","NpriceLowest", "NpriceOpen"]]
        dfh.reset_index(inplace=True)

        td=self.td[(self.td>=self.data_start_s)&(self.td<=self.data_end_s)]
        dftd = pd.DataFrame(data=td, columns=["date"])
        dftd["date"]=dftd["date"].astype(int)
        dfHprice = pd.merge(dftd, dfh, how="left", left_on="date", right_on="date")
        dfHprice.drop(columns="index", inplace=True)
        dfHprice = dfHprice.ffill().bfill()

        return dfHprice

    def get_ESRs(self): #ESR ET Stock Reward
        fn_ESR_seed="{0}_ESR__{1}.pkl"
        #ana_dir=os.path.join(sc.base_dir_RL_system, self.system_name, "analysis")
        des_dir=self.dir_analysis
        for sub_dir in [self.process_name]:
            des_dir =os.path.join(des_dir,sub_dir)
            if not os.path.exists(des_dir): os.mkdir(des_dir)

        row=len(self.LEvalT)
        col=len(self.Lstock)
        print row,col
        fnwp = os.path.join(des_dir,fn_ESR_seed.format(self.process_name, row))
        if os.path.exists(fnwp):
            with open(fnwp, 'rb') as f:
                np_CSR_sum, np_CSR_mean, np_CSR_median, np_CSR_std, np_CSR_count  = pickle.load(f)
            return np_CSR_sum, np_CSR_mean, np_CSR_median, np_CSR_std, np_CSR_count

        np_CSR_sum, np_CSR_mean, np_CSR_median, np_CSR_std, np_CSR_count=\
            np.zeros((row, col)),np.zeros((row, col)),np.zeros((row, col)),np.zeros((row, col)),np.zeros((row, col))

        lfn=[fn for fn in os.listdir(des_dir) if ("{0}_ESR__".format(self.process_name) in fn) and fn.endswith("pkl")]
        if len(lfn)!=0:
            lfn.sort(key=natural_keys)
            ETskipidx=int (re.split(r'(\d+)', lfn[-1])[-2])  #re.split('(\d+)', "Eval_0_ESR__1000.csv")['Eval_', '0', '_ESR__', '1000', '.csv']

            with open(os.path.join(des_dir, lfn[-1]), 'rb') as f:
                pre_np_CSR_sum, pre_np_CSR_mean, pre_np_CSR_median, pre_np_CSR_std, pre_np_CSR_count = pickle.load(f)

            print lfn[-1]
            print ETskipidx
            print pre_np_CSR_sum.shape
            print np_CSR_sum.shape
            for idx in range(ETskipidx):
                np_CSR_sum[idx, :]      = pre_np_CSR_sum[idx, :]
                np_CSR_mean[idx, :]     = pre_np_CSR_mean[idx, :]
                np_CSR_median[idx, :]   = pre_np_CSR_median[idx, :]
                np_CSR_std[idx, :]      = pre_np_CSR_std[idx, :]
                np_CSR_count[idx, :]    = pre_np_CSR_count[idx, :]
            for fn in lfn:
                os.remove(os.path.join(des_dir,fn))
        else:
            ETskipidx=-1
        for ETidx,ET in enumerate(self.LEvalT):
            if ETidx <ETskipidx:
                continue
            print "handling ET ", ET
            flag_skip=False
            for Stockidx, stock in enumerate(self.Lstock):
                #df,_ = self.get_verified_are_data(stock, ET)
                flag_opt, df,_=self._get_are_summary_1stock_1ET(stock, ET)
                if not flag_opt:
                    #flag_skip=True
                    np_CSR_sum[ETidx,Stockidx]      = 0
                    np_CSR_mean[ETidx,Stockidx]     = 0
                    np_CSR_median[ETidx,Stockidx]   = 0
                    np_CSR_std[ETidx,Stockidx]      = 0
                    np_CSR_count[ETidx,Stockidx]    = 0
                    #break
                else:
                    np_CSR_sum[ETidx,Stockidx]      = df["reward"].sum()
                    np_CSR_mean[ETidx,Stockidx]     = df["reward"].mean()
                    np_CSR_median[ETidx,Stockidx]   = df["reward"].median()
                    np_CSR_std[ETidx,Stockidx]      = df["reward"].std()
                    np_CSR_count[ETidx,Stockidx]    = df["reward"].count()
            '''
            if flag_skip:
                np_CSR_sum      =np_CSR_sum[:ETidx,:]
                np_CSR_mean     =np_CSR_mean[:ETidx,:]
                np_CSR_median   =np_CSR_median[:ETidx,:]
                np_CSR_std      =np_CSR_std[:ETidx,:]
                np_CSR_count    =np_CSR_count[:ETidx,:]
                print ETidx
                fnwp = os.path.join(des_dir,fn_ESR_seed.format(self.process_name, ETidx-1))
            '''
        with open(fnwp, 'wb') as f:
            pickle.dump([np_CSR_sum, np_CSR_mean, np_CSR_median, np_CSR_std, np_CSR_count], f)

        return np_CSR_sum, np_CSR_mean, np_CSR_median, np_CSR_std, np_CSR_count


    def get_TransDensity_RewardDistribution(self, stock, evalT):
        start_year_i=int(self.data_start_s[:4])
        end_year_i=int(self.data_end_s[:4])
        #row=self.total_year_i*4
        row = (end_year_i-start_year_i+1) * 4
        col=93
        unit=10
        img_density = np.zeros((row, col))
        img_reward  = np.zeros((row, col))
        flag_opt, df, _ = self._get_are_summary_1stock_1ET(stock, evalT)
        if not flag_opt:
            return img_density,img_reward

        for _, row in df.iterrows():
            period= self.td[(self.td>=str(row.trans_start)) &(self.td<=str(row.trans_end))]
            for day in period:
                date_i = int(day)
                date_year = date_i / 10000
                date_month = (date_i / 100) % 100
                date_day = date_i % 100
                np_row = (date_year - start_year_i) * 4 + (date_month - 1) / 3
                np_column = (date_month - 1) % 3 * 31 + (date_day - 1)
                img_density[np_row, np_column] += unit
                img_reward[np_row, np_column]  += row["reward"]
        return img_density,img_reward

class ana_reward_plot:
    def __init__(self, system_name, process_name,Lstock, LEvalT, LYM,lgc):
        self.process_name   =   process_name
        self.system_name    =   system_name
        self.i_ana_data=ana_reward_data( self.system_name, self.process_name,Lstock, LEvalT, LYM,lgc)
        self.df,self.dfse=self.i_ana_data.get_are_summary()

        #self.Lstock,self.LEvalT,self.LYM=self.i_ana_data.get_init_data()
        self.Lstock, self.LEvalT, self.LYM, self.lgc = Lstock, LEvalT, LYM,lgc
        self.Cidx_stock,self.Cidx_EvalT, self.Cidx_LYM=0,0,0

    def check_data_ready(self, attr_name):
        if hasattr(self,attr_name):
            return True
        else:
            if attr_name=="df_bs":
                flag_opt, df, _ = self.i_ana_data._get_are_summary_1stock_1ET(self.Lstock[self.Cidx_stock],
                                                                       self.LEvalT[self.Cidx_EvalT])
                self.df_bs=df if flag_opt else pd.DataFrame()

            elif attr_name=="df_price":
                self.df_price = self.i_ana_data.get_stock_hprice_data(self.Lstock[self.Cidx_stock])
            elif attr_name in ["np_CSR_sum", "np_CSR_mean","np_CSR_median", "np_CSR_std", "np_CSR_count"]:
                self.np_CSR_sum, self.np_CSR_mean, self.np_CSR_median, self.np_CSR_std, self.np_CSR_count = self.i_ana_data.get_ESRs()
            elif attr_name in ["np_trans_density", "np_reward_distribution"]:
                self.np_trans_density, self.np_reward_distribution = self.i_ana_data. \
                    get_TransDensity_RewardDistribution(self.Lstock[self.Cidx_stock], self.LEvalT[self.Cidx_EvalT])
            else:
                assert False, "{0} not supported".format(attr_name)
            return True

    def plot_effective_count(self,ax):
        ax.clear()
        for item in ["effective_buy","effective_sell","effective_trans"]:
            ax.plot(self.dfse.index, self.dfse[item],label=item)

        ax.legend(loc='upper right')

    def scatter_transaction(self, ax,ET1, ET2, reward_threadhold):
        ax.clear()
        df1, dicts1=self.i_ana_data._get_are_summary_1ET(ET1)
        df1=df1[df1["reward"]<=reward_threadhold]
        df1["stock_idx"]=df1["stock"]
        df1["stock_idx"].apply(lambda x: self.Lstock.index(x))
        df1["trans_start"]=df1["trans_start"]

        df2, dicts2=self.i_ana_data._get_are_summary_1ET(ET2)
        df2=df2[df2["reward"]<=reward_threadhold]


    def plot_reward_count(self, ax,EvalT):
        df=self.df
        dfr=df[["EvalT","reward"]].groupby(["EvalT"]).sum()
        dfr.rename(columns={"reward":"reward_sum"}, inplace=True)
        dfr.reset_index(inplace=True)

        ax.clear()
        ax.set_title("accumulate reward v.s. total number of transaction")
        ax.plot(dfr.index,dfr["reward_sum"],label="reward_sum")

        ax.legend(loc='upper left')
        ax.tick_params(axis='x', rotation=90)
        ax.set_xticks(range(len(self.LEvalT)+1))
        ax.set_xticklabels(self.LEvalT, fontsize=7)
        ax.set_ylabel("reward", fontsize=10)

        ax2 = ax.twinx()
        dfr = df[["EvalT", "trans_id"]].groupby(["EvalT"]).count()
        dfr.rename(columns={"trans_id": "count"}, inplace=True)
        dfr.reset_index(inplace=True)
        ax2.plot(dfr.index, dfr["count"],label="number of transaction", color="r")
        ax2.set_ylabel("num of transaction(sell)", fontsize=10)
        ax2.legend(loc='upper right')

        ymin,ymax=ax.get_ylim()
        stc = EvalT / self.i_ana_data.lgc.num_train_to_save_model - 1
        ax.plot([stc,stc],[ymin,ymax])

    def hist_reward(self, ax, EvalT,hist_param):
        if len(hist_param)==0:
            #min_, max_, show_step=globals()[self.i_ana_data.lgc.CLN_env_reward_eval]().hist_scale()
            min_, max_, show_step = env_reward_basic(self.i_ana_data.lgc.eval_reward_scaler_factor,self.i_ana_data.lgc.eval_reward_type).hist_scale()
        else:
            max_, min_, show_step=hist_param
            print max_, min_, show_step
        df = self.df
        dfr = df[df["EvalT"] == EvalT]

        ax.clear()
        ax.set_title("reward hist @ Trains count {0} ".format(EvalT))
        ax.hist(dfr["reward"],bins=np.arange(min_, max_, show_step))
        ax.set_ylabel("count", fontsize=10)
        ax.set_xlabel("reward", fontsize=10)

        l_exceed_showscope = dfr[(dfr["reward"] < min_) | (dfr["reward"] > max_)].reward.tolist()
        if len(l_exceed_showscope)!=0:
            l_exceed_showscope_count = [[item, count] for item, count in collections.Counter(l_exceed_showscope).items()]
            dfw = pd.DataFrame(l_exceed_showscope_count, columns=["reward", "TC"])
            dfw.sort_values(by=["reward"], inplace=True)
            dfw.reset_index(inplace=True,drop=True)
            negative_str=""
            positive_str=""
            for _, row in dfw.iterrows():
                if row.reward>0:
                    positive_str = "{0} {1:.2f}: {2} \n".format(positive_str, row.reward, int(row.TC))
                else:
                    negative_str = "{0} {1:.2f}: {2} \n".format(negative_str, row.reward, int(row.TC))
            ax.text(0.08, 0.001, negative_str,verticalalignment='bottom', horizontalalignment='right',
                    transform=ax.transAxes,color='green', fontsize=8)
            ax.text(0.99, 0.001, positive_str,verticalalignment='bottom', horizontalalignment='right',
                    transform=ax.transAxes, color='green', fontsize=8)

    def scatter_buy_time_vs_reward(self,ax, EvalT):
        df = self.df
        dfr = df[df["EvalT"] == EvalT]
        ax.clear()
        ax.set_title("reward vs buy_time @ Trains count {0} ".format(EvalT))
        ax.scatter(dfr.reward, dfr.buy_count)
        ax.set_yticks(range(6))
        ax.set_yticklabels(range(6), fontsize=10)
        ax.set_ylabel("buy_time", fontsize=10)
        ax.set_xlabel("reward", fontsize=10)

    def scatter_duration_vs_reward(self,ax, EvalT,rd_param):
        df = self.df
        dfr = df[df["EvalT"] == EvalT]
        if len(dfr)!=0:
            dfg=dfr[["reward","duration","stock"]].groupby(["duration","reward"]).count()
            dfg.rename(columns={"stock":"t_count"}, inplace=True)
            dfg.reset_index(inplace=True)
            if len(rd_param)!=0:
                max_reward_show, min_reward_show,max_duration_show =rd_param
                dff=dfg[(dfg["reward"]>=min_reward_show) &(dfg["reward"]<=max_reward_show) ]
                dff =dff[dff["duration"]<=max_duration_show]
            else:
                dff=dfg
            ax.clear()
            ax.set_title("reward vs duration @ Trains count {0} ".format(EvalT))
            ax.scatter(dff.reward, dff.duration)
            ax.set_ylabel("duration", fontsize=10)
            ax.set_xlabel("reward", fontsize=10)

        else:
            ax.clear()

    def Month_P1NX_FD(self, SYM, nx):
        #SYM="201401"
        IM=int(SYM[-2:])
        IY=int(SYM[:-2])
        IMP = 12 if IM == 1 else IM - 1
        IYP = IY - 1 if IM == 1 else IY
        SYMP_FD= "{0:04d}{1:02d}01".format(IYP,IMP)
        IMN2= (IM+nx)%12
        IYN2= IY +1 if (IM+nx)/12==1 else IY
        SYMN2_FD = "{0:04d}{1:02d}01".format(IYN2,IMN2)
        return SYMP_FD, SYMN2_FD


    def plot_price_buy_sell(self, ax,SYM, df_bs, df_price):
        #SYM=self.LYM[self.Cidx_LYM]

        SYMP1_FD, SYMN2_FD = self.Month_P1NX_FD(SYM,3)

        df_trans=df_bs[(df_bs.trans_start >= int(SYMP1_FD)) & (df_bs.trans_end <= int(SYMN2_FD))]
        dfp = df_price[(df_price.date >= int(SYMP1_FD)) & (df_price.date <= int(SYMN2_FD))]
        dfp.reset_index(inplace=True)

        ax.clear()
        ax.set_title("train saction detail from {0} to {1}".format(SYMP1_FD, SYMN2_FD))
        ax.plot(dfp.index.tolist(), dfp.NpriceHighest,label="High")
        ax.plot(dfp.index.tolist(), dfp.NpriceLowest,label="low")
        for _,row in df_trans.iterrows():
            lindex=dfp[dfp["date"] == row.trans_start].index.tolist()
            if len(lindex)!=1:
                print "skip", row
                continue
            trans_start_index=lindex[0]
            lindex=dfp[dfp["date"] == row.trans_end].index.tolist()
            if len(lindex)!=1:
                print "skip", row
                continue
            trans_end_index=lindex[0]
            ax.plot([trans_start_index,trans_end_index],[row.buy_price, row.sell_price])
        ax.legend()
        ax.tick_params(axis='x', rotation=90)
        ax.set_xticks(dfp.index.tolist())
        ax.set_xticklabels(dfp.date.tolist(), fontsize=7)


    def img_reward_date(self, fig, ax,img_data,start_year_i, end_year_i,Dimgc):
        if not self.check_data_ready("np_trans_density"): assert False
        divider3 = make_axes_locatable(ax)
        cax = divider3.append_axes("right", size="1%", pad=0.05)
        ax.clear()
        a = np.array(img_data, copy=True)
        if Dimgc[0]=="Origin":
            pass
        elif Dimgc[0]=="Greater":
            a[a < Dimgc[1]] = img_data.min()
        else:
            assert Dimgc[0]=="Less"
            a[a > Dimgc[1]] = img_data.max()
        im = ax.imshow(a, norm=mpl.colors.Normalize(vmin=img_data.min(), vmax=img_data.max()))

        l_ytick_labals=[]
        for year_i in range(start_year_i, end_year_i+1):
            for quarter_i in range(4):
                l_ytick_labals.append("{0}Q{1}".format(year_i, quarter_i+1))

        ax.xaxis.set_ticks(np.arange(0, 93, 31))
        ax.set_xticklabels(["M1", "M2", "M3"])
        ax.set_yticks(range(20))
        ax.set_yticklabels(l_ytick_labals, fontsize=8)
        cax.tick_params(labelsize=8)

        fig.colorbar(im, cax=cax, format='%.0e')

    def show_stock_reward_on_all_ET(self, ax,stock,sub_choice):
        if not self.check_data_ready("np_CSR_sum"): assert False
        self.Cidx_stock = self.Lstock.index(stock)
        #ax = allaxes[3]
        ax.set_title("reward on {0} ALL ET".format(stock))
        ax.plot(self.np_CSR_sum[:,self.Cidx_stock],label="sum", color="b")
        ax.xaxis.set_ticks(range(len(self.LEvalT)))
        ax.set_xticklabels(self.LEvalT, fontsize=6)
        ax.plot([0,len(self.np_CSR_sum[:,self.Cidx_stock])],[0,0], color="y")
        ax.legend(loc='upper left')
        ax2 = ax.twinx()
        if sub_choice=="mean":
            ax2.plot(self.np_CSR_mean[:,self.Cidx_stock],label="mean", color="r")
        elif sub_choice=="median":
            ax2.plot(self.np_CSR_median[:,self.Cidx_stock],label="median", color="r")
        elif sub_choice=="count":
            ax2.plot(self.np_CSR_count[:,self.Cidx_stock],label="count", color="r")
        else:# show_choice=="std"
            ax2.plot(self.np_CSR_std[:,self.Cidx_stock],label="std", color="r")
        ax2.legend(loc='upper right')


    def show_ET_reward_on_all_stock(self, ax, EvalT, sub_choice):
        if not self.check_data_ready("np_CSR_sum"): assert False
        self.Cidx_EvalT = self.LEvalT.index(EvalT)
        ax.set_title("reward on ET{0}  all stock".format(EvalT))
        ax.plot(self.np_CSR_sum[self.Cidx_EvalT,:],label="sum", color="b")
        ax.xaxis.set_ticks(range(len(self.Lstock)))
        ax.set_xticklabels(self.Lstock, fontsize=6)
        ax.plot([0, len(self.np_CSR_sum[self.Cidx_EvalT,:])], [0, 0], color="y")
        for tick in ax.get_xticklabels():
            tick.set_rotation(-90)
        ax.legend(loc='upper left')
        ax2 = ax.twinx()
        if sub_choice=="mean":
            ax2.plot(self.np_CSR_mean[self.Cidx_EvalT,:],label="mean", color="r")
        elif sub_choice=="median":
            ax2.plot(self.np_CSR_median[self.Cidx_EvalT,:],label="median", color="r")
        elif sub_choice=="count":
            ax2.plot(self.np_CSR_count[self.Cidx_EvalT,:],label="count", color="r")
        else:# show_choice=="std"
            ax2.plot(self.np_CSR_std[self.Cidx_EvalT,:],label="std", color="r")
        ax2.legend(loc='upper right')





    def show_reward_all_ET_all_stock(self,ax,fig, Dimgc):
        if not self.check_data_ready("np_CSR_sum"): assert False
        divider3 = make_axes_locatable(ax)
        cax = divider3.append_axes("right", size="1%", pad=0.05)
        ax.clear()
        ax.set_title("reward on ET vs stock")
        a = np.array(self.np_CSR_sum, copy=True)
        if Dimgc[0]=="Origin":
            pass
        elif Dimgc[0]=="Greater":
            a[a < Dimgc[1]] = self.np_CSR_sum.min()
        else:
            assert Dimgc[0]=="Less"
            a[a > Dimgc[1]] = self.np_CSR_sum.max()
        #im=ax.imshow(a,norm=mpl.colors.Normalize(vmin=self.np_CSR_sum.min(), vmax=self.np_CSR_sum.max()))
        im = ax.imshow(a, norm=mpl.colors.Normalize(vmin=self.np_CSR_sum.min(), vmax=self.np_CSR_sum.max()))
        ax.xaxis.set_ticks(range(len(self.Lstock)))
        ax.set_xticklabels(self.Lstock, fontsize=6)
        ax.set_yticks(range(len(self.LEvalT)))
        ax.set_yticklabels(self.LEvalT, fontsize=8)
        cax.tick_params(labelsize=8)
        fig.colorbar(im, cax=cax, format='%.0e')
        #cbar.set_clim(self.np_CSR_sum.min(),self.np_CSR_sum.max())
        for tick in ax.get_xticklabels():
            tick.set_rotation(-90)



class ana_reward(ana_reward_plot):
    def show_reward(self,fig, EvalT,rd_param,hist_param):
        allaxes = fig.get_axes()
        for axe in allaxes:
            axe.remove()
        fig.add_subplot(231)
        fig.add_subplot(232)
        fig.add_subplot(233)
        fig.add_subplot(234)
        fig.add_subplot(235)
        fig.add_subplot(236)

        allaxes = fig.get_axes()
        fig.suptitle("Summary on/at {0} EvalT{1}".format(self.system_name,EvalT), fontsize=14)
        fig.subplots_adjust(bottom=0.05, top=0.9, left=0.03, right=0.97, wspace=0.1, hspace=0.3)
        self.plot_reward_count(allaxes[0],EvalT)
        self.scatter_duration_vs_reward(allaxes[1], EvalT,rd_param)
        self.plot_effective_count(allaxes[2])

        self.hist_reward(allaxes[3], EvalT,hist_param)
        self.scatter_buy_time_vs_reward(allaxes[4], EvalT)

    def show_reward_detail(self,fig, stock, EvalT, YM,Dimgc):
        if not (stock== self.Lstock[self.Cidx_stock] and EvalT==self.LEvalT[self.Cidx_EvalT]):
            self.Cidx_stock=self.Lstock.index(stock)
            self.Cidx_EvalT=self.LEvalT.index(EvalT)
            self.df_price = self.i_ana_data.get_stock_hprice_data(self.Lstock[self.Cidx_stock])
            self.np_trans_density, self.np_reward_distribution=self.i_ana_data.\
                get_TransDensity_RewardDistribution(stock, EvalT)
        else:
            if not self.check_data_ready("df_bs"): assert False
            if not self.check_data_ready("df_price"): assert False
            if not self.check_data_ready("np_trans_density"): assert False
            if not self.check_data_ready("np_CSR_sum"): assert False

        self.Cidx_LYM=self.LYM.index(YM)
        allaxes = fig.get_axes()
        for axe in allaxes:
            axe.remove()
        fig.add_subplot(221)
        fig.add_subplot(222)
        fig.add_subplot(223)
        fig.add_subplot(224)
        allaxes = fig.get_axes()
        fig.subplots_adjust(bottom=0.05, top=0.9, left=0.03, right=0.97, wspace=0.1, hspace=0.3)
        fig.suptitle("Summary at EvalT{0}  {1}  {2} {3}".format(self.system_name,EvalT,stock,YM), fontsize=14)
        self.plot_reward_count(allaxes[0],EvalT)
        SYM = self.LYM[self.Cidx_LYM]
        if not self.check_data_ready("df_bs"): assert False
        if not self.check_data_ready("df_price"): assert False

        self.plot_price_buy_sell(allaxes[1], SYM, self.df_bs, self.df_price)

        ax=allaxes[2]
        self.img_reward_date(fig, ax, self.np_trans_density,int(self.i_ana_data.data_start_s[:4]),
                                                            int(self.i_ana_data.data_end_s[:4]),Dimgc)
        ax.set_title("transaction density for {0} at ET{1}".format(stock, EvalT))

        ax=allaxes[3]
        self.img_reward_date(fig, ax, self.np_reward_distribution,int(self.i_ana_data.data_start_s[:4]),
                                                                    int(self.i_ana_data.data_end_s[:4]),Dimgc)
        ax.set_title("Reward Distribution for {0} at ET{1}".format(stock, EvalT))

    def show_reward_distribution(self,fig,stock, EvalT,Dimgc,sub_choice="median"):  #show_choice="mean" "median", "count", "std"
        allaxes = fig.get_axes()
        for axe in allaxes:
            axe.remove()
        fig.add_subplot(221)
        fig.add_subplot(222)
        fig.add_subplot(223)
        fig.add_subplot(224)
        allaxes = fig.get_axes()
        fig.subplots_adjust(bottom=0.07, top=0.9, left=0.03, right=0.97, wspace=0.1, hspace=0.3)
        fig.suptitle("Reward distribution on ET and stock", fontsize=14)

        self.plot_reward_count(allaxes[0], EvalT)
        self.show_reward_all_ET_all_stock(allaxes[1], fig, Dimgc)
        self.show_ET_reward_on_all_stock(allaxes[2], EvalT, sub_choice)
        self.show_stock_reward_on_all_ET(allaxes[3], stock, sub_choice)

    def show_reward_compare_ET(self, fig,lpriliminary_choice, lsecondary_choice,hist_param,Dimgc,sub_choice):
        _, ET1, _=lpriliminary_choice
        _, ET2, _=lsecondary_choice
        allaxes = fig.get_axes()
        for axe in allaxes:
            axe.remove()
        fig.add_subplot(231)
        fig.add_subplot(232)
        fig.add_subplot(233)
        fig.add_subplot(234)
        fig.add_subplot(235)
        fig.add_subplot(236)

        allaxes = fig.get_axes()
        fig.suptitle("{0} compare ET {1}  {2}".format(self.system_name,ET1, ET2), fontsize=14)
        fig.subplots_adjust(bottom=0.05, top=0.9, left=0.03, right=0.97, wspace=0.1, hspace=0.3)
        ax=allaxes[0]
        self.plot_reward_count(ax,ET1)
        ymin,ymax=ax.get_ylim()
        stc = ET2 / self.i_ana_data.lgc.num_train_to_save_model - 1
        ax.plot([stc,stc],[ymin,ymax])

        self.hist_reward(allaxes[1], ET1,hist_param)
        self.show_ET_reward_on_all_stock(allaxes[2], ET1, sub_choice)
        self.show_reward_all_ET_all_stock(allaxes[3], fig, Dimgc)
        self.hist_reward(allaxes[4], ET2, hist_param)
        self.show_ET_reward_on_all_stock(allaxes[5], ET2, sub_choice)

    def show_reward_compare_ET__stock(self, fig,lpriliminary_choice, lsecondary_choice,hist_param,Dimgc,sub_choice):
        stock1, ET1, LM1=lpriliminary_choice
        stock2, ET2, LM2=lsecondary_choice
        #assert stock1==stock2
        if not (stock1== self.Lstock[self.Cidx_stock] and ET1==self.LEvalT[self.Cidx_EvalT]):
            self.Cidx_stock=self.Lstock.index(stock1)
            self.Cidx_EvalT=self.LEvalT.index(ET1)
            #self.df_bs = self.i_ana_data.get_tranaction_price_data(self.Lstock[self.Cidx_stock],
            #                                                       self.LEvalT[self.Cidx_EvalT])
            flag_opt, df, _ = self.i_ana_data._get_are_summary_1stock_1ET(self.Lstock[self.Cidx_stock],
                                                                          self.LEvalT[self.Cidx_EvalT])
            self.df_bs = df if flag_opt else pd.DataFrame()

            self.df_price = self.i_ana_data.get_stock_hprice_data(self.Lstock[self.Cidx_stock])
            self.np_trans_density, self.np_reward_distribution=self.i_ana_data.\
                get_TransDensity_RewardDistribution(stock1, ET1)

        if not hasattr(self,"Cidx_stock2") or not hasattr(self,"Cidx_EvalT2"):
            self.Cidx_stock2=self.Lstock.index(stock2)
            self.Cidx_EvalT2=self.LEvalT.index(ET2)
            #self.df_bs2= self.i_ana_data.get_tranaction_price_data(self.Lstock[self.Cidx_stock2],
            #                                                       self.LEvalT[self.Cidx_EvalT2])
            flag_opt, df, _ = self.i_ana_data._get_are_summary_1stock_1ET(self.Lstock[self.Cidx_stock2],
                                                                          self.LEvalT[self.Cidx_EvalT2])

            self.df_bs2 = df if flag_opt else pd.DataFrame()

            self.df_price2 = self.i_ana_data.get_stock_hprice_data(self.Lstock[self.Cidx_stock2])
            self.np_trans_density2, self.np_reward_distribution2=self.i_ana_data.\
                get_TransDensity_RewardDistribution(stock2, ET2)
        elif not (stock2== self.Lstock[self.Cidx_stock2] and ET2==self.LEvalT[self.Cidx_EvalT2]):
            self.Cidx_stock2=self.Lstock.index(stock2)
            self.Cidx_EvalT2=self.LEvalT.index(ET2)
            #self.df_bs2= self.i_ana_data.get_tranaction_price_data(self.Lstock[self.Cidx_stock2],
            #                                                       self.LEvalT[self.Cidx_EvalT2])
            flag_opt, df, _ = self.i_ana_data._get_are_summary_1stock_1ET(self.Lstock[self.Cidx_stock2],
                                                                          self.LEvalT[self.Cidx_EvalT2])

            self.df_bs2 = df if flag_opt else pd.DataFrame()

            self.df_price2 = self.i_ana_data.get_stock_hprice_data(self.Lstock[self.Cidx_stock2])

        allaxes = fig.get_axes()
        for axe in allaxes:
            axe.remove()
        fig.add_subplot(221)
        fig.add_subplot(222)
        fig.add_subplot(223)
        fig.add_subplot(224)

        allaxes = fig.get_axes()
        fig.suptitle("{0} compare ET {1} vs {2}".format(self.system_name,ET1, ET2), fontsize=14)
        fig.subplots_adjust(bottom=0.05, top=0.9, left=0.03, right=0.97, wspace=0.1, hspace=0.3)
        ax=allaxes[0]
        self.plot_reward_count(ax,ET1)
        ymin,ymax=ax.get_ylim()
        stc = ET2 / self.i_ana_data.lgc.num_train_to_save_model - 1
        ax.plot([stc,stc],[ymin,ymax])

        if not self.check_data_ready("df_bs"): assert False
        if not self.check_data_ready("df_price"): assert False
        ax=allaxes[1]
        self.plot_price_buy_sell(ax, LM1, self.df_bs, self.df_price)
        ax.get_title()+"@ ET {0}".format(ET1)

        ax=allaxes[2]
        self.show_stock_reward_on_all_ET(ax, stock1, sub_choice)
        ymin,ymax=ax.get_ylim()
        stc = ET1 / self.i_ana_data.lgc.num_train_to_save_model - 1
        ax.plot([stc,stc],[ymin,ymax],color="r")
        stc = ET2 / self.i_ana_data.lgc.num_train_to_save_model - 1
        ax.plot([stc,stc],[ymin,ymax],color="y")

        ax=allaxes[3]
        self.plot_price_buy_sell(ax, LM2, self.df_bs2, self.df_price2)
        ax.get_title() + "@ ET {0}".format(ET2)

