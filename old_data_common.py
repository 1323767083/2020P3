#from n_utility import envirometn_config
import config as sc
import os,re
import pandas as pd
import numpy as np
import random
from sklearn.utils import shuffle

hfq_src_base_dir = "/home/rdchujf/Stk_qz_3_support/Stk_Day_FQ_WithHS"
index_src_base_dir = "/home/rdchujf/Stk_qz_3_support/Stk_Day_Idx"

def keyboard_input():
    data_name=input ("Data Name to create: ")
    while True:
        choice = input ("select stock type s(h) or s(z) sh+sz(a): ")
        if choice in ["h", "z","a"]:
            break
    stock_type="SH" if choice=="h" else "SZ" if choice=="z" else "sh+sz"

    while True:
        date_i = eval(input ("IPO date later than YYYYMMDD: "))
        if date_i>20100101:
            break
    return data_name, stock_type,date_i


class API_qz_data_source_related:
    #V2_start_tims_i = 20180101
    #V2_end_tims_i = 20190731
    #V2_start_tims_i = int(sc.V2_start_tims_s)
    #V2_end_tims_i = int(sc.V2_end_tims_s)

    def __init__(self):
        pass
    def check_according_dates(self, date_s):
        if int(date_s)< int(sc.V2_start_tims_s):
            return "_V1_"
        elif int(date_s)>= int(sc.V2_start_tims_s) and int(date_s)<=int(sc.V2_end_tims_s):
            return "_V2_"
        else:
            raise  ValueError ("{0} does not support".format(date_s))

    def check_according_data_name(self, data_name):
        regex = r'_V\d_'
        result=re.findall(regex, data_name)
        if len(result)==0:
            return "_V1_"
        elif len(result)==1:
            return result[0]
        else:
            raise ValueError("{0} does not recognize data_source".format(data_name))

    def get_data_state_end_time_s(self, data_name, market):
        data_type=self.check_according_data_name(data_name)
        assert market in ["SH", "SZ"]
        assert data_type  in ["_V1_", "_V2_"]
        if data_type =="_V1_":
            return [sc.qz_sh_avail_start, sc.qz_avail_end] if market=="SH" else [sc.qz_sz_avail_start, sc.qz_avail_end]
        else:
            return [sc.V2_start_tims_s,sc.V2_end_tims_s] if market=="SH" else ["Not_support","Not_support"]

    def get_valid_periods(self,data_type,stock):
        assert data_type in ["_V1_", "_V2_"]
        if data_type == "_V1_":
            skip_days = 100
            least_length = 23
            periods = self.V1_stock_avail_periods(stock, skip_days, least_length)
        elif data_type == "_V2_":
            #V2_start_tims_s = "20180101"
            #V2_end_tims_s = "20190731"
            td = API_trade_date().np_date_s
            df_HFQ = API_HFQ_from_file().get_df_HFQ(stock)
            start_date_s = sc.V2_start_tims_s if sc.V2_start_tims_s >= str(df_HFQ["date"].iloc[0]) \
                else str(df_HFQ["date"].iloc[0])
            end_date_s = sc.V2_end_tims_s if sc.V2_end_tims_s <= str(df_HFQ["date"].iloc[-1]) \
                else str(df_HFQ["date"].iloc[-1])
            periods = [td[(td >= start_date_s) & (td <= end_date_s)]]
        else:
            raise ValueError("Not support data_type {0} \n data_type == _V1_  means first qz data \n "
                             "data_type == _V2_  means 20180101-20190731 qz data")
        return periods


    def V1_stock_avail_periods(self,stock,ipo_skip_days,least_length):
        date_s_sh_avail_start = sc.qz_sh_avail_start
        date_s_sz_avail_start = sc.qz_sz_avail_start
        date_s_avail_end = sc.qz_avail_end


        td= API_trade_date().np_date_s
        df_mz=self.prepare_qz_mising_zero_df()

        periods=[]
        df_hfq = API_HFQ_from_file().get_df_HFQ(stock)

        first_avaliable_date_s = date_s_sh_avail_start if stock[:2] == "SH" else date_s_sz_avail_start
        IPO_date_s_add_skip_days = df_hfq["date"].iloc[ipo_skip_days]
        start_s= IPO_date_s_add_skip_days if IPO_date_s_add_skip_days >= first_avaliable_date_s \
            else first_avaliable_date_s
        end_s= date_s_avail_end

        l_breaks=df_mz[df_mz["stock"] == stock]["date"].values.tolist()

        if len(l_breaks)==0:
            periods.append(td[(td >= start_s) & (td <= end_s)])
            return periods

        l_breaks.sort(reverse=False)
        np_breaks=np.array(l_breaks)

        np_breaks=np_breaks[(np_breaks>start_s)&(np_breaks<end_s)]


        for idx, onebreak in enumerate(np_breaks):
            if idx==0:
                potential_period=td[(td>=start_s) &(td < onebreak)]
                if len(potential_period)>least_length:
                    periods.append(potential_period)
                last_break = onebreak
            elif idx==(len(np_breaks)-1):
                potential_period=td[(td>onebreak) &(td<=end_s)]
                if len(potential_period)>least_length:
                    periods.append(potential_period)
            else:
                potential_period=td[(td>last_break) &(td<onebreak)]
                if len(potential_period)>least_length:
                    periods.append(potential_period)
                last_break = onebreak

        return periods

    def prepare_qz_mising_zero_df(self):
        df_missing = pd.read_csv(missing_data_fnwp,header=0, names=["stock", "date"])
        df_missing["date"]=df_missing["date"].astype(str)
        df_missing["type"]="missing"
        df_zero=pd.read_csv(zero_len_data_fnwp, header=0,names=["stock", "date"])
        df_zero["date"] = df_zero["date"].astype(str)
        df_zero["type"]="zero"

        df=pd.concat([df_missing,df_zero], axis=0)
        return df
class API_qz_from_file:
    #V1 setting
    base_dir_qz_1 = sc.base_dir_qz_1
    base_dir_qz_2 = sc.base_dir_qz_2

    qz_sh_avail_start = sc.qz_sh_avail_start
    qz_sz_avail_start = sc.qz_sz_avail_start
    qz_avail_end = sc.qz_avail_end

    #V2 setting
    V2_start_tims_i = 20180101
    V2_end_tims_i = 20190731
    V2_src_base_dir = "/mnt/backup_6G/Stk_qz_3"

    def __init__(self):
        self.title_qz= ["TranID", "Time", "Price", "Volume", "SaleOrderVolume", "BuyOrderVolume", "Type", "SaleOrderID",
                 "SaleOrderPrice", "BuyOrderID", "BuyOrderPrice"]
        self.dtype_qz=  {
            "TranID":       int,    "Time":            str,     "Price":           float,
            "Volume":       int,    "SaleOrderVolume": int,     "BuyOrderVolume":  int,     "Type":         str,
            "SaleOrderID":  int,    "SaleOrderPrice":  float,   "BuyOrderID":      int,     "BuyOrderPrice":float}

    def _V2_get_fnwp_qz(self,date_s, stock):
        assert len(date_s)==8
        assert len(stock)==8
        i_year  = int(date_s[0:4])
        i_month = int(date_s[4:6])
        i_day   = int(date_s[6:8])
        stock_type=stock[0:2]
        stock_code=stock[2:]
        if int(date_s)>=self.V2_start_tims_i and int(date_s)<=self.V2_end_tims_i:
            pass
        else:
            raise ValueError("Does not has data for {0} {1}".format(date_s, stock))

        month_sub_dir= "{0}{1:02d}".format(i_year, i_month)
        day_sub_dir= "{0}-{1:02d}-{2:02d}".format(i_year, i_month,i_day)
        if date_s>="20180716" and date_s<="20180719":
            fn = "{0}{1}.csv".format(stock_type.upper(), stock_code)
        else:
            fn="{0}.csv".format(stock_code)
        fnwp=os.path.join(self.V2_src_base_dir,month_sub_dir, day_sub_dir,fn)
        return fnwp

    def _V1_get_fnwp_qz(self,date_s, stock):
        assert len(date_s)==8
        assert len(stock)==8
        year=date_s[0:4]
        year_month=date_s[0:6]

        stock_type=stock[0:2]
        stock_code=stock[2:]

        if stock_type=="SH":
            assert date_s >=self.qz_sh_avail_start

        if stock_type=="SZ":
            assert date_s >=self.qz_sz_avail_start

        assert date_s<=self.qz_avail_end

        date_with_dash="{0}-{1}-{2}".format(date_s[0:4], date_s[4:6], date_s[6:8])

        if year in ["2013", "2014"]:
            return os.path.join(self.base_dir_qz_1,year,"{0}{1}".format(year_month,stock_type),date_with_dash,"{0}.csv".format(stock_code))
        elif year in ["2016"]:
            return os.path.join(self.base_dir_qz_1, year, date_with_dash,"{0}.csv".format(stock_code))
        elif year in ["2015"]:
            return os.path.join(self.base_dir_qz_2, year, date_with_dash,"{0}.csv".format(stock_code))
        elif year in ["2017"]:
            return os.path.join(self.base_dir_qz_2, year, date_with_dash,"{0}.csv".format(stock_code))
        else:
            raise ValueError( "Does not has data for {0} {1}".format(date_s,stock))

    def _qz_filter(self, date_s, df):
        if date_s>="20170801":
            return df
        else:
            sampled_index = random.sample(df.index, int(len(df) * 0.995))
            df=df.loc[sampled_index]
            df.reset_index(inplace=True, drop=True)
            return df


    def get_fnwp_qz(self,date_s, stock):
        data_source_type=API_qz_data_source_related().check_according_dates(date_s)
        if data_source_type=="_V1_":
            fnwp = self._V1_get_fnwp_qz(date_s, stock)
        elif data_source_type=="_V2_":
            fnwp = self._V2_get_fnwp_qz(date_s, stock)
        else:
            raise  ValueError ("{0} does not support".format(date_s))
        return fnwp


    def get_df_qz(self,date_s,stock):
        fnwp=self.get_fnwp_qz(date_s, stock)
        if not os.path.exists(fnwp):
            raise ValueError("{0} does not exists".format(fnwp))
        else:
           try:
               df = pd.read_csv(fnwp, header=0, names=self.title_qz, dtype=self.dtype_qz)
           except ValueError as e:
               # this is to handle 13 in file as 12.99999997 situation
               if "cannot safely convert passed user dtype of int64 for float64 dtyped data" in str(e):
                   df = pd.read_csv(fnwp, header=0, names=self.title_qz)
                   for item in list(self.dtype_qz.keys()):
                       df[item]=df[item].astype(self.dtype_qz[item])
               # 6883,undefined,undefined,NaN,NaN,NaN,undefined,undefined,undefined,undefined,undefined
               #error message could not convert string to float: undefined
               elif "undefined" in str(e):
                   df = pd.read_csv(fnwp, header=0, names=self.title_qz)
                   df.dropna(inplace=True)
                   for item in list(self.dtype_qz.keys()):
                       df[item]=df[item].astype(self.dtype_qz[item])
               else:
                   raise ValueError(str(e))
           #df = pd.read_csv(fnwp, header=0, names=self.title_qz, dtype=self.dtype_qz)
           df.drop(["TranID"], axis=1, inplace=True)
           if df.empty:
               raise ValueError("{0} is empty".format(fnwp))

           df["Time"] = df["Time"].apply(lambda x: int(x.replace(":","") ))


           df.at[df[df["Time"] >= 150000].index, "Time"]= 145959
           df.at[df[df["Time"] < 92500].index, "Time"]=92500
           df.at[df[(df["Time"] < 130000)&(df["Time"] >= 113000)].index, "Time"]=112959

           df["Money"]=df["Volume"]*df["Price"]

        return self._qz_filter(date_s, df)

class API_HFQ_from_file:

    def __init__(self):
        self.hfq_src_base_dir = hfq_src_base_dir
        self.title_hfq=["code","date","open_price","highest_price","lowest_price","close_price","amount_gu",
                        "amount_yuan", "exchange_ratio_for_tradable_part", "exchange_ratio_for_whole",
                        "coefficient_fq"]
        self.dtype_hfq={"code":str,"date":str,"open_price":float,"highest_price":float,"lowest_price":float,
                        "close_price":float,"amount_gu":int,"amount_yuan":float,
                        "exchange_ratio_for_tradable_part":float, "exchange_ratio_for_whole":float,"coefficient_fq":float}


    def _HFQ_fnwp(self, stock):
        assert len(stock)==8
        fnwp=os.path.join(self.hfq_src_base_dir,stock.upper()+".csv")
        return fnwp

    def get_df_HFQ(self, stock):
        assert len(stock)==8
        fnwp=self._HFQ_fnwp(stock)
        if not os.path.exists(fnwp):
            raise ValueError("{0} does not exists".format(fnwp))
        else:
            df=pd.read_csv(fnwp,encoding="gb18030", header=0, names=self.title_hfq, dtype=self.dtype_hfq)
            df["date"] = df["date"].apply(lambda date_s: date_s.replace("-",""))
            #df["date"] = df["date"].apply(lambda date_s: date_s[0:4] + date_s[5:7] + date_s[8:10])
        return df
class API_index_from_file:

    def __init__(self):
        self.index_src_base_dir = index_src_base_dir
        self.title_index=["code","date","open_price","highest_price","lowest_price","close_price","amount_gu",
                        "amount_yuan"]
        self.dtype_index={"code":str,   "date":str, "open_price":float, "highest_price":float,  "lowest_price":float,
                          "close_price":float,"amount_gu":int,"amount_yuan":float}

    def _fnwp_index(self, index):
        assert len(index) == 8
        fnwp=os.path.join(self.index_src_base_dir,index.lower()+".csv")
        return fnwp

    def get_df_index(self, index):
        assert len(index)==8
        fnwp=self._fnwp_index(index)
        if not os.path.exists(fnwp):
            raise ValueError("{0} does not exists".format(fnwp))
        else:
            df=pd.read_csv(fnwp,encoding="gb18030", header=0, names=self.title_index,dtype=self.dtype_index)
            df["date"] = df["date"].apply(lambda date_s: date_s.replace("-", ""))
            #df["date"] = df["date"].apply(lambda date_s: date_s[0:4] + date_s[5:7] + date_s[8:10])
        return df
class API_trade_date:
    def __init__(self):
        df = API_index_from_file().get_df_index("SH000001")
        self.np_date_s = df[df["date"] >= "20100927"]["date"].astype(str).values
        #self.np_date_s=df["date"].astype(str).values

    def check_trading_date(self,date_s):
        #return np.any(self.np_date_s == date_s)
        return date_s in self.np_date_s
    def get_n_trade_days_after_non_trade_date(self,date_s, n):
        assert not self.check_trading_date(date_s)
        return self.np_date_s[np.where(self.np_date_s > date_s)[0][n-1]]
    def get_n_trade_days_before_non_trade_date(self, date_s, n):
        assert not self.check_trading_date(date_s)
        return self.np_date_s[np.where(self.np_date_s < date_s)[0][-n]]
    def get_n_trade_days_after_trade_date(self,date_s, n):
        assert self.check_trading_date(date_s)
        return self.np_date_s[np.where(self.np_date_s >= date_s)[0][n]]
    def get_n_trade_days_before_trade_date(self, date_s, n):
        assert self.check_trading_date(date_s)
        return self.np_date_s[np.where(self.np_date_s <= date_s)[0][-n-1]]
class ginfo_one_stock:
    def __init__(self, stock):
        self.stock = stock

        self.i_tdate=API_trade_date()
        self.df_hfq=API_HFQ_from_file().get_df_HFQ(self.stock)
    def check_not_tinpai(self,date_s):
        assert self.i_tdate.check_trading_date(date_s)
        #return np.any( self.df_hfq["date"] == date_s)
        return date_s in self.df_hfq["date"].values
    def check_after_IPO(self, date_s):
        assert self.i_tdate.check_trading_date(date_s)
        IPO_date_s=self.df_hfq["date"][0]
        return date_s>=IPO_date_s

    def hfq_amount(self, date_s):  #this is called after check_not_tinpai
        return self.df_hfq[self.df_hfq["date"] == date_s].iloc[0]["amount_gu"]
    def hfq_ratio(self,date_s): #this is called after check_not_tinpai
        return self.df_hfq[self.df_hfq["date"]==date_s].iloc[0]["coefficient_fq"]

    def get_closest_close_Nprice(self, date_s):
        df=self.df_hfq[self.df_hfq["date"] <= date_s]
        assert len(df)>0, "no_price_before_that_day {0} {1}".format(self.stock,date_s)
        hfq_price = df.iloc[-1]["close_price"]
        hfq_ratio = df.iloc[-1]["coefficient_fq"]
        Nprice=hfq_toolbox().get_Nprice_from_hfq_price(hfq_price,hfq_ratio )

        return Nprice,hfq_ratio

    def get_open_price(self,date_s): #this is called after check_not_tinpai
        df=self.df_hfq[self.df_hfq["date"] == date_s]
        assert len(df)>0
        hfq_price = df.iloc[0]["open_price"]
        hfq_ratio = df.iloc[0]["coefficient_fq"]
        Nprice = hfq_toolbox().get_Nprice_from_hfq_price(hfq_price, hfq_ratio)
        return Nprice, hfq_ratio

    def get_exchange_ratio_for_tradable_part(self, date_s):
        return self.df_hfq[self.df_hfq["date"] == date_s].iloc[0]["exchange_ratio_for_tradable_part"]

class hfq_toolbox:
    def get_Nprice_from_hfq_price(self, hfq_price, hfq_ratio):
        return hfq_price/hfq_ratio
    def get_update_volume_on_hfq_ratio_change(self, old_hfq_ratio, new_hfq_ratio, old_volume):
        #return old_volume*old_hfq_ratio/new_hfq_ratio
        return old_volume *  new_hfq_ratio/ old_hfq_ratio
    def get_hfqprice_from_Nprice(self, Nprice, hfq_ratio):
        return Nprice * hfq_ratio

class exclude_stock_list:
    def __init__(self, data_name):
        self.fnwp = os.path.join(sc.base_dir_RL_data, data_name, "exlude_stock_list.csv")
        if not os.path.exists(self.fnwp):
            self.df=pd.DataFrame(columns=["stock", "reason"])
        else:
            try:
                self.df=pd.read_csv(self.fnwp, header=0, names=["stock", "reason"], dtype={"stock": str,"reason":"str"})
            except:
                self.df = pd.read_csv(self.fnwp, header=0, names=["stock"],dtype={"stock": str})
                self.df["reason"]="not_specified"
    def check_in_exclude_list(self, stock):
        return stock in self.df["stock"].tolist()
    def add_to_exlude_list(self, stock, reason="not_specified"):
        if not self.check_in_exclude_list(stock):
            self.df.loc[len(self.df)]=[stock, reason]
            self.df.to_csv(self.fnwp, index=False)


class _API_common_sl:
    sl_fn = "Fake_file_name"

    def __init__(self,data_name):
        self.data_name=data_name
        self.dir_hfg=hfq_src_base_dir
        self.least_num_days=sc.RL_data_least_length+sc.RL_data_skip_days
        self.i_exlude_list = exclude_stock_list(data_name)
        self.working_dir=os.path.join(sc.base_dir_RL_data,self.data_name)
        if not os.path.exists(self.working_dir): os.mkdir(self.working_dir)
        self.sl_fnwp=os.path.join(self.working_dir,self.sl_fn)
        self.flag_exists = True if self.check_exsits() else False
        self.prepare_stock_list()

    def get_proper_sh_stock_list(self):  # 100 skip day 20 observe day 2 buy sell
        sh_stock_list=[]
        raw_sh_stock_list=[fn[0:8] for fn in os.listdir(self.dir_hfg) if "SH6" in fn]
        for stock in raw_sh_stock_list:
            fnwp=os.path.join(self.dir_hfg, "{0}.csv".format(stock))
            df = pd.read_csv(fnwp, encoding="gb18030", header=0)
            if len(df)>=self.least_num_days:
                sh_stock_list.append(stock)
            else:
                print("\t{0} does not have enough days after IPO".format(stock))
        return sh_stock_list

    def get_proper_sz_stock_list(self):
        sz_stock_list=[]
        raw_sz_stock_list=[fn[0:8] for fn in os.listdir(self.dir_hfg) if "SZ000" in fn]
        for stock in raw_sz_stock_list:
            fnwp=os.path.join(self.dir_hfg, "{0}.csv".format(stock))
            df = pd.read_csv(fnwp, encoding="gb18030", header=0)
            if len(df)>=self.least_num_days:
                sz_stock_list.append(stock)
            else:
                print("\t{0} does not have enough days after IPO".format(stock))
        return sz_stock_list

    def check_exsits(self):
        return os.path.exists(self.sl_fnwp)

    def load_total_stock_list(self):  #this is before empty and missing data has made
        df = pd.read_csv(self.sl_fnwp, header=0, names=["Stock"])
        return df["Stock"].values.tolist()

    def load_total_proper_stock_list(self):  #this exclude the unproper stock
        df = pd.read_csv(self.sl_fnwp, header=0, names=["Stock"])
        stock_list=df["Stock"].values.tolist()
        for stock in list(stock_list):
            if self.i_exlude_list.check_in_exclude_list(stock):
                stock_list.remove(stock)
        return stock_list


    def load_stock_list(self, base,num):
        df = pd.read_csv(self.sl_fnwp, header=0, names=["Stock"])
        stock_list=[row[0] for idx, row in df.iterrows() if idx%base==num]
        for stock in list(stock_list):
            if self.i_exlude_list.check_in_exclude_list(stock):
                stock_list.remove(stock)
        return stock_list

    def save_stock_list(self, stock_list):
        df = pd.DataFrame(stock_list, columns=["Stock"])
        df = shuffle(df)
        df.to_csv(self.sl_fnwp, index=False)
        return df["Stock"].values.tolist()

    def prepare_stock_list(self):
        assert  False, "this is a fake function"

class API_SH_SZ_total_sl(_API_common_sl):
    sl_fn="stock_list.csv"
    def prepare_stock_list(self):
        if self.flag_exists:
            return
        lsh=self.get_proper_sh_stock_list()
        lsz=self.get_proper_sz_stock_list()
        lsh.extend(lsz)
        self.save_stock_list(lsh)
class API_SH_sl(_API_common_sl):
    sl_fn="SH_stock_list.csv"
    def prepare_stock_list(self):
        if self.flag_exists:
            return
        lsh=self.get_proper_sh_stock_list()
        self.save_stock_list(lsh)
class API_SZ_sl(_API_common_sl):
    sl_fn="SZ_stock_list.csv"
    def prepare_stock_list(self):
        if self.flag_exists:
            return
        lsh=self.get_proper_sz_stock_list()
        self.save_stock_list(lsh)



#actually not used
class API_G_IPO_sl(_API_common_sl):

    def __init__(self,data_name, stock_type, latest_IPO_date_s):
        self.stock_type=stock_type
        self.latest_IPO_date_s=latest_IPO_date_s
        self.sl_fn="IPO_{0}_G{1}_stock_list.csv".format(stock_type,latest_IPO_date_s)
        _API_common_sl.__init__(self,data_name)
    def prepare_stock_list(self):
        if self.flag_exists:
            return
        assert self.stock_type in ["SH","SZ","ALL"]
        if self.stock_type == "SH":
            input_sl = self.get_proper_sh_stock_list()
        elif self.stock_type == "SZ":
            input_sl = self.get_proper_sz_stock_list()
        else:
            assert self.stock_type == "ALL"
            input_sl = self.get_proper_sh_stock_list()
            lsz = self.get_proper_sz_stock_list()
            input_sl.extend(lsz)

        selected_sl=[]
        for stock in input_sl:
            df_hfq=API_HFQ_from_file().get_df_HFQ(stock)
            if len(df_hfq)==0:
                raise ValueError("{0} is empty".format(stock))
            if str(df_hfq.iloc[0]["date"])<=self.latest_IPO_date_s:
                print("add {0} in stock_list".format(stock))
                selected_sl.append(stock)
        self.save_stock_list(selected_sl)



#legecy
missing_data_fnwp = sc.missing_data_fnwp
zero_len_data_fnwp = sc.zero_len_data_fnwp
class V1_qz_extract_data_sanity_check:
    def __init__(self):
        self.rar_base_dir=sc.qz_rar_dir
        self.des_base_dir=sc.base_dir_qz_1
        self.des_base_dir_2 = sc.base_dir_qz_2

        self.convert_date_s = lambda date_s: "{0}-{1}-{2}".format(date_s[0:4], date_s[4:6], date_s[6:8])

        self.date_s_sh_avail_start = sc.qz_sh_avail_start
        self.date_s_sz_avail_start = sc.qz_sz_avail_start
        self.date_s_avail_end = sc.qz_avail_end


        self.td= API_trade_date().np_date_s
        self.df_mz=self.prepare_qz_mising_zero_df()

    def check_missing_data(self):

        date_s_sh_start=sc.qz_sh_avail_start
        date_s_sz_start = sc.qz_sz_avail_start
        date_s_end=sc.qz_avail_end

        log = []
        stock_list = API_SH_SZ_total_sl("ft", 1).load_total_stock_list()
        i_HFQ=API_HFQ_from_file()
        td=API_trade_date().np_date_s
        i_qz=API_qz_from_file()
        for stock in stock_list:
        #for stock in ["SH600000","SZ000001"]:
            df_hfq=i_HFQ.get_df_HFQ(stock)
            date_s_stock_ipo=df_hfq["date"].iloc[0]
            if stock[:2]=="SH":
                start_s = date_s_sh_start if date_s_stock_ipo<date_s_sh_start else date_s_stock_ipo
            else:
                start_s = date_s_sh_start if date_s_stock_ipo < date_s_sz_start else date_s_stock_ipo
            end_s=date_s_end
            period = td[(td >= start_s) & (td <= end_s)]

            for day in period:
                if np.any(df_hfq["date"] == day): #not tinpai
                    fnwp=i_qz.get_fnwp_qz(day,stock)
                    if not os.path.exists(fnwp):
                        print("missing {0} {1}".format(stock, day))
                        log.append([stock, day])


        df_log=pd.DataFrame(log, columns=["stock","date"])
        df_log.to_csv(missing_data_fnwp,index=False)

    def check_empty_file(self):
        date_s_sh_start=sc.qz_sh_avail_start
        date_s_sz_start = sc.qz_sz_avail_start
        date_s_end=sc.qz_avail_end

        log = []
        stock_list = API_SH_SZ_total_sl("ft", 1).load_total_stock_list()  # SKIP DAY IS 1 ONLY FOR CHECK EMPTY FILE
        i_hfq=API_HFQ_from_file()
        td=API_trade_date().np_date_s
        i_qz=API_qz_from_file()
        for stock in stock_list:
        #for stock in ["SH600000","SZ000001"]:
            df_hfq=i_hfq.get_df_HFQ(stock)
            date_s_stock_ipo=df_hfq["date"].iloc[0]
            if stock[:2]=="SH":
                start_s = date_s_sh_start if date_s_stock_ipo<date_s_sh_start else date_s_stock_ipo
            else:
                start_s = date_s_sh_start if date_s_stock_ipo < date_s_sz_start else date_s_stock_ipo
            end_s=date_s_end
            period = td[(td >= start_s) & (td <= end_s)]

            for day in period:
                if np.any(df_hfq["date"] == day): #not tinpai
                    fnwp=i_qz.get_fnwp_qz(day,stock)
                    if os.path.exists(fnwp):
                        b = os.path.getsize(fnwp)
                        if b==0:
                            print("missing {0} {1}".format(stock, day))
                            log.append([stock, day])


        df_log=pd.DataFrame(log, columns=["stock","date"])
        df_log.to_csv(zero_len_data_fnwp,index=False)

    def analysis_missing_data(self):
        import pandas as pd
        df = pd.read_csv(missing_data_fnwp, names=["stock", "date"])

        w_drop_list=["20170904","20170905","20170906","20170907","20170908","20170801","20170802","20170803","20170804","20131008","20140218","20171127"]
        c_drop_list=["20161024","20161025"]
        drop_list=w_drop_list+c_drop_list
        for date in drop_list:
            df=df.drop(df[df["date"] == date].index)

        stock_drop_list=["SH600832","SH600253","SH600656","SH601299","SH601360","SZ000527","SZ000562","SZ000594","SZ000602"]
        for stock in stock_drop_list:
            df=df.drop(df[df["stock"] == stock].index)

        print(df.groupby(["date"]).count())
        print(df.groupby(["stock"]).count())

    #----------------interface data available
    def prepare_qz_mising_zero_df(self):

        df_missing = pd.read_csv(missing_data_fnwp,header=0, names=["stock", "date"])
        df_missing["date"]=df_missing["date"].astype(str)
        df_missing["type"]="missing"
        df_zero=pd.read_csv(zero_len_data_fnwp, header=0,names=["stock", "date"])
        df_zero["date"] = df_zero["date"].astype(str)
        df_zero["type"]="zero"

        df=pd.concat([df_missing,df_zero], axis=0)
        return df

