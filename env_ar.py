from data_common import hfq_toolbox,ginfo_one_stock
def ent_ar_init(lgc):
    global lc
    lc=lgc
    global flag_accout_store_transaction_record
    flag_accout_store_transaction_record = False

class env_account:
    param_shou_xu_fei = 0.00025
    param_yin_hua_shui = 0.001
    param_communicatefei = 1
    param_guohufei = 1
    transit_record_template = {
        "Action": "",  # "Buy"
        "date": "",
        "volume_gu": 0,
        "Nprice": 0,
        "Hratio": 1.0,  # for caculate volume later
        "action_cost": 0.0
    }

    def __init__(self):
        self.i_hfq_tb = hfq_toolbox()
        self._account_reset()
        assert lc.env_min_invest_per_round<=lc.env_max_invest_per_round
        self.invest_per_term=lc.env_min_invest_per_round
        self.max_num_invest=int(lc.env_max_invest_per_round/lc.env_min_invest_per_round)
        self.current_stock=""
        self.stock_ginfom=""

    #### common function
    def _sell_stock_cost(self, volume, price):
        tmp_total_money = volume * price
        sell_cost = tmp_total_money * (self.param_shou_xu_fei + self.param_yin_hua_shui) + \
                    self.param_communicatefei + \
                    self.param_guohufei * int(volume / 1000.0)
        return sell_cost

    def _buy_stock_cost(self, volume, price):
        tmp_total_money = volume * price
        buy_cost = tmp_total_money * self.param_shou_xu_fei + \
                   self.param_communicatefei + \
                   self.param_guohufei * int(volume / 1000.0)
        return buy_cost

    def _get_stock_ginform(self,stock):
        if stock!=self.current_stock:
            self.stock_ginfom= ginfo_one_stock(stock)
            self.current_stock=stock
        return self.stock_ginfom

    #### account API
    def _account_reset(self):
        self.volume_gu = 0
        self.total_invest = 0.0
        self.Hratio = 1.0
        self.buy_times = 0
        if flag_accout_store_transaction_record:
            if hasattr(self,"transit_history"):
                del self.transit_history[:]
            else:
                self.transit_history = []

    def _account_buy(self, date_s,trade_Nprice, trade_hfq_ratio):
        if self.buy_times < self.max_num_invest:
            volume_gu = int(self.invest_per_term * 0.995 / (trade_Nprice * 100)) * 100
            assert volume_gu != 0, "{0} {1} {2} can not buy one hand".format(self.current_stock, date_s,
                                                                                   self.invest_per_term)
            buy_cost= self._buy_stock_cost(volume_gu, trade_Nprice)
            if flag_accout_store_transaction_record:
                tr=dict(self.transit_record_template)
                tr["Action"]        =   "Buy"
                tr["date"]          =   date_s
                tr["volume_gu"]     =   volume_gu
                tr["Nprice"]        =   trade_Nprice
                tr["Hratio"]        =   trade_hfq_ratio
                tr["action_cost"]   =   buy_cost
                self.transit_history.append(tr)
            current_holding_volume_gu = self.i_hfq_tb.get_update_volume_on_hfq_ratio_change(old_hfq_ratio=self.Hratio,
                                                                                new_hfq_ratio=trade_hfq_ratio,
                                                                                old_volume=self.volume_gu)
            self.volume_gu = current_holding_volume_gu + volume_gu
            self.Hratio    = trade_hfq_ratio
            self.total_invest += volume_gu * trade_Nprice + buy_cost
            self.buy_times += 1
            return True, "Success"
        else:
            return False, "Exceed_limit"

    def _account_sell(self,date_s,trade_Nprice, trade_hfq_ratio):
        if self.volume_gu != 0:
            current_holding_volume_gu = self.i_hfq_tb.get_update_volume_on_hfq_ratio_change \
                (old_hfq_ratio=self.Hratio, new_hfq_ratio=trade_hfq_ratio, old_volume=self.volume_gu)
            sell_cost = self._sell_stock_cost(current_holding_volume_gu, trade_Nprice)
            total_money_back = current_holding_volume_gu * trade_Nprice - sell_cost
            total_money_invest = self.total_invest
            profit = total_money_back / total_money_invest - 1.0
            self._account_reset()
            return True, "Success", profit
        else:
            return False, "No_holding", 0.0

    ####env_account interface
    def reset(self):
        self._account_reset()

    def buy(self, trade_Nprice, trade_hfq_ratio, stock, date_s):
        i_ginform=self._get_stock_ginform(stock)
        if not i_ginform.check_not_tinpai(date_s):
            return False,"Tinpai"
        return self._account_buy(date_s,trade_Nprice, trade_hfq_ratio)

    def sell(self, trade_Nprice,trade_hfq_ratio,stock, date_s):
        i_ginform=self._get_stock_ginform(stock)
        if not i_ginform.check_not_tinpai(date_s):  # if tinpai
            return False,"Tinpai",0.0
        return self._account_sell(date_s,trade_Nprice, trade_hfq_ratio)

    def eval(self):  #this trade day price
        return 1 if self.volume_gu!=0 else 0
        #if self.volume_gu != 0
            #current_holding_volume_gu = self.i_hfq_tb.get_update_volume_on_hfq_ratio_change \
            #    (old_hfq_ratio=self.Hratio, new_hfq_ratio=trade_hfq_ratio, old_volume=self.volume_gu)
            #potential_sell_cost = self._sell_stock_cost(current_holding_volume_gu, trade_Nprice)
            #potential_money_back = current_holding_volume_gu * trade_Nprice - potential_sell_cost
            #potential_profit=potential_money_back/self.total_invest-1.0
            #return 1
        #else:
        #    return 0


class env_account_old:
    param_shou_xu_fei = 0.00025
    param_yin_hua_shui = 0.001
    param_communicatefei = 1
    param_guohufei = 1
    transit_record_template = {
        "Action": "",  # "Buy"
        "date": "",
        "volume_gu": 0,
        "Nprice": 0,
        "Hratio": 1.0,  # for caculate volume later
        "action_cost": 0.0
    }

    def __init__(self):
        self.i_hfq_tb = hfq_toolbox()
        self._account_reset()
        assert lc.env_min_invest_per_round<=lc.env_max_invest_per_round
        self.invest_per_term=lc.env_min_invest_per_round
        self.max_num_invest=int(lc.env_max_invest_per_round/lc.env_min_invest_per_round)
        self.current_stock=""
        self.stock_ginfom=""

    #### common function
    def _sell_stock_cost(self, volume, price):
        tmp_total_money = volume * price
        sell_cost = tmp_total_money * (self.param_shou_xu_fei + self.param_yin_hua_shui) + \
                    self.param_communicatefei + \
                    self.param_guohufei * int(volume / 1000.0)
        return sell_cost

    def _buy_stock_cost(self, volume, price):
        tmp_total_money = volume * price
        buy_cost = tmp_total_money * self.param_shou_xu_fei + \
                   self.param_communicatefei + \
                   self.param_guohufei * int(volume / 1000.0)
        return buy_cost

    def _get_stock_ginform(self,stock):
        if stock!=self.current_stock:
            self.stock_ginfom= ginfo_one_stock(stock)
            self.current_stock=stock
        return self.stock_ginfom

    #### account API
    def _account_reset(self):
        self.volume_gu = 0
        self.total_invest = 0.0
        self.Hratio = 1.0
        self.buy_times = 0
        if flag_accout_store_transaction_record:
            if hasattr(self,"transit_history"):
                del self.transit_history[:]
            else:
                self.transit_history = []

    def _account_buy(self, date_s,trade_Nprice, trade_hfq_ratio):
        if self.buy_times < self.max_num_invest:
            volume_gu = int(self.invest_per_term * 0.995 / (trade_Nprice * 100)) * 100
            assert volume_gu != 0, "{0} {1} {2} can not buy one hand".format(self.current_stock, date_s,
                                                                                   self.invest_per_term)
            buy_cost= self._buy_stock_cost(volume_gu, trade_Nprice)
            if flag_accout_store_transaction_record:
                tr=dict(self.transit_record_template)
                tr["Action"]        =   "Buy"
                tr["date"]          =   date_s
                tr["volume_gu"]     =   volume_gu
                tr["Nprice"]        =   trade_Nprice
                tr["Hratio"]        =   trade_hfq_ratio
                tr["action_cost"]   =   buy_cost
                self.transit_history.append(tr)
            current_holding_volume_gu = self.i_hfq_tb.get_update_volume_on_hfq_ratio_change(old_hfq_ratio=self.Hratio,
                                                                                new_hfq_ratio=trade_hfq_ratio,
                                                                                old_volume=self.volume_gu)
            self.volume_gu = current_holding_volume_gu + volume_gu
            self.Hratio    = trade_hfq_ratio
            self.total_invest += volume_gu * trade_Nprice + buy_cost
            self.buy_times += 1
            return True, "Success"
        else:
            return False, "Exceed_limit"

    def _account_sell(self,date_s,trade_Nprice, trade_hfq_ratio):
        if self.volume_gu != 0:
            current_holding_volume_gu = self.i_hfq_tb.get_update_volume_on_hfq_ratio_change \
                (old_hfq_ratio=self.Hratio, new_hfq_ratio=trade_hfq_ratio, old_volume=self.volume_gu)
            sell_cost = self._sell_stock_cost(current_holding_volume_gu, trade_Nprice)
            total_money_back = current_holding_volume_gu * trade_Nprice - sell_cost
            total_money_invest = self.total_invest
            profit = total_money_back / total_money_invest - 1.0
            self._account_reset()
            return True, "Success", profit
        else:
            return False, "No_holding", 0.0

    ####env_account interface
    def reset(self):
        self._account_reset()

    def buy(self, trade_Nprice, trade_hfq_ratio, stock, date_s):
        i_ginform=self._get_stock_ginform(stock)
        if not i_ginform.check_not_tinpai(date_s):
            return False,"Tinpai"
        return self._account_buy(date_s,trade_Nprice, trade_hfq_ratio)

    def sell(self, trade_Nprice,trade_hfq_ratio,stock, date_s):
        i_ginform=self._get_stock_ginform(stock)
        if not i_ginform.check_not_tinpai(date_s):  # if tinpai
            return False,"Tinpai",0.0
        return self._account_sell(date_s,trade_Nprice, trade_hfq_ratio)

    def eval(self, trade_Nprice,trade_hfq_ratio,stock, date_s):  #this trade day price
        if self.volume_gu!=0:
            current_holding_volume_gu = self.i_hfq_tb.get_update_volume_on_hfq_ratio_change \
                (old_hfq_ratio=self.Hratio, new_hfq_ratio=trade_hfq_ratio, old_volume=self.volume_gu)
            potential_sell_cost = self._sell_stock_cost(current_holding_volume_gu, trade_Nprice)
            potential_money_back = current_holding_volume_gu * trade_Nprice - potential_sell_cost
            potential_profit=potential_money_back/self.total_invest-1.0
            return self.buy_times*1.0/self.max_num_invest,potential_profit
        else:
            return 0.0, 0.0

class env_reward_basic:
    def __init__(self,scaler_factor,reward_type):
        self.Sucess_buy = 0.0
        self.Fail_buy = 0.0
        self.Fail_sell = 0.0
        self.No_action =0.0
        self.Tinpai=0.0
        self.scale_factor=scaler_factor
        if reward_type =="scaler":
            self.Success_sell=self.scaler_Success_sell
        elif reward_type =="scaler_clip":
            self.Success_sell = self.scaler_clip_Success_sell
        elif reward_type =="double_scaler_clip":
            self.Success_sell = self.double_scaler_clip_Success_sell
        else:
            assert False, "reward_type Only support scaler, scaler_clip"
    def scaler_Success_sell(self,profit):
        return profit*self.scale_factor

    def scaler_clip_Success_sell(self,profit):
        raw_profit=profit * self.scale_factor
        return 1 if raw_profit>1 else -1 if raw_profit< -1 else raw_profit
        #return profit*self.scale_factor

    def double_scaler_clip_Success_sell(self,profit):
        raw_profit=profit * self.scale_factor*2
        return 1 if raw_profit>1 else -1 if raw_profit< -1 else raw_profit


    def hist_scale(self):
        return -0.3*self.scale_factor, 0.3*self.scale_factor, 0.01*self.scale_factor

