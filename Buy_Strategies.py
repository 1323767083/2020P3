import random
from State import *
class Buy_Strategies:
    def __init__(self,lc):
        pass
    def Buy_Strategy_one_time(self, dateI, num_stock_could_invest, l_a_OB, l_a_OS, l_holding, L_Eval_Profit_low_flag):
        Sidxs_all = set(list(range(len(l_a_OB))))
        Sidxs_OB_buy = set([idx for idx, action in enumerate(l_a_OB) if action == 0])
        Sidxs_holding = set([idx for idx, holding_flag in enumerate(l_holding) if holding_flag])
        Sidxs_OS_sell = set([idx for idx, action in enumerate(l_a_OS) if action == 2])
        Sidxs_Eval_Low = set([idx for idx, flag in enumerate(L_Eval_Profit_low_flag) if flag])

        Sidxs_buy = (Sidxs_OB_buy - Sidxs_Eval_Low) & (Sidxs_all - Sidxs_holding)
        Sidxs_sell = (Sidxs_OS_sell | Sidxs_Eval_Low) & Sidxs_holding
        Sidxs_not_buy_due_limit = set()  # empty set should create by set() not {}
        if len(Sidxs_buy) > num_stock_could_invest:
            lidxs_0 = list(Sidxs_buy)
            random.shuffle(lidxs_0)
            Sidxs_buy = set(lidxs_0[:num_stock_could_invest])
            Sidxs_not_buy_due_limit = set(lidxs_0[num_stock_could_invest:])
        assert len(Sidxs_buy & Sidxs_sell) == 0, "Sidxs_buy {0} & Sidxs_sell {1}".format(Sidxs_buy, Sidxs_sell)
        Sidxs_no_action_1 = Sidxs_all - Sidxs_holding - Sidxs_buy
        Sidxs_no_action_3 = Sidxs_holding - Sidxs_sell
        assert len(Sidxs_buy) + len(Sidxs_no_action_1) + len(Sidxs_sell) + len(Sidxs_no_action_3) == len(Sidxs_all)
        l_a = [0 for _ in range(len(l_a_OB))]
        for action, s_idx in enumerate([Sidxs_buy, Sidxs_no_action_1, Sidxs_sell, Sidxs_no_action_3]):
            for idx in list(s_idx):
                l_a[idx] = action

        l_not_buy_due_low_profit = list(Sidxs_Eval_Low & Sidxs_buy)
        l_sell_due_low_profit = list(Sidxs_Eval_Low & Sidxs_sell)
        l_not_buy_due_limit = list(Sidxs_not_buy_due_limit)
        l_not_buy_due_low_profit.sort()
        l_sell_due_low_profit.sort()
        l_not_buy_due_limit.sort()

        l_multibuy = list()
        l_multibuy.sort()

        l_ADlog = [dateI,
                   "_".join(map(str, l_not_buy_due_limit)),
                   "_".join(map(str, l_not_buy_due_low_profit)),
                   "_".join(map(str, l_sell_due_low_profit)),
                   "_".join(map(str, l_multibuy))]
        return l_a,l_ADlog

    def Buy_Strategy_multi_time(self, dateI, num_stock_could_invest, l_a_OB, l_a_OS, l_holding, L_Eval_Profit_low_flag):
        Sidxs_all = set(list(range(len(l_a_OB))))
        Sidxs_OB_buy = set([idx for idx, action in enumerate(l_a_OB) if action == 0])
        Sidxs_holding = set([idx for idx, holding_flag in enumerate(l_holding) if holding_flag])
        Sidxs_OS_sell = set([idx for idx, action in enumerate(l_a_OS) if action == 2])
        Sidxs_Eval_Low = set([idx for idx, flag in enumerate(L_Eval_Profit_low_flag) if flag])

        Sidxs_buy = (Sidxs_OB_buy - Sidxs_Eval_Low)
        Sidxs_not_buy_due_limit = set()
        if len(Sidxs_buy) > num_stock_could_invest:
            lidxs_0 = list(Sidxs_buy)
            random.shuffle(lidxs_0)
            Sidxs_buy = set(lidxs_0[:num_stock_could_invest])
            Sidxs_not_buy_due_limit = set(lidxs_0[num_stock_could_invest:])

        # Sidxs_sell = ((Sidxs_OS_sell | Sidxs_Eval_Low)-Sidxs_buy)& Sidxs_holding
        Sidxs_sell = (Sidxs_OS_sell | Sidxs_Eval_Low) & Sidxs_holding - Sidxs_buy

        assert len(Sidxs_buy & Sidxs_sell) == 0, "Sidxs_buy {0} & Sidxs_sell {1}".format(Sidxs_buy, Sidxs_sell)
        Sidxs_no_action_1 = Sidxs_all - Sidxs_holding - Sidxs_buy
        Sidxs_no_action_3 = Sidxs_holding - Sidxs_sell - Sidxs_buy  # there is part of holding in buy action due to multi buy
        assert len(Sidxs_buy) + len(Sidxs_no_action_1) + len(Sidxs_sell) + len(Sidxs_no_action_3) == len(Sidxs_all)

        l_a = [0 for _ in range(len(l_a_OB))]
        for action, s_idx in enumerate([Sidxs_buy - Sidxs_holding, Sidxs_no_action_1, Sidxs_sell, Sidxs_no_action_3]):
            for idx in list(s_idx):
                l_a[idx] = self.Fabricate_V3EvalCC_MultiplexAction(action=action, PSS_action=0)
        for idx in list(Sidxs_buy & Sidxs_holding):
            l_a[idx] = self.Fabricate_V3EvalCC_MultiplexAction(action=0, PSS_action=1)

        l_not_buy_due_low_profit = list(Sidxs_Eval_Low & Sidxs_buy)
        l_sell_due_low_profit = list(Sidxs_Eval_Low & Sidxs_sell)
        l_not_buy_due_limit = list(Sidxs_not_buy_due_limit)
        l_multibuy = list(Sidxs_buy & Sidxs_holding)
        l_not_buy_due_low_profit.sort()
        l_sell_due_low_profit.sort()
        l_not_buy_due_limit.sort()
        l_multibuy.sort()

        l_ADlog = [dateI,
                   "_".join(map(str, l_not_buy_due_limit)),
                   "_".join(map(str, l_not_buy_due_low_profit)),
                   "_".join(map(str, l_sell_due_low_profit)),
                   "_".join(map(str, l_multibuy))]

        return l_a,l_ADlog


    def Buy_Strategy_multi_time_Direct_sell(self, dateI, num_stock_could_invest, l_a_OB, l_a_OS, l_holding,L_Eval_Profit_low_flag):
        #import pickle,os
        #pickle.dump([dateI, num_stock_could_invest, l_a_OB, l_a_OS, l_holding], open(os.path.join("/home/rdchujf/","{0}.pickle".format(dateI)),"wb"))
        Sidxs_all = set(list(range(len(l_a_OB))))
        Sidxs_OB_buy = set([idx for idx, action in enumerate(l_a_OB) if action == 0])
        Sidxs_holding = set([idx for idx, holding_flag in enumerate(l_holding) if holding_flag])
        Sidxs_OS_sell = set([idx for idx, action in enumerate(l_a_OS) if action == 2])
        Sidxs_sell = (Sidxs_OS_sell - Sidxs_OB_buy) & Sidxs_holding

        Sidxs_buy = Sidxs_OB_buy - Sidxs_holding
        assert len(Sidxs_sell & Sidxs_buy)==0, "{0} {1}".format(Sidxs_sell , Sidxs_buy)
        Sidxs_not_buy_due_limit = set()
        if len(Sidxs_buy) > num_stock_could_invest:
            lidxs_0 = list(Sidxs_buy)
            random.shuffle(lidxs_0)
            Sidxs_buy = set(lidxs_0[:num_stock_could_invest])
            Sidxs_not_buy_due_limit = set(lidxs_0[num_stock_could_invest:])

        assert len(Sidxs_buy & Sidxs_sell) == 0, "Sidxs_buy {0} & Sidxs_sell {1}".format(Sidxs_buy, Sidxs_sell)
        Sidxs_no_action_1 = Sidxs_all - Sidxs_holding - Sidxs_buy
        Sidxs_no_action_3 = Sidxs_holding - Sidxs_sell
        assert len(Sidxs_buy) + len(Sidxs_no_action_1) + len(Sidxs_sell) + len(Sidxs_no_action_3) == len(Sidxs_all)

        l_a = [0 for _ in range(len(l_a_OB))]
        for action, s_idx in enumerate([Sidxs_buy, Sidxs_no_action_1, Sidxs_sell, Sidxs_no_action_3]):
            for idx in list(s_idx):
                l_a[idx] = self.Fabricate_V3EvalCC_MultiplexAction(action=action, PSS_action=0)

        l_not_buy_due_low_profit = list()
        l_sell_due_low_profit = list()
        l_not_buy_due_limit = list(Sidxs_not_buy_due_limit)
        l_multibuy = list(Sidxs_OB_buy & Sidxs_holding)  # here multibuy actually change to keep not sell in direct sell
        l_not_buy_due_low_profit.sort()
        l_sell_due_low_profit.sort()
        l_not_buy_due_limit.sort()
        l_multibuy.sort()

        l_ADlog = [dateI,
                   "_".join(map(str, l_not_buy_due_limit)),
                   "_".join(map(str, l_not_buy_due_low_profit)),
                   "_".join(map(str, l_sell_due_low_profit)),
                   "_".join(map(str, l_multibuy))]

        return l_a,l_ADlog

    #Todo absolet function should consider to delete the multplex
    def Extract_V3EvalCC_MultiplexAction(self, MultiplexAction):
        action = MultiplexAction % 10
        PSS_action = MultiplexAction // 10 % 10
        return action, PSS_action

    def Fabricate_V3EvalCC_MultiplexAction(self, action, PSS_action):
        return PSS_action * 10 + action

