import random,os
import pandas as pd
from env import Simulator_LHPP2V2,Simulator_LHPP2V3,Simulator_LHPP2V5
from data_common import API_SH_sl
class lconfig:
    def __init__(self):
        self.CLN_simulator=""
        self.CLN_GenStockList =""
        self.system_working_dir =""


def init_gc(lgc):
    global lc
    lc=lconfig()
    for key in list(lc.__dict__.keys()):
        lc.__dict__[key] = lgc.__dict__[key]
    global Csimulator,CGenStockList
    Csimulator =globals()[lc.CLN_simulator]
    CGenStockList=globals()[lc.CLN_GenStockList]



class divide_sl_to_process:
    def __init__(self,data_name):
        self.data_name=data_name

    def get_sl_full_divided(self, data_base, data_index, num_to_divided):
        total_sl=CGenStockList(self.data_name).load_stock_list(data_base, data_index)
        checked_sl = self.sanity_check_stock_list(total_sl, data_base, data_index)
        random.shuffle(checked_sl)
        num_stock_per_group= len(checked_sl) // num_to_divided
        divided_sl=[]
        start_idx=0
        for _ in range(num_to_divided):
            divided_sl.append(checked_sl[start_idx:start_idx + num_stock_per_group])
            start_idx = start_idx + num_stock_per_group
        check_idx=0
        while start_idx<len(checked_sl):
            divided_sl[check_idx].append(checked_sl[start_idx])
            start_idx +=1
            if start_idx == len(checked_sl):
                break
            check_idx =0 if check_idx == num_to_divided-1 else check_idx+1
        return divided_sl


    def get_sl_fix_length(self,data_base, data_index, num_to_divided, length_per_list):
        total_list = CGenStockList(self.data_name).load_stock_list(data_base, data_index)
        checked_list = self.sanity_check_stock_list (total_list, data_base,data_index)
        random.shuffle(checked_list)

        assert len(checked_list) > num_to_divided* length_per_list, \
            "Not enough stock to divide total number of stock({0})< number to divided({1}) *length per list({2})".\
                format(len(checked_list),num_to_divided, length_per_list)
        l_sl = []
        for idx in range(num_to_divided):
            l_sl.append(checked_list[idx * length_per_list:(idx + 1) * length_per_list])
        return l_sl



    def sanity_check_stock_list(self, stock_list, data_base, data_index):
        calledby="explore"  # "eval" or "explore"
        remove_list_fnwp = os.path.join(lc.system_working_dir,
            "{0}_remove_stock_list_b{1}i{2}.csv".format(self.data_name,data_base, data_index))
        if not os.path.exists(remove_list_fnwp):
            remove_stock_list = []
            for stock in stock_list:
                try:
                    Csimulator(self.data_name,stock,calledby)
                except Exception as e:
                    remove_stock_list.append(stock)
                    print("add {0} to removed stock list".format(stock))
            df_remove_stock_list = pd.DataFrame(remove_stock_list, columns=["stock"])
            df_remove_stock_list.to_csv(remove_list_fnwp, index=False)
            print("store remove list to {0}".format(remove_list_fnwp))
        else:
            df_remove_stock_list = pd.read_csv(remove_list_fnwp, header=0, names=["stock"])
            remove_stock_list = df_remove_stock_list["stock"].tolist()
            print("load remove list from {0}".format(remove_list_fnwp))

        return list(set(stock_list) - set(remove_stock_list))

def prepare_eval_sl(l_eval_data_name, data_base, data_index, length_per_list):
    T5_count = l_eval_data_name.count("T5")
    T5_V2__count = l_eval_data_name.count("T5_V2_")
    assert T5_count + T5_V2__count == len(l_eval_data_name)
    i_d_sl = divide_sl_to_process("T5")
    T5_eval_sl = i_d_sl.get_sl_fix_length(data_base, data_index, T5_count, length_per_list)

    i_d_sl = divide_sl_to_process("T5_V2_")
    T5_V2__eval_sl = i_d_sl.get_sl_fix_length(data_base, data_index,  T5_V2__count, length_per_list)

    l_eval_sl = []
    T5_eval_sl_idx = 0
    T5_V2__eval_sl_idx = 0
    for data_name in l_eval_data_name:
        if data_name == "T5":
            l_eval_sl.append(T5_eval_sl[T5_eval_sl_idx])
            T5_eval_sl_idx += 1
        else:
            assert data_name == "T5_V2_"
            l_eval_sl.append(T5_V2__eval_sl[T5_V2__eval_sl_idx])
            T5_V2__eval_sl_idx += 1
    return l_eval_sl

