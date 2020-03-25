import os
import random
import numpy as np
from data_common import API_SH_SZ_total_sl,API_SH_sl, API_SZ_sl,API_trade_date,ginfo_one_stock,hfq_toolbox
from data_T5 import R_T5,R_T5_scale,R_T5_balance, R_T5_skipSwh,R_T5_skipSwh_balance
import config as sc
from recorder import record_sim_stock_data
from env_get_data import *
from env_ar import *

def init_gc(lgc):
    global lc
    lc=lgc
    env_get_data_init(lgc)
    ent_ar_init(lgc)
    global Caccount                         #,Creward,Cenv_read_data
    Caccount=globals()[lc.CLN_env_account]

class Simulator_LHPP2V2:
    def __init__(self, data_name, stock,called_by):
        assert not lc.flag_multi_buy
        self.called_by = called_by
        self.data_name = data_name

        self.i_account = Caccount()
        if self.called_by == "explore":
            assert lc.CLN_env_get_data_train == "env_get_data_LHP_train"
            self.i_reward = env_reward_basic(lc.train_reward_scaler_factor,lc.train_reward_type)
            self.i_get_data = globals()[lc.CLN_env_get_data_train](self.data_name, stock,
                            flag_episode_random_start=lc.env_flag_random_start_in_episode_for_explore)
        else:
            assert self.called_by == "eval"
            assert lc.CLN_env_get_data_eval == "env_get_data_LHP_eval"
            self.i_reward = env_reward_basic(lc.eval_reward_scaler_factor,lc.eval_reward_type)
            self.i_get_data = globals()[lc.CLN_env_get_data_eval](self.data_name, stock,
                                flag_episode_random_start=lc.env_flag_random_start_in_episode_for_eval)

        if not self.i_get_data.flag_proper_data_avaliable:
            raise ValueError("{0} {1} does not exisit".format(self.data_name, stock))

        self.step= self.step_OS if lc.P2_current_phase=="Train_Sell" else self.step_OB
        if lc.flag_record_sim:
            self.i_record_sim_stock_data=record_sim_stock_data(os.path.join(sc.base_dir_RL_system,
                                                        lc.RL_system_name,"record_sim"), stock,lc.flag_record_sim)
    def reset(self):
        self.LHP_CH_flag=False
        self.LHP_CHP=0
        self.i_account.reset()  # clear account inform if last period not sold out finally
        state, support_view_dic = self.i_get_data.reset_get_data()
        self._fabricate_av_to_stat(state)
        support_view_dic["action_return_message"] = "from reset"  # True?
        support_view_dic["holding"] = 0
        if lc.flag_record_sim:
            self.i_record_sim_stock_data.saver([state, support_view_dic],support_view_dic["date"])
        return state, support_view_dic

    def step_OS(self, action04):
        if self.LHP_CH_flag and self.LHP_CHP>=lc.LHP-1:
            action_str="sell"
            flag_force_sell = True
        else:
            action_str=lc.action_type_dict[action04]
            flag_force_sell = False
        return self.step_common(action_str,flag_force_sell)


    def step_OB(self, action):
        assert False , "{0} not support OS".format(self.__class__.__name__)

    def step_common(self,action_str,flag_force_sell):
        state, support_view_dic, done_flag = self.i_get_data.next_get_data()
        if lc.flag_record_sim:
            self.i_record_sim_stock_data.saver([state, support_view_dic],support_view_dic["date"])
        if action_str == "buy":  # This is only support single buy
            assert not self.LHP_CH_flag
            assert self.LHP_CHP==0
            support_view_dic["action_taken"] = "Buy"
            flag_operation_result, return_message = self.i_account.buy(support_view_dic["this_trade_day_Nprice"],
                                                                       support_view_dic["this_trade_day_hfq_ratio"],
                                                                       support_view_dic["stock"],
                                                                       support_view_dic["date"])
            if flag_operation_result:
                reward = self.i_reward.Sucess_buy
                self.LHP_CH_flag = True  #start holding
                self.LHP_CHP =1
            else:
                if return_message == "Tinpai":
                    reward = self.i_reward.Tinpai
                elif return_message == "Exceed_limit":
                    reward = self.i_reward.Fail_buy
                else:
                    raise ValueError("unexpected buy return message {0}".format(return_message))
            Done_flag = support_view_dic["last_day_flag"]
        elif action_str == "sell":  # sell
            assert self.LHP_CH_flag

            support_view_dic["action_taken"] = "Sell"
            flag_operation_result, return_message, current_profit = self.i_account.sell(
                support_view_dic["this_trade_day_Nprice"],
                support_view_dic["this_trade_day_hfq_ratio"],
                support_view_dic["stock"],
                support_view_dic["date"])

            self.LHP_CHP += 1
            if flag_operation_result:  # success sell
                reward = self.i_reward.Success_sell(current_profit)
                self.LHP_CH_flag = False
                Done_flag = True
            else:
                if return_message == "Tinpai":
                    reward = self.i_reward.Tinpai
                elif return_message == "No_holding":
                    reward = self.i_reward.Fail_sell
                else:
                    raise ValueError("unexpected sell return message {0}".format(return_message))
                if flag_force_sell:
                    Done_flag = True
                else:
                    Done_flag = support_view_dic["last_day_flag"]
        elif action_str == "no_action":
            support_view_dic["action_taken"] = "No_action"
            if self.LHP_CH_flag:
                self.LHP_CHP += 1
            reward = self.i_reward.No_action
            return_message = "No_action"  # if want no action always no action as result
            Done_flag = support_view_dic["last_day_flag"]

        else:
            raise ValueError("does not support action {0}".format(action_str))

        #assert (holding!=0 and self.LHP_CH_flag) or (holding==0 and not self.LHP_CH_flag),\
        #    "Fail in sanity check account holding and env holding flag holding= {0} self.LHP_CH_flag= {1}  {2}".\
        #        format(holding,self.LHP_CH_flag,action_str)

        state = self._fabricate_av_to_stat(state)
        support_view_dic["action_return_message"] = return_message
        support_view_dic["holding"] = 1 if self.LHP_CH_flag else 0
        support_view_dic["flag_force_sell"] = flag_force_sell   # this is add for trainer to decide whether or not to train or not train using the mask
        return state, reward,Done_flag,support_view_dic

    def data_reset(self): # this is copy from old code at 20200207, fill right, but not tested
        if self.called_by=="explore":
            assert False, "{0}  does not support data_reset as only called by explore process not eval process".format(self.__class__.__name__)
        else:
            assert self.called_by=="eval"
            self.i_get_data.data_reset()

    def _fabricate_av_to_stat(self, state):
        assert not lc.flag_multi_buy, "{0} not support multi buy".format(self.__class__.__name__)
        lhd=[0 for _ in range(lc.LHP+1)]
        lhd[self.LHP_CHP]=1
        state.append(np.array(lhd).reshape(1, -1))
        return state

class Simulator_LHPP2V3(Simulator_LHPP2V2):
    def __init__(self, data_name, stock,called_by):
        Simulator_LHPP2V2.__init__(self, data_name, stock,called_by)

    def reset(self):
        #if self.called_by == "explore":
        #    self.BB_NBDC=0
        self.BB_NBDC = 0
        return Simulator_LHPP2V2.reset(self)

    def step_OS(self, action, flag_debug=False):
        assert False, "{0} not support OS".format(self.__class__.__name__)


    def step_OB(self,action):
        flag_force_sell = False
        flag_wait_to_buy_finished = False
        if not self.LHP_CH_flag:
            if lc.specific_param.BB_NBD!=-1:
                if self.BB_NBDC>=lc.specific_param.BB_NBD:
                    if self.called_by=="explore":
                        action_str = "buy"
                    else:
                        assert self.called_by=="eval"
                        action_str = lc.action_type_dict[action]
                        flag_wait_to_buy_finished= True
                else:
                    action_str = lc.action_type_dict[action]
                    self.BB_NBDC+=1
            else:
                action_str = lc.action_type_dict[action]
                self.BB_NBDC += 1
        else: #elif self.LHP_CH_flag and self.LHP_CHP>=lc.LHP-1:
            if self.LHP_CHP>=lc.LHP-1:
                action_str="sell"
                flag_force_sell = True
            else:
                action_str=lc.action_type_dict[action]
        state, reward,Done_flag,support_view_dic= self.step_common(action_str,flag_force_sell)
        if support_view_dic["action_return_message"] == "Success" and support_view_dic["action_taken"] == "Buy":
            self.BB_NBDC=0
        else:
            if flag_wait_to_buy_finished and self.called_by == "eval":
                Done_flag =True
        return state, reward,Done_flag,support_view_dic




    '''
    def step_OB_fun(self, called_by):
        def step_OB_base(action):
            flag_force_sell = False
            flag_wait_to_buy_finished = False
            if not self.LHP_CH_flag:
                if lc.specific_param.BB_NBD!=-1:
                    if self.BB_NBDC>=lc.specific_param.BB_NBD:
                        if called_by=="explore":
                            action_str = "buy"
                        else:
                            assert called_by=="eval"
                            action_str = lc.action_type_dict[action]
                            flag_wait_to_buy_finished= True
                    else:
                        action_str = lc.action_type_dict[action]
                        self.BB_NBDC+=1
                else:
                    action_str = lc.action_type_dict[action]
                    self.BB_NBDC += 1
            else: #elif self.LHP_CH_flag and self.LHP_CHP>=lc.LHP-1:
                if self.LHP_CHP>=lc.LHP-1:
                    action_str="sell"
                    flag_force_sell = True
                else:
                    action_str=lc.action_type_dict[action]
            state, reward,Done_flag,support_view_dic= self.step_common(action_str,flag_force_sell)
            if support_view_dic["action_return_message"] == "Success" and support_view_dic["action_taken"] == "Buy":
                self.BB_NBDC=0
            else:
                if flag_wait_to_buy_finished and called_by == "eval":
                    Done_flag =True
            return state, reward,Done_flag,support_view_dic
        return step_OB_base
    '''
    '''
    def step_OB_train(self, action):
        flag_force_sell = False
        if not self.LHP_CH_flag:
            if lc.specific_param.BB_NBD!=0:
                if self.BB_NBDC>=lc.specific_param.BB_NBD and lc.specific_param.BB_NBD!=-1:
                    action_str = "buy"
                else:
                    action_str = lc.action_type_dict[action]
                    self.BB_NBDC+=1
            else:
                action_str = lc.action_type_dict[action]
                self.BB_NBDC += 1
        else: #elif self.LHP_CH_flag and self.LHP_CHP>=lc.LHP-1:
            if self.LHP_CHP>=lc.LHP-1:
                action_str="sell"
                flag_force_sell = True
            else:
                action_str=lc.action_type_dict[action]
        state, reward,Done_flag,support_view_dic= self.step_common(action_str,flag_force_sell)
        if support_view_dic["action_return_message"] == "Success" and support_view_dic["action_taken"] == "Buy":
            self.BB_NBDC=0
        return state, reward,Done_flag,support_view_dic

    def step_OB_eval(self, action):
        flag_force_sell = False
        flag_wait_to_buy_finished = False
        if not self.LHP_CH_flag:
            if lc.specific_param.BB_NBD!=0:
                if self.BB_NBDC>=lc.specific_param.BB_NBD and lc.specific_param.BB_NBD!=-1:
                    action_str = lc.action_type_dict[action]
                    flag_wait_to_buy_finished= True
                else:
                    action_str = lc.action_type_dict[action]
                    self.BB_NBDC+=1
            else:
                action_str = lc.action_type_dict[action]
                self.BB_NBDC += 1
        else:
            if self.LHP_CHP>=lc.LHP-1:
                action_str="sell"
                flag_force_sell = True
            else:
                action_str=lc.action_type_dict[action]
        state, reward,Done_flag,support_view_dic= self.step_common(action_str,flag_force_sell)
        if support_view_dic["action_return_message"] == "Success" and support_view_dic["action_taken"] == "Buy":
            self.BB_NBDC=0
        elif flag_wait_to_buy_finished:
            Done_flag =True
        else:
            pass
        return state, reward,Done_flag,support_view_dic
    '''

class Simulator_LHPP2V5(Simulator_LHPP2V3):
    def __init__(self, data_name, stock,called_by):
        Simulator_LHPP2V3.__init__(self, data_name, stock,called_by)


    def step_OB(self,action):
        if action==4:  # dirty solution to use "no_action" and change the action related item in support_view after return
            state, reward, Done_flag, support_view_dic = self.step_common("no_action", False)
            support_view_dic["action_taken"] = "No_trans"
            support_view_dic["action_return_message"] = "No_trans"
        else:
            state, reward,Done_flag,support_view_dic = Simulator_LHPP2V3.step_OB(self,action)
        return state, reward,Done_flag,support_view_dic

