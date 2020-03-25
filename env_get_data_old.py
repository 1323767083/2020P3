import random
import numpy as np
from data_T5 import R_T5,R_T5_scale,R_T5_balance,R_T5_skipSwh,R_T5_skipSwh_balance
def env_get_data_init(lgc):
    global lc, Cenv_read_data
    lc=lgc
    Cenv_read_data = globals()[lc.CLN_env_read_data]

class env_get_data_base_base:
    def __init__(self, data_name, stock,length_threadhold,flag_episode_randon_start):
        #length_threadhold=50
        self.data_name=data_name
        self.stock=stock
        self.length_threadhold=length_threadhold
        self.flag_episode_random_start= flag_episode_randon_start
        self.i_sd=Cenv_read_data(self.data_name, self.stock)
        #self.flag_proper_data_avaliable=False
        self.flag_just_initialed=False
        if self.i_sd.flag_prepare_data_ready:
            self.valid_list_period_idx = self._get_periods_over_threadhold()
            self.flag_proper_data_avaliable=self._check_proper_data_avaliable()
            self.flag_just_initialed = True
        else:
            self.flag_proper_data_avaliable=False

    # these period_idx operation related to valid_preiod_idx which over threadhold
    def _get_periods_over_threadhold(self):
        return [idx for idx, np_date_s in enumerate(self.i_sd.data.l_np_date_s) if len(np_date_s)>self.length_threadhold]

    def _check_proper_data_avaliable(self):
        return True if len(self.valid_list_period_idx)>0 else False

    def _get_next_period_idx(self,idx):
        assert self.flag_proper_data_avaliable
        idx_idx=self.valid_list_period_idx.index(idx)
        if idx_idx==len(self.valid_list_period_idx)-1:
            raise ValueError("idx_idx is equal to len(self.valid_list_period_idx)-1")
        else:
            return self.valid_list_period_idx[idx_idx + 1]
    def _check_next_period_idx(self,idx):
        assert self.flag_proper_data_avaliable
        idx_idx=self.valid_list_period_idx.index(idx)
        if idx_idx==len(self.valid_list_period_idx)-1:
            return False
        else:
            return True

    # these period_idx operation not related valid_period_idx
    def _set_period_idx(self,idx):
        assert self.flag_proper_data_avaliable
        self.period_index=idx
        if self.flag_episode_random_start:
            # need rest >50 so -50 -1  ;-len-1 is for index length converting
            if len (self.i_sd.data.l_np_date_s[self.period_index])<=self.length_threadhold+1:
                self.idx_in_period = random.randrange(0, len(self.i_sd.data.l_np_date_s[self.period_index]) - 1 - self.length_threadhold/2)
            else:
                assert len(self.i_sd.data.l_np_date_s[self.period_index])-1-self.length_threadhold>0,"{0} Period_idx:{1}  len:{2}".format(self.stock,self.period_index,len(self.i_sd.data.l_np_date_s[self.period_index]) )
                self.idx_in_period=random.randrange(0, len(self.i_sd.data.l_np_date_s[self.period_index])-1-self.length_threadhold)
        else:
            self.idx_in_period=0
        self.period_end_idx=len(self.i_sd.data.l_np_date_s[self.period_index])-1

    # these method are for calling from exteral
    def data_reset(self):
        assert self.flag_proper_data_avaliable
        self.flag_just_initialed=True

    def reset_get_data(self):
        assert self.flag_proper_data_avaliable
        if self.flag_just_initialed:
            period_idx_to_set=self.valid_list_period_idx[0]
            self.flag_just_initialed=False
            flag_all_period_explored = False
        else:
            if self._check_next_period_idx(self.period_index):
                period_idx_to_set=self._get_next_period_idx(self.period_index)
                flag_all_period_explored = False
            else:
                #set all explroed flag to be true and reset data
                period_idx_to_set = self.valid_list_period_idx[0]
                self.flag_just_initialed = False
                flag_all_period_explored = True

        self._set_period_idx(period_idx_to_set)
        state, support_view_dic = self.i_sd.read_one_day_data_by_index(self.period_index,self.idx_in_period)
        done_flag = support_view_dic["last_day_flag"]
        assert not done_flag, "{0} period_idx:{1} idx in period: {2}".format(self.stock, self.period_index, self.idx_in_period)
        support_view_dic["period_idx"]=self.period_index
        support_view_dic["idx_in_period"] = self.idx_in_period

        support_view_dic["flag_all_period_explored"] = True if flag_all_period_explored else False

        #return state, support_view_dic, False
        return state, support_view_dic

    def next_get_data(self):
        assert self.flag_proper_data_avaliable

        if self.idx_in_period <self.period_end_idx:
            self.idx_in_period = self.idx_in_period+1
        else:
            raise ValueError("unexpected situation {0} period_idx = {1} idx_in_period={2} exceed period_end_idx={3}".
                             format(self.stock,self.period_index, self.idx_in_period, self.period_end_idx))

        state, support_view_dic = self.i_sd.read_one_day_data_by_index(self.period_index,self.idx_in_period)
        if self.idx_in_period == self.period_end_idx:

            if not support_view_dic["last_day_flag"]:
                print "**********************************************************************"
                print "{0} end correct".format(self.stock)
                print "**********************************************************************"

            support_view_dic["last_day_flag"]=True

        done_flag = support_view_dic["last_day_flag"]
        support_view_dic["period_idx"]=self.period_index
        support_view_dic["idx_in_period"] = self.idx_in_period
        return state, support_view_dic, done_flag

'''To removed __system_type_remove__
class env_get_data_base(env_get_data_base_base):
    def __init__(self, data_name, stock, flag_episode_randon_start):
        length_threadhold=50
        env_get_data_base_base.__init__(self, data_name, stock, length_threadhold,flag_episode_randon_start)

class env_get_data_LHP_eval(env_get_data_base_base):
    def __init__(self, data_name, stock, flag_episode_randon_start):
        length_threadhold=lc.LHP+1
        assert not flag_episode_randon_start, "{0} only support flag_episode_randon_start=False".format(self.__class__.__name__)
        env_get_data_base_base.__init__(self, data_name, stock, length_threadhold,flag_episode_randon_start)
'''



class env_get_data_LHP_train(env_get_data_base_base):
    def __init__(self, data_name, stock, flag_episode_randon_start):
        length_threadhold=lc.LHP+1
        assert flag_episode_randon_start, "{0} only support flag_episode_randon_start=True".format(self.__class__.__name__)
        env_get_data_base_base.__init__(self, data_name, stock, length_threadhold,flag_episode_randon_start)

        if self.flag_proper_data_avaliable:
            list_valid_period_len= [len(self.i_sd.data.l_np_date_s[valid_period_idx])-self.length_threadhold
                                    for valid_period_idx in self.valid_list_period_idx]
            valid_period_len_sum= sum(list_valid_period_len)*1.0
            self.list_valid_period_prob=[lenth/valid_period_len_sum for lenth in list_valid_period_len]


    def _get_next_period_idx(self,idx):
        assert False, "method _get_next_period_idx not used in class env_get_data_LHP_train"


    def _check_next_period_idx(self,idx):
        assert False, "method _check_next_period_idx not used in class env_get_data_LHP_train"

    def _set_period_idx(self,idx):
        assert False, "method _set_period_idx not used in class env_get_data_LHP_train"

    # these method are for calling from exteral
    def data_reset(self):
        assert False, "method data_reset not used in class env_get_data_LHP_train"

    # these period_idx operation not related valid_period_idx
    def _random_set_period_idx_old(self):
        assert self.flag_proper_data_avaliable
        idx_valid_list_period_idx=random.randrange(0, len(self.valid_list_period_idx))
        self.period_index=self.valid_list_period_idx[idx_valid_list_period_idx]
        assert len(self.i_sd.data.l_np_date_s[
                       self.period_index]) - self.length_threadhold >0, "{0} Period_idx:{1}  len:{2}".format(
            self.stock, self.period_index, len(self.i_sd.data.l_np_date_s[self.period_index]))
        self.idx_in_period = random.randrange(0, len(
            self.i_sd.data.l_np_date_s[self.period_index]) - self.length_threadhold)

        self.period_end_idx=len(self.i_sd.data.l_np_date_s[self.period_index])-1
    def _random_set_period_idx(self):
        assert self.flag_proper_data_avaliable

        self.period_index =np.random.choice(self.valid_list_period_idx, p=self.list_valid_period_prob)
        assert len(self.i_sd.data.l_np_date_s[
                       self.period_index]) - self.length_threadhold >0, "{0} Period_idx:{1}  len:{2}".format(
            self.stock, self.period_index, len(self.i_sd.data.l_np_date_s[self.period_index]))
        self.idx_in_period = random.randrange(0, len(
            self.i_sd.data.l_np_date_s[self.period_index]) - self.length_threadhold)

        self.period_end_idx=len(self.i_sd.data.l_np_date_s[self.period_index])-1

    def reset_get_data(self):
        assert self.flag_proper_data_avaliable
        self._random_set_period_idx()
        state, support_view_dic = self.i_sd.read_one_day_data_by_index(self.period_index,self.idx_in_period)
        done_flag = support_view_dic["last_day_flag"]
        assert not done_flag, "{0} period_idx:{1} idx in period: {2}".format(self.stock, self.period_index, self.idx_in_period)
        support_view_dic["period_idx"]=self.period_index
        support_view_dic["idx_in_period"] = self.idx_in_period
        support_view_dic["flag_all_period_explored"] = False
        return state, support_view_dic

    def next_get_data(self):
        assert self.flag_proper_data_avaliable
        if self.idx_in_period <self.period_end_idx:
            self.idx_in_period = self.idx_in_period+1
        else:
            raise ValueError("unexpected situation {0} period_idx = {1} idx_in_period={2} exceed period_end_idx={3}".
                             format(self.stock,self.period_index, self.idx_in_period, self.period_end_idx))
        state, support_view_dic = self.i_sd.read_one_day_data_by_index(self.period_index,self.idx_in_period)
        if self.idx_in_period == self.period_end_idx:
            if not support_view_dic["last_day_flag"]:
                print "**********************************************************************"
                print "{0} end correct".format(self.stock)
                print "**********************************************************************"
            support_view_dic["last_day_flag"]=True

        done_flag = support_view_dic["last_day_flag"]
        support_view_dic["period_idx"]=self.period_index
        support_view_dic["idx_in_period"] = self.idx_in_period
        return state, support_view_dic, done_flag

class env_get_data_LHP_eval_nc(env_get_data_LHP_train):  # nc means not continue this is to make the eval simulator work as train simulator but has flag_all_period_explored set
    def __init__(self, data_name, stock, flag_episode_randon_start):
        env_get_data_LHP_train.__init__(self, data_name, stock, flag_episode_randon_start)
        self.eval_reset_count=0

    def reset_get_data(self):
        state, support_view_dic=env_get_data_LHP_train.reset_get_data(self)
        self.eval_reset_count +=1
        #if self.eval_reset_count%50==0:
        #    print self.stock, self.eval_reset_count
        #if self.eval_reset_count == lc.evn_eval_rest_total_times:
        if self.eval_reset_count > lc.evn_eval_rest_total_times:
            support_view_dic["flag_all_period_explored"] = True
            self.eval_reset_count = 0
        else:
            support_view_dic["flag_all_period_explored"] = False
        return state, support_view_dic

    def data_reset(self):
        assert self.flag_proper_data_avaliable
        self.flag_just_initialed=True
