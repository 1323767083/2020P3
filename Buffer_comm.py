import numpy as np
import sys
import random
from action_comm import actionOBOS
from State import *
# brain train buffer/ worker eval loss buffer
class brain_buffer:
    def __init__(self, lc):
        self.lc=lc
        self.train_queue = [[] for _ in range(lc.Buffer_nb_Features + 2 * 2)]  # each state have 3 substate so add 2
    def train_get(self, size):
        if len(self.train_queue[0]) < size:
            return False, None, None , None, None, None, None  , None, None, None, None
        t_s_lv, t_s_sv, t_s_av, t_a, t_r, t_s__lv, t_s__sv, t_s__av, t_flag_done , t_inform = self.train_queue
        s_lv = t_s_lv[0:size]
        s_sv = t_s_sv[0:size]
        s_av = t_s_av[0:size]
        a = t_a[0:size]
        r = t_r[0:size]
        s__lv = t_s__lv[0:size]
        s__sv = t_s__sv[0:size]
        s__av = t_s__av[0:size]
        flag_done = t_flag_done[0:size]
        inform = t_inform[0:size]

        del t_s_lv[0:size]
        del t_s_sv[0:size]
        del t_s_av[0:size]
        del t_a[0:size]
        del t_r[0:size]
        del t_s__lv[0:size]
        del t_s__sv[0:size]
        del t_s__av[0:size]
        del t_flag_done[0:size]
        del t_inform[0:size]

        return True,s_lv, s_sv, s_av, a, r, s__lv, s__sv, s__av, flag_done , inform

    def _train_push_one(self, s, a, r, s_, flag_done, inform):

        self.train_queue[0].append(s[0])
        self.train_queue[1].append(s[1])
        self.train_queue[2].append(s[2])
        self.train_queue[3].append(a)
        self.train_queue[4].append(r)
        self.train_queue[5].append(s_[0])
        self.train_queue[6].append(s_[1])
        self.train_queue[7].append(s_[2])
        self.train_queue[8].append(flag_done)
        self.train_queue[9].append(inform)

    def train_push_many(self, in_buffer):
        for iterm in in_buffer:
            s, a, r, s_, flag_done, inform = iterm
            self._train_push_one(s, a, r, s_, flag_done, inform)

    def get_buffer_size(self):
        return len(self.train_queue[0])

    def reset_tb(self):
        for t_idx in range(self.lc.Buffer_nb_Features + 2 * 2):
            del self.train_queue[t_idx][:]

class brain_buffer_reuse:
    def __init__(self, lc):
        self.lc=lc
        self.tq_numb_col        =   lc.Buffer_nb_Features + 2 * 2+1
        self.tq_count_idx       =   self.tq_numb_col-1
        self.tq_out_numb_col    =   self.tq_numb_col-1
        self.reuse_times        =   lc.brain_buffer_reuse_times
        self.tq = [[] for _ in range(self.tq_numb_col)]

    def train_get(self, size):
        if len(self.tq[0])>=size:
            l_selected_idx=[]
            for opt_reuse_count in reversed(list(range(self.reuse_times))):
                lgz = [idx for idx, count in enumerate(self.tq[self.tq_count_idx]) if count ==opt_reuse_count +1]
                if len(l_selected_idx) + len(lgz)>=size:
                    random.shuffle(lgz)
                    l_selected_idx.extend(lgz[:size-len(l_selected_idx)])
                    break
                else:
                    l_selected_idx.extend(lgz)
                    continue
            else:
                assert False, "should not reach here size= {0} len of l_selected_idx = {1)".format(size,len(l_selected_idx))

            lresult=[]
            assert len(l_selected_idx)==size
            for tq_idx in range(self.tq_out_numb_col):
                lresult.append(list(map(self.tq[tq_idx].__getitem__, l_selected_idx)))
            ltoremove_idx=[]
            for tq_idx_idx in l_selected_idx:
                self.tq[self.tq_count_idx][tq_idx_idx]-=1
                if self.tq[self.tq_count_idx][tq_idx_idx] ==0:
                    ltoremove_idx.append(tq_idx_idx)
            for tq_idx_idx in sorted(ltoremove_idx, reverse=True):   # remove from reversed sorted index
                for tq_idx in range(self.tq_numb_col):
                    del self.tq[tq_idx][tq_idx_idx]
            return [True] + lresult
        else:
            return [False] + [None for _ in range(self.tq_out_numb_col)]


    def reset_tb(self):
        for t_idx in range(self.tq_numb_col):
            del self.tq[t_idx][:]


    def _train_push_one(self, s, a, r, s_, flag_done, inform):

        self.tq[0].append(s[0])
        self.tq[1].append(s[1])
        self.tq[2].append(s[2])
        self.tq[3].append(a)
        self.tq[4].append(r)
        self.tq[5].append(s_[0])
        self.tq[6].append(s_[1])
        self.tq[7].append(s_[2])
        self.tq[8].append(flag_done)
        self.tq[9].append(inform)
        self.tq[10].append(self.reuse_times)
        assert self.tq_count_idx==10


    def train_push_many(self, in_buffer):
        for iterm in in_buffer:
            s, a, r, s_, flag_done, inform = iterm
            self._train_push_one(s, a, r, s_, flag_done, inform)

    def get_buffer_size(self):
        return len(self.tq[0])



#worker with eval / worker with eval prepare data send to server
class buffer_to_train:
    def __init__(self, lc,num_stocks_in_one_group):
        self.lc=lc
        self.num_stocks_in_one_group = num_stocks_in_one_group
        self.train_buffer_to_server = []
        self.fd_buffer = []
        for _ in range(num_stocks_in_one_group):
            self.fd_buffer.append(globals()[self.lc.CLN_TDmemory](self.lc,self.train_buffer_to_server))

    def add_one_record(self, group_idx, s, a_onehot, np_r, s_, done, support_view_dic):
        self.fd_buffer[group_idx].add_to_train_buffer_to_server(s, a_onehot, np_r, s_, done, support_view_dic)

    def get_len_train_buffer_to_server(self):
        return len(self.train_buffer_to_server)

    def get_train_buffer_to_server(self):
        return self.train_buffer_to_server

    def empty_train_buffer_to_server(self):
        del self.train_buffer_to_server[:]

    def get_train_buffer_spicify_size(self, size):
        assert len(self.train_buffer_to_server)>=size
        buffer_out=self.train_buffer_to_server[:size]
        del self.train_buffer_to_server[:size]
        return buffer_out

class buffer_series:
    def __init__(self):
        self.max=sys.maxsize
        self.bs=0
    #common
    def get_current(self):
        return self.bs

    #client side
    def set_get_next(self):
        self.bs=self.bs+1 if self.bs<self.max else 0
        return self.bs
    ## server side
    def set(self, bs_to_set):
        self.bs=bs_to_set
    def valify(self,bs_to_varify):
        return  bs_to_varify ==self.bs+1 if self.bs<self.max else bs_to_varify==0

class TD_memory_integrated:
    def __init__(self, lc, output_buffer):
        self.memory = []  # used for n_step return
        self.lc=lc
        self.FDn = self.lc.TDn
        self.gamma = self.lc.Brain_gamma
        self.output_buffer = output_buffer
        self.i_actionOBOS=actionOBOS(self.lc.train_action_type)
        self.iavh=AV_Handler(self.lc)
        #self.get_verified_record =getattr(self,lgc.TD_get_verified_record)
        if "V2" in self.lc.system_type:
            self.get_verified_record=self.V2_verified_train_record
        elif "V3" in self.lc.system_type:
            self.get_verified_record = self.V3_verified_train_record
        else:
            assert False, "{0} only support V2 V3".format(self.__class__.__name__)

    def get_sample(self, memory, n):
        s, a, _, _, done, support_view_dic = memory[0]
        _, _, _, s_, _, _support_view_dic = memory[n - 1]
        return s, a, None, s_, done, support_view_dic,_support_view_dic

    def get_accumulate_R(self, Num_record):
        AccR=0
        for idx in list(reversed(list(range(Num_record)))):
            AccR=self.memory[idx][2]+AccR*self.gamma
        return AccR

    def V2_verified_train_record(self):
        _, _, _, Ls_, Ldone, Lsupport_view_dic = self.memory[-1]
        flag_in_HP, idx_HP=self.iavh.get_HP_status_On_S_(Ls_[2].reshape((-1,)))
        if not flag_in_HP:
            if idx_HP == -1:       #NB phase not finished
                del self.memory[:]
                return False
            else:
                assert idx_HP>=0,  idx_HP  #HP phase finished
                del self.memory[:-(idx_HP+1)]
                #self.iavh.set_final_record_AV(self.memory[-1][0][2][0])  #-1 last record 0 means state in memory 2 means AV 0 means av item has shape (0,13)
                self.iavh.set_final_record_AV(self.memory[-1][3][2][0])  # -1 last record 3 means state_ in memory 2 means AV 0 means av item has shape (0,13)
                return True
        else:                       #HP phase not finished
            del self.memory[:]
            return False

    def V3_verified_train_record(self):
        _, _, _, Ls_, _, Lsupport_view_dic = self.memory[-1]
        flag_in_HP, idx_HP=self.iavh.get_HP_status_On_S_(Ls_[2].reshape((-1,)))
        if not flag_in_HP:
            if idx_HP==-1:                  #NB phase not finished
                del self.memory[:]
                return False
                #self.iavh.set_final_record_AV(self.memory[-1][3][2][0])  # -1 last record 3 means state_ in memory 2 means AV 0 means av item has shape (0,13)
                #return True

            else:
                assert idx_HP>=0            #HP phase finished
                raw_reward = self.memory[-1][2]
                if raw_reward==0:
                    del self.memory[:]
                    return False
                del self.memory[-(idx_HP + 1):]
                self.memory[-1][2] = raw_reward * self.gamma ** (idx_HP + 1)
                del self.memory[:-1]  # this is to remove all the non action as FDn is only 1, so rest no action reward is 0

                #self.iavh.set_final_record_AV(self.memory[-1][0][2][0])
                self.iavh.set_final_record_AV(self.memory[-1][3][2][0])  # -1 last record 3 means state_ in memory 2 means AV 0 means av item has shape (0,13)
                return True
        else:                               #HP phase not finished
            assert idx_HP==self.lc.LHP
            assert Lsupport_view_dic["action_return_message"] == "Tinpai", \
                " V3 has forece sell in Explore so only fail finish HP is Tinpai"
            del self.memory[:]  # as it is tinpai so can not avoid should not punish
            return False

    def add_to_train_buffer_to_server(self, si, aa04, r, si_, done, support_view_dic):
        self.memory.append([si, aa04, r, si_, done, support_view_dic])
        if done:
            if not self.get_verified_record():
                return
            while len(self.memory) > 0:
                n = len(self.memory)
                SdisS_=self.FDn if n >= self.FDn else n
                s, aa04, _, s_, train_done, support_view_dic, _support_view_dic = self.get_sample(self.memory, SdisS_)
                adj_r = self.get_accumulate_R(SdisS_)
                np_adj_r = np.array([[adj_r]])
                adjust_a = self.i_actionOBOS.I_TD_buffer(aa04)
                support_view_dic["_support_view_dic"] = dict(_support_view_dic)
                support_view_dic["SdisS_"]=SdisS_
                np_support_view = np.array([[support_view_dic]])
                self.output_buffer.append([s, adjust_a, np_adj_r, s_, train_done, np_support_view])
                self.memory.pop(0)
