import numpy as np
import sys
import random
from action_comm import actionOBOS
#from env_ar import env_reward_basic#,env_reward_10,env_reward_100,env_reward_1000

def init_gc(gc):
    global Cmemory,lgc
    lgc = gc
    Cmemory=globals()[lgc.CLN_TDmemory]

# brain train buffer/ worker eval loss buffer
class brain_buffer:
    def __init__(self, Buffer_nb_Features):  #Buffer_nb_Features =gc.Buffer_nb_Features
        self.train_queue = [[] for _ in range(Buffer_nb_Features + 2 * 2)]  # each state have 3 substate so add 2
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


class brain_buffer_reuse:
    def __init__(self, Buffer_nb_Features):  #Buffer_nb_Features =gc.Buffer_nb_Features
        self.tq_numb_col        =   Buffer_nb_Features + 2 * 2+1
        self.tq_count_idx       =   self.tq_numb_col-1
        self.tq_out_numb_col    =   self.tq_numb_col-1
        self.reuse_times        =   lgc.brain_buffer_reuse_times
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


    def reset(self):
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


class TD_memory_LHPP2V2_old:
    def __init__(self, Brain_gamma, Brain_FDn, output_buffer):
        assert Brain_FDn==1, "This TD_memory_LHPP2V2_old only can support td=1, otherwise all state base on the final state, " \
                             "which only while Td=1 is ok"
        self.memory = []  # used for n_step return
        self.FDn = Brain_FDn
        self.gamma = Brain_gamma
        self.output_buffer = output_buffer
        self.i_actionOBOS=actionOBOS(lgc.train_action_type)

    def get_sample(self, memory, n):      #use infrom to transfer instant r
        s, a, _, _, done, support_view_dic = memory[0]
        _, _, _, s_, _, _support_view_dic = memory[n - 1]
        return s, a, None, s_, done, support_view_dic,_support_view_dic

    def add_to_train_buffer_to_server(self, si, aa04, r, si_, done, support_view_dic):

        self.memory.append([si, aa04, r, si_, done, support_view_dic])
        if done:
            Ls, La, _, _, Ldone, Lsupport_view_dic = self.memory[-1]
            if Lsupport_view_dic["action_return_message"] == "Success" and Lsupport_view_dic[
                "action_taken"] == "Sell" and Lsupport_view_dic[
                "holding"] == 0:
                pass
            else:
                del self.memory[:]
                return

            success_buy_idx, AccR=self.get_after_buy_accumulate_R_buy()
            # need to delete all the buy and buy no action record as the V2 trainer has action only have 2 dimention now
            del self.memory[: success_buy_idx+1]  # # delete to first after buy need to delete
            while len(self.memory) > 0:
                n = len(self.memory)
                SdisS_=self.FDn if n >= self.FDn else n
                s, aa04, _, s_, train_done, support_view_dic, _support_view_dic = self.get_sample(self.memory, SdisS_)
                adj_r = self.get_accumulate_R(SdisS_)
                np_adj_r = np.array([[adj_r]])
                adjust_a = self.i_actionOBOS.I_TD_buffer(aa04)
                #adjust_a = aa04[:, 2:]
                adjust_s = [s[0], s[1], s[2][:, :lgc.specific_param.OS_AV_shape[0]]]
                adjust_s_ = [s_[0], s_[1], s_[2][:, :lgc.specific_param.OS_AV_shape[0]]]
                support_view_dic["_support_view_dic"] = dict(_support_view_dic)
                support_view_dic["SdisS_"]=SdisS_
                np_support_view = np.array([[support_view_dic]])
                self.output_buffer.append([adjust_s, adjust_a, np_adj_r, adjust_s_, train_done, np_support_view])
                self.memory.pop(0)

    def get_after_buy_accumulate_R_buy(self):
        AccR=0
        success_buy_idx=0
        flag_got_buy=False
        for idx in list(reversed(list(range(len(self.memory))))):
            on_sv_dic = self.memory[idx][5]
            AccR = self.memory[idx][2] + AccR * self.gamma
            if on_sv_dic["action_taken"] == "Buy" and on_sv_dic["action_return_message"] == "Success":
                success_buy_idx=idx
                break
        return success_buy_idx, AccR

    def get_accumulate_R(self, Num_record):
        AccR=0
        for idx in list(reversed(list(range(Num_record)))):
            AccR=self.memory[idx][2]+AccR*self.gamma
        return AccR

class TD_memory_LHPP2V2_new:
    def __init__(self, Brain_gamma, Brain_FDn, output_buffer):
        assert Brain_FDn!=1, "This TD_memory_LHPP2V2_new can support td!=1, try to solve the limitation of the TD_memory_LHPP2V2_old"
        self.memory = []  # used for n_step return
        self.FDn = Brain_FDn
        self.gamma = Brain_gamma
        self.output_buffer = output_buffer
        self.i_actionOBOS=actionOBOS(lgc.train_action_type)

    def get_sample(self, memory, n):      #use infrom to transfer instant r
        s, a, _, _, done, support_view_dic = memory[0]
        _, _, _, s_, _, _support_view_dic = memory[n - 1]
        return s, a, None, s_, done, support_view_dic,_support_view_dic

    def add_to_train_buffer_to_server(self, si, aa04, r, si_, done, support_view_dic):

        self.memory.append([si, aa04, r, si_, done, support_view_dic])
        if done:
            Ls, La, _, _, Ldone, Lsupport_view_dic = self.memory[-1]
            if Lsupport_view_dic["action_return_message"] == "Success" and Lsupport_view_dic[
                "action_taken"] == "Sell" and Lsupport_view_dic[
                "holding"] == 0:
                pass
            else:
                del self.memory[:]
                return

            success_buy_idx, AccR=self.get_after_buy_accumulate_R_buy()
            # need to delete all the buy and buy no action record as the V2 trainer has action only have 2 dimention now
            del self.memory[: success_buy_idx+1]  # # delete to first after buy need to delete
            while len(self.memory) > 0:
                n = len(self.memory)

                if n >= self.FDn:
                    SdisS_ = self.FDn
                elif n==1:
                    SdisS_ = 1
                else:
                    SdisS_ = n-1

                s, aa04, _, s_, train_done, support_view_dic, _support_view_dic = self.get_sample(self.memory, SdisS_)
                adj_r = self.get_accumulate_R(SdisS_)
                np_adj_r = np.array([[adj_r]])
                adjust_a = self.i_actionOBOS.I_TD_buffer(aa04)
                #adjust_a = aa04[:, 2:]
                adjust_s = [s[0], s[1], s[2][:, :lgc.specific_param.OS_AV_shape[0]]]
                adjust_s_ = [s_[0], s_[1], s_[2][:, :lgc.specific_param.OS_AV_shape[0]]]
                support_view_dic["_support_view_dic"] = dict(_support_view_dic)
                support_view_dic["SdisS_"]=SdisS_
                np_support_view = np.array([[support_view_dic]])
                self.output_buffer.append([adjust_s, adjust_a, np_adj_r, adjust_s_, train_done, np_support_view])
                self.memory.pop(0)

    def get_after_buy_accumulate_R_buy(self):
        AccR=0
        success_buy_idx=0
        flag_got_buy=False
        for idx in list(reversed(list(range(len(self.memory))))):
            on_sv_dic = self.memory[idx][5]
            AccR = self.memory[idx][2] + AccR * self.gamma
            if on_sv_dic["action_taken"] == "Buy" and on_sv_dic["action_return_message"] == "Success":
                success_buy_idx=idx
                break
        return success_buy_idx, AccR

    def get_accumulate_R(self, Num_record):
        AccR=0
        for idx in list(reversed(list(range(Num_record)))):
            AccR=self.memory[idx][2]+AccR*self.gamma
        return AccR


#TD_memory_LHPP2V2=TD_memory_LHPP2V2_new  #-3
TD_memory_LHPP2V2=TD_memory_LHPP2V2_old   #-4

class TD_memory_LHPP2V3:
    def __init__(self, Brain_gamma, Brain_FDn, output_buffer):
        assert Brain_FDn == 1, "{0} only support FDn ==1".format(self.__class__.__name__)
        self.memory = []  # used for n_step return
        self.FDn = Brain_FDn
        self.gamma = Brain_gamma
        self.output_buffer = output_buffer
        self.punish_r = -1.0/(lgc.Brain_gamma ** (lgc.LHP-1))
        self.i_actionOBOS = actionOBOS(lgc.train_action_type)
    def get_sample(self, memory, n):
        s, a, _, _, done, support_view_dic = memory[0]
        _, _, _, s_, _, _support_view_dic = memory[n - 1]
        return s, a, None, s_, done, support_view_dic,_support_view_dic

    def add_to_train_buffer_to_server(self, si, aa04, r, si_, done, support_view_dic):

        self.memory.append([si, aa04, r, si_, done, support_view_dic])
        if done:
            _, _, _, _, _, Lsupport_view_dic = self.memory[-1]
            if Lsupport_view_dic["flag_force_sell"]:
                assert Lsupport_view_dic["action_taken"] == "Sell"
                #assert aa04[0, 2] ==1, "{0} {1}".format(Lsupport_view_dic,aa04)

                self.memory[-1][2] = self.punish_r
            elif Lsupport_view_dic["action_return_message"] == "Success" and Lsupport_view_dic[
                "action_taken"] == "Sell" and Lsupport_view_dic["holding"] == 0:
                pass
            else:
                del self.memory[:]
                return

            success_buy_idx, AccR=self.get_after_buy_accumulate_R_buy()

            self.memory[success_buy_idx][2] =  1 if AccR >=1 else -1 if AccR <=-1 else AccR
            #assert False, "here have temporaty implementation on delete every thing except the buy record"
            #del self.memory[:success_buy_idx -1]  # delete before buy need to delete
            del self.memory[success_buy_idx+1:]  # delete from first after buy need to delete
            if len(self.memory)>lgc.specific_param.max_record_taken and lgc.specific_param.max_record_taken!=-1:
                del self.memory[:-lgc.specific_param.max_record_taken]
            while len(self.memory) > 0:
                n = len(self.memory)
                SdisS_=self.FDn if n >= self.FDn else n
                s, aa04, _, s_, train_done, support_view_dic, _support_view_dic = self.get_sample(self.memory, SdisS_)
                adj_r = self.get_accumulate_R(SdisS_)
                np_adj_r = np.array([[adj_r]])
                adjust_a=self.i_actionOBOS.I_TD_buffer(aa04)
                #adjust_a = aa04[:, :2]
                adjust_s = [s[0], s[1], s[2][:, :lgc.specific_param.OS_AV_shape[0]]]
                adjust_s_ = [s_[0], s_[1], s_[2][:, :lgc.specific_param.OS_AV_shape[0]]]
                support_view_dic["_support_view_dic"] = dict(_support_view_dic)
                support_view_dic["SdisS_"]=SdisS_
                np_support_view = np.array([[support_view_dic]])
                self.output_buffer.append([adjust_s, adjust_a, np_adj_r, adjust_s_, train_done, np_support_view])
                self.memory.pop(0)

    def get_after_buy_accumulate_R_buy(self):
        AccR=0
        success_buy_idx=0
        flag_got_buy=False
        for idx in list(reversed(list(range(len(self.memory))))):
            on_sv_dic = self.memory[idx][5]
            AccR = self.memory[idx][2] + AccR * self.gamma
            if on_sv_dic["action_taken"] == "Buy" and on_sv_dic["action_return_message"] == "Success":
                success_buy_idx=idx
                assert self.memory[idx][1][0, 0]==1, "{0} {1}".format(on_sv_dic, self.memory[idx][1])
                break
        return success_buy_idx, AccR


    def get_accumulate_R(self, Num_record):
        AccR=0
        for idx in list(reversed(list(range(Num_record)))):
            AccR=self.memory[idx][2]+AccR*self.gamma
        return AccR

class TD_memory_LHPP2V5(TD_memory_LHPP2V3):
    def __init__(self, Brain_gamma, Brain_FDn, output_buffer):
        TD_memory_LHPP2V3.__init__(self, Brain_gamma, Brain_FDn, output_buffer)
        self.punish_r = -100.0 / (lgc.Brain_gamma ** (lgc.LHP - 1))

class TD_memory_LHPP2V6(TD_memory_LHPP2V5):
    def __init__(self, Brain_gamma, Brain_FDn, output_buffer):
        TD_memory_LHPP2V5.__init__(self, Brain_gamma, Brain_FDn, output_buffer)


class TD_memory_LHPP2V7:
    def __init__(self, Brain_gamma, Brain_FDn, output_buffer):
        assert Brain_FDn == 1, "{0} only support FDn ==1".format(self.__class__.__name__)
        self.memory = []  # used for n_step return
        self.FDn = Brain_FDn
        self.gamma = Brain_gamma
        self.output_buffer = output_buffer
        self.punish_r = -1.0/(lgc.Brain_gamma ** (lgc.LHP-1))
        self.i_actionOBOS = actionOBOS(lgc.train_action_type)
    def get_sample(self, memory, n):
        s, a, _, _, done, support_view_dic = memory[0]
        _, _, _, s_, _, _support_view_dic = memory[n - 1]
        return s, a, None, s_, done, support_view_dic,_support_view_dic

    def add_to_train_buffer_to_server(self, si, aa04, r, si_, done, support_view_dic):

        self.memory.append([si, aa04, r, si_, done, support_view_dic])
        if done:
            _, _, _, _, _, Lsupport_view_dic = self.memory[-1]
            if Lsupport_view_dic["flag_force_sell"]:
                assert Lsupport_view_dic["action_taken"] == "Sell"
                #assert aa04[0, 2] ==1, "{0} {1}".format(Lsupport_view_dic,aa04)

                self.memory[-1][2] = self.punish_r
            elif Lsupport_view_dic["action_return_message"] == "Success" and Lsupport_view_dic[
                "action_taken"] == "Sell" and Lsupport_view_dic["holding"] == 0:
                pass
            else:
                del self.memory[:]
                return

            success_buy_idx, AccR=self.get_after_buy_accumulate_R_buy()

            #self.memory[success_buy_idx][2] =  1 if AccR >=1 else -1 if AccR <=-1 else AccR
            #del self.memory[success_buy_idx+1:]  # delete from first after buy need to delete
            assert self.memory[success_buy_idx+1][0][2][0][1]==1, "holding status second element of av should be 1 {0}".format(self.memory[success_buy_idx+1][0][2])
            self.memory[success_buy_idx+1][2] = 1 if AccR >= 1 else -1 if AccR <= -1 else AccR
            del self.memory[success_buy_idx + 2:]  # delete from second after buy need to delete
            if len(self.memory)>lgc.specific_param.max_record_taken and lgc.specific_param.max_record_taken!=-1:
                del self.memory[:-lgc.specific_param.max_record_taken]
            while len(self.memory) > 0:
                n = len(self.memory)
                SdisS_=self.FDn if n >= self.FDn else n
                s, aa04, _, s_, train_done, support_view_dic, _support_view_dic = self.get_sample(self.memory, SdisS_)
                adj_r = self.get_accumulate_R(SdisS_)
                np_adj_r = np.array([[adj_r]])
                adjust_a=self.i_actionOBOS.I_TD_buffer(aa04)
                #adjust_a = aa04[:, :2]
                #adjust_s = [s[0], s[1], s[2][:, :lgc.specific_param.OS_AV_shape[0]]]
                #adjust_s_ = [s_[0], s_[1], s_[2][:, :lgc.specific_param.OS_AV_shape[0]]]
                adjust_s = s
                adjust_s_ = s_
                support_view_dic["_support_view_dic"] = dict(_support_view_dic)
                support_view_dic["SdisS_"]=SdisS_
                np_support_view = np.array([[support_view_dic]])
                self.output_buffer.append([adjust_s, adjust_a, np_adj_r, adjust_s_, train_done, np_support_view])
                self.memory.pop(0)

    def get_after_buy_accumulate_R_buy_old(self):
        AccR=0
        success_buy_idx=0
        flag_got_buy=False
        for idx in list(reversed(list(range(len(self.memory))))):
            on_sv_dic = self.memory[idx][5]
            # make the sell reward all on buy in OB train
            AccR = self.memory[idx][2] + AccR * self.gamma
            if on_sv_dic["action_taken"] == "Buy" and on_sv_dic["action_return_message"] == "Success":
                success_buy_idx=idx
                assert self.memory[idx][1][0, 0]==1, "{0} {1}".format(on_sv_dic, self.memory[idx][1])
                break
        return success_buy_idx, AccR

    def get_after_buy_accumulate_R_buy_new(self):
        AccR=0
        success_buy_idx=0
        flag_got_buy=False
        flag_last=True
        for idx in list(reversed(list(range(len(self.memory))))):
            on_sv_dic = self.memory[idx][5]

            if flag_last:
                AccR = self.memory[idx][2]  # sell result
                flag_last=False
            else:
                assert self.memory[idx][2]==0 # all are no action
            if on_sv_dic["action_taken"] == "Buy" and on_sv_dic["action_return_message"] == "Success":
                success_buy_idx=idx
                assert self.memory[idx][1][0, 0]==1, "{0} {1}".format(on_sv_dic, self.memory[idx][1])
                break
        return success_buy_idx, AccR

    get_after_buy_accumulate_R_buy=get_after_buy_accumulate_R_buy_new

    def get_accumulate_R(self, Num_record):
        AccR=0
        for idx in list(reversed(list(range(Num_record)))):
            AccR=self.memory[idx][2]+AccR*self.gamma
        return AccR

'''
#TD_memory_LHPP2V5 is same as TD_memory_LHPP2V3
class TD_memory_LHPP2V5:
    def __init__(self, Brain_gamma, Brain_FDn, output_buffer):
        assert Brain_FDn == 1, "{0} only support FDn ==1".format(self.__class__.__name__)
        self.memory = []  # used for n_step return
        self.FDn = Brain_FDn
        self.gamma = Brain_gamma
        self.output_buffer = output_buffer
        self.punish_r = -100.0/(lgc.Brain_gamma ** (lgc.LHP-1))
        #self.punish_r = -1
        #self.punish_r = lgc.specific_param.punish_r_base * lgc.train_reward_scaler_factor / (lgc.Brain_gamma ** (lgc.LHP-1))
        #self.punish_r = lgc.specific_param.punish_r_base * lgc.train_reward_scaler_factor/(lgc.Brain_gamma**lgc.LHP)
        #self.punish_r = lgc.specific_param.punish_r_base * lgc.train_reward_scaler_factor
        self.i_actionOBOS = actionOBOS(lgc.train_action_type)
    def get_sample(self, memory, n):
        s, a, _, _, done, support_view_dic = memory[0]
        _, _, _, s_, _, _support_view_dic = memory[n - 1]
        return s, a, None, s_, done, support_view_dic,_support_view_dic

    def add_to_train_buffer_to_server(self, si, aa05, r, si_, done, support_view_dic):

        self.memory.append([si, aa05, r, si_, done, support_view_dic])
        if done:
            _, _, _, _, _, Lsupport_view_dic = self.memory[-1]
            if Lsupport_view_dic["flag_force_sell"]:
                assert Lsupport_view_dic["action_taken"] == "Sell"
                #assert aa04[0, 2] ==1, "{0} {1}".format(Lsupport_view_dic,aa04)

                self.memory[-1][2] = self.punish_r
                pass
            elif Lsupport_view_dic["action_return_message"] == "Success" and Lsupport_view_dic[
                "action_taken"] == "Sell" and Lsupport_view_dic["holding"] == 0:
                pass
            else:
                del self.memory[:]
                return

            success_buy_idx, AccR=self.get_after_buy_accumulate_R_buy()

            self.memory[success_buy_idx][2] =  1 if AccR >=1 else -1 if AccR <=-1 else AccR
            #assert False, "here have temporaty implementation on delete every thing except the buy record"
            #del self.memory[:success_buy_idx -1]  # delete before buy need to delete
            del self.memory[success_buy_idx+1:]  # delete from first after buy need to delete
            if len(self.memory)>lgc.specific_param.max_record_taken and lgc.specific_param.max_record_taken!=-1:
                del self.memory[:-lgc.specific_param.max_record_taken]
            while len(self.memory) > 0:
                n = len(self.memory)
                SdisS_=self.FDn if n >= self.FDn else n
                s, aa05, _, s_, train_done, support_view_dic, _support_view_dic = self.get_sample(self.memory, SdisS_)
                adj_r = self.get_accumulate_R(SdisS_)
                np_adj_r = np.array([[adj_r]])
                adjust_a=self.i_actionOBOS.I_TD_buffer(aa05)
                #adjust_a = aa04[:, :2]
                adjust_s = [s[0], s[1], s[2][:, :lgc.specific_param.OS_AV_shape[0]]]
                adjust_s_ = [s_[0], s_[1], s_[2][:, :lgc.specific_param.OS_AV_shape[0]]]
                support_view_dic["_support_view_dic"] = dict(_support_view_dic)
                support_view_dic["SdisS_"]=SdisS_
                np_support_view = np.array([[support_view_dic]])
                self.output_buffer.append([adjust_s, adjust_a, np_adj_r, adjust_s_, train_done, np_support_view])
                self.memory.pop(0)

    def get_after_buy_accumulate_R_buy(self):
        AccR=0
        success_buy_idx=0
        flag_got_buy=False
        for idx in list(reversed(list(range(len(self.memory))))):
            on_sv_dic = self.memory[idx][5]
            AccR = self.memory[idx][2] + AccR * self.gamma
            if on_sv_dic["action_taken"] == "Buy" and on_sv_dic["action_return_message"] == "Success":
                success_buy_idx=idx
                assert self.memory[idx][1][0, 0]==1, "{0} {1}".format(on_sv_dic, self.memory[idx][1])
                break
        return success_buy_idx, AccR


    def get_accumulate_R(self, Num_record):
        AccR=0
        for idx in list(reversed(list(range(Num_record)))):
            AccR=self.memory[idx][2]+AccR*self.gamma
        return AccR

class TD_memory_LHPP2V6:
    def __init__(self, Brain_gamma, Brain_FDn, output_buffer):
        assert Brain_FDn == 1, "{0} only support FDn ==1".format(self.__class__.__name__)
        self.memory = []  # used for n_step return
        self.FDn = Brain_FDn
        self.gamma = Brain_gamma
        self.output_buffer = output_buffer
        self.punish_r = -100.0 / (lgc.Brain_gamma ** (lgc.LHP - 1))
        # self.punish_r = -1
        # self.punish_r = lgc.specific_param.punish_r_base * lgc.train_reward_scaler_factor / (lgc.Brain_gamma ** (lgc.LHP-1))
        # self.punish_r = lgc.specific_param.punish_r_base * lgc.train_reward_scaler_factor/(lgc.Brain_gamma**lgc.LHP)
        # self.punish_r = lgc.specific_param.punish_r_base * lgc.train_reward_scaler_factor
        self.i_actionOBOS = actionOBOS(lgc.train_action_type)

    def get_sample(self, memory, n):
        s, a, _, _, done, support_view_dic = memory[0]
        _, _, _, s_, _, _support_view_dic = memory[n - 1]
        return s, a, None, s_, done, support_view_dic, _support_view_dic

    def add_to_train_buffer_to_server(self, si, aa05, r, si_, done, support_view_dic):

        self.memory.append([si, aa05, r, si_, done, support_view_dic])
        if done:
            _, _, _, _, _, Lsupport_view_dic = self.memory[-1]
            if Lsupport_view_dic["flag_force_sell"]:
                assert Lsupport_view_dic["action_taken"] == "Sell"
                # assert aa04[0, 2] ==1, "{0} {1}".format(Lsupport_view_dic,aa04)

                self.memory[-1][2] = self.punish_r
                pass
            elif Lsupport_view_dic["action_return_message"] == "Success" and Lsupport_view_dic[
                "action_taken"] == "Sell" and Lsupport_view_dic["holding"] == 0:
                pass
            else:
                del self.memory[:]
                return

            success_buy_idx, AccR = self.get_after_buy_accumulate_R_buy()

            self.memory[success_buy_idx][2] = 1 if AccR >= 1 else -1 if AccR <= -1 else AccR
            # assert False, "here have temporaty implementation on delete every thing except the buy record"
            # del self.memory[:success_buy_idx -1]  # delete before buy need to delete
            del self.memory[success_buy_idx + 1:]  # delete from first after buy need to delete
            if len(self.memory) > lgc.specific_param.max_record_taken and lgc.specific_param.max_record_taken != -1:
                del self.memory[:-lgc.specific_param.max_record_taken]
            while len(self.memory) > 0:
                n = len(self.memory)
                SdisS_ = self.FDn if n >= self.FDn else n
                s, aa05, _, s_, train_done, support_view_dic, _support_view_dic = self.get_sample(self.memory,
                                                                                                  SdisS_)
                adj_r = self.get_accumulate_R(SdisS_)
                np_adj_r = np.array([[adj_r]])
                adjust_a = self.i_actionOBOS.I_TD_buffer(aa05)
                # adjust_a = aa04[:, :2]
                adjust_s = [s[0], s[1], s[2][:, :lgc.specific_param.OS_AV_shape[0]]]
                adjust_s_ = [s_[0], s_[1], s_[2][:, :lgc.specific_param.OS_AV_shape[0]]]
                support_view_dic["_support_view_dic"] = dict(_support_view_dic)
                support_view_dic["SdisS_"] = SdisS_
                np_support_view = np.array([[support_view_dic]])
                self.output_buffer.append([adjust_s, adjust_a, np_adj_r, adjust_s_, train_done, np_support_view])
                self.memory.pop(0)

    def get_after_buy_accumulate_R_buy(self):
        AccR = 0
        success_buy_idx = 0
        flag_got_buy = False
        for idx in list(reversed(list(range(len(self.memory))))):
            on_sv_dic = self.memory[idx][5]
            AccR = self.memory[idx][2] + AccR * self.gamma
            if on_sv_dic["action_taken"] == "Buy" and on_sv_dic["action_return_message"] == "Success":
                success_buy_idx = idx
                assert self.memory[idx][1][0, 0] == 1, "{0} {1}".format(on_sv_dic, self.memory[idx][1])
                break
        return success_buy_idx, AccR

    def get_accumulate_R(self, Num_record):
        AccR = 0
        for idx in list(reversed(list(range(Num_record)))):
            AccR = self.memory[idx][2] + AccR * self.gamma
        return AccR
'''

#worker with eval / worker with eval prepare data send to server
class buffer_to_train:
    #def __init__(self, num_stocks_in_one_group, FDn=1):
    #def __init__(self, num_stocks_in_one_group, Brain_gamma, Brain_FDn, memory_buffer_class_name):
    def __init__(self, num_stocks_in_one_group):
        self.num_stocks_in_one_group = num_stocks_in_one_group
        self.Brain_FDn = lgc.TDn
        self.Brain_gamma = lgc.Brain_gamma
        self.train_buffer_to_server = []
        self.fd_buffer = []
        for _ in range(num_stocks_in_one_group):
            self.fd_buffer.append(Cmemory(self.Brain_gamma, self.Brain_FDn,self.train_buffer_to_server))

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


