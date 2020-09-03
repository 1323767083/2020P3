import numpy as np

def init_gc(lgc):
    global lc
    lc=lgc


class Phase_State_template:
    def __init__(self):
        self.init_data()
        self.sanity_check_av_shape()

    def init_data(self):
        self.P_Num = 2  # total Number of Phases
        self.P_NB = 0  # Phase no buy
        self.P_HP = 1
        self.P_END = 2  # finished

        self.P_init=self.P_NB
        self.CuPs_explore_force_actions = [[0], [2]]
        self.APSDic = {
        }
        self.CuPs_limits = []
        assert lc.raw_AV_shape==(1,)

    def reset(self):
        self.CuP=self.P_init
        self.CuPs_idx = [0 for _ in range(self.P_Num)]

    def check_need_force_state(self, action):
        if self.CuPs_idx[self.CuP] >= self.CuPs_limits[self.CuP]-1:
            adj_action=self.CuPs_explore_force_actions[self.CuP][0]
            return adj_action
        return action


    def update_phase_state(self, support_view_dic, action, return_message):
        if action in [0,2] and return_message !="Success":
            actual_action=1 if action ==0 else 3
        else:
            actual_action=action
        lPS2PS=self.APSDic[actual_action]
        l_next_PS=[PS2PS[1] for PS2PS in lPS2PS if self.CuP==PS2PS[0]]
        assert len(l_next_PS)==1
        self.CuP=l_next_PS[0]
        if self.CuP!=self.P_END:
            self.CuPs_idx[self.CuP] += 1
            if self.CuPs_idx[self.CuP] >= self.CuPs_limits[self.CuP]:
                Done_flag = True
            else:
                #Done_flag=support_view_dic["last_day_flag"]
                Done_flag = support_view_dic["Flag_LastDay"]
        else:
            assert actual_action == 2
            self.CuPs_idx[self.P_HP] = 0  # this is set the holding period to 0 after successful sell
            Done_flag = True
        return Done_flag

    def sanity_check_av_shape(self):
        return

class Phase_State_V2(Phase_State_template):
    def __init__(self):
        Phase_State_template.__init__(self)

    def init_data(self):
        self.P_Num = 1  # total Number of Phases
        self.P_HP = 0  # phase holding period
        self.P_END = 1  # finished

        self.P_init = self.P_HP
        self.CuPs_explore_force_actions = [[2]]
        self.APSDic = {
            0: [[self.P_HP, self.P_HP]],
            1: [[self.P_HP, self.P_HP]],
            2: [[self.P_HP, self.P_END]],
            3: [[self.P_HP, self.P_HP]]
        }
        self.CuPs_limits = [lc.LHP]
        assert lc.raw_AV_shape == (lc.LHP + 1,)

    def fabricate_av_and_update_support_view(self, state, support_view_dic, flag_force_sell):
        assert not lc.flag_multi_buy, "{0} not support multi buy".format(self.__class__.__name__)
        idx_HP=self.CuPs_idx[self.P_HP]
        lav=[1 if idx ==idx_HP else 0 for idx in list(range(lc.raw_AV_shape[0]))]
        state.append(np.array(lav).reshape(1, -1))
        support_view_dic["holding"] = 1 if state[2][0][0] == 0 else 0
        support_view_dic["flag_force_sell"]=flag_force_sell
        #support_view_dic["flag_force_sell"] = True if self.CuP==self.P_END and \
        #                    self.CuPs_idx[self.P_HP] == self.CuPs_limits[self.P_HP] else False
        #support_view_dic["flag_force_sell"] = True if self.CuP==self.P_END and \
        #                    self.CuPs_idx[self.P_HP] == self.CuPs_limits[self.P_HP]-1 else False



    def get_OS_av(self,av):
        assert av.shape[1]==lc.raw_AV_shape[0], "av.shape={0}. lc.raw_AV_shape= {1}".format(av.shape,lc.raw_AV_shape)
        return av

    def get_OB_av(self,av):
        assert False

    def check_holding_item(self,av_item):
        return False if av_item[0] == 1 else True

    def sanity_check_av_shape(self):
        assert lc.OS_AV_shape== (lc.LHP + 1,)
        assert lc.raw_AV_shape == (lc.LHP + 1,)


class Phase_State_V3__1(Phase_State_template):
    def __init__(self):
        Phase_State_template.__init__(self)

    def init_data(self):
        self.P_Num = 2  # total Number of Phases
        self.P_NB = 0  # Phase no buy
        self.P_HP = 1  # phase holding period
        self.P_END = 2  # finished

        self.P_init = self.P_NB
        self.CuPs_explore_force_actions = [[0], [2]]
        self.APSDic = {
            0: [[self.P_NB, self.P_HP]],
            1: [[self.P_NB, self.P_NB]],
            2: [[self.P_HP, self.P_END]],
            3: [[self.P_HP, self.P_HP]]
        }
        self.CuPs_limits = [lc.specific_param.LNB, lc.LHP]
        assert lc.raw_AV_shape == (lc.specific_param.LNB + 1 + lc.LHP + 1,)

    def fabricate_av_and_update_support_view(self, state, support_view_dic,flag_force_sell):
        assert not lc.flag_multi_buy, "{0} not support multi buy".format(self.__class__.__name__)
        idx_HP=self.CuPs_idx[self.P_HP]
        idx_NB=self.CuPs_idx[self.P_NB] + lc.LHP + 1
        lav=[1 if idx in [idx_HP,idx_NB] else 0 for idx in list(range(lc.raw_AV_shape[0]))]
        #state.append(np.array(lav).reshape(1, -1))
        nav=np.array(lav).reshape(1, -1)
        if self.CuP in [self.P_HP,self.P_END]:
            nav[0, lc.LHP + 1:] = 0
        state.append(nav)
        support_view_dic["holding"] = 1 if state[2][0][0] == 0 else 0
        support_view_dic["flag_force_sell"] = flag_force_sell
        #support_view_dic["flag_force_sell"] = True if self.CuP==self.P_END and \
        #                    self.CuPs_idx[self.P_HP] == self.CuPs_limits[self.P_HP] else False

    def get_OS_av(self,av):
        assert av.shape[1]==lc.raw_AV_shape[0]
        return av[:,:lc.LHP + 1]

    def get_OB_av(self,av):
        assert av.shape[1]==lc.raw_AV_shape[0]
        return av[:, lc.LHP + 1:]

    def check_holding_item(self,av_item):
        return False if av_item[0] == 1 else True

    def sanity_check_av_shape(self):
        assert lc.OS_AV_shape == (lc.LHP + 1,)
        assert lc.OB_AV_shape == (lc.specific_param.LNB + 1,)
        assert lc.raw_AV_shape == (lc.specific_param.LNB + 1 + lc.LHP + 1,)

class Phase_State_V3__2(Phase_State_V3__1):
    def sanity_check_av_shape(self):
        assert lc.OS_AV_shape== (lc.LHP + 1,)
        assert lc.OB_AV_shape == (1,)
        assert lc.raw_AV_shape == (lc.specific_param.LNB + 1 + lc.LHP + 1,)
    def get_OB_av(self,av):
        assert av.shape[1]==lc.raw_AV_shape[0]
        return av[:, :1]



class Phase_State_V8(Phase_State_template):
    def __init__(self):
        Phase_State_template.__init__(self)

    def init_data(self):
        self.P_Num = 3  # total Number of Phases
        self.P_NT = 0  # Phase no trans
        self.P_NB = 1  # Phase no buy
        self.P_HP = 2  # phase holding period
        self.P_END = 3  # finished

        CuPs_explore_force_actions = [[1], [0], [2]]

        APSDic = {
            0: [[self.P_NT, self.P_HP], [self.P_NB, self.P_HP]],
            1: [[self.P_NT, self.P_NB], [self.P_NB, self.P_NB]],
            2: [[self.P_HP, self.P_END]],
            3: [[self.P_HP, self.P_HP]],
            4: [[self.P_NT, self.P_NT]]
        }
        self.CuPs_limits = [lc.specific_param.LNT, lc.specific_param.LNB, lc.LHP]
        assert lc.raw_AV_shape[0]==lc.specific_param.LNT+1+lc.specific_param.LNB+1+lc.LHP+1

    def fabricate_av_and_update_support_view(self, state, support_view_dic,flag_force_sell):
        assert not lc.flag_multi_buy, "{0} not support multi buy".format(self.__class__.__name__)
        idx_HP=self.CuPs_idx[self.P_HP]
        idx_NT=self.CuPs_idx[self.P_NT] + lc.LHP + 1
        idx_NB=self.CuPs_idx[self.P_NB] + lc.LHP + 1 + lc.specific_param.LNT +1
        lav=[1 if idx in [idx_HP,idx_NT,idx_NB] else 0 for idx in list(range(lc.raw_AV_shape[0]))]
        #state.append(np.array(lav).reshape(1, -1))
        nav=np.array(lav).reshape(1, -1)
        if self.CuP==self.P_NB:
            nav[0,lc.LHP + 1:lc.LHP + 1 + lc.specific_param.LNT +1]=0
        elif self.CuP in [self.P_HP,self.P_END]:
            nav[0, lc.LHP + 1:] = 0
        else:
            assert self.CuP==self.P_NT
        state.append(nav)
        support_view_dic["holding"] = 1 if state[2][0][0] == 0 else 0

        #support_view_dic["flag_force_sell"] = True if self.CuP==self.P_END and \
        #                    self.CuPs_idx[self.P_HP] == self.CuPs_limits[self.P_HP] else False
        support_view_dic["flag_force_sell"] =flag_force_sell

    def get_OS_av(self,av):
        assert av.shape[1]==lc.raw_AV_shape[0]
        return av[:,:lc.LHP + 1]


    def get_OB_av(self,av):
        assert av.shape[1]==lc.raw_AV_shape[0]
        av_OB=np.concatenate([av[:,:1],av[:,lc.LHP + 1:]], axis=-1)
        return av_OB

    def check_holding_item(self,av_item):
        return False if av_item[0] == 1 else True

    def sanity_check_av_shape(self):
        assert lc.OS_AV_shape== (lc.LHP + 1,)
        assert lc.OB_AV_shape == (lc.specific_param.LNT + 1 + lc.specific_param.LNB + 1 + 1,)
        assert lc.raw_AV_shape == (lc.specific_param.LNT + 1 + lc.specific_param.LNB + 1 + lc.LHP + 1,)


