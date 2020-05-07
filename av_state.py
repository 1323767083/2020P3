import numpy as np

def init_gc(lgc):
    global lc
    lc=lgc


class Phase_State_V3:
    P_Num = 2  # total Number of Phases
    P_NB = 0  # Phase no buy
    P_HP = 1  # phase holding period
    P_END =2  # finished

    CuPs_explore_force_actions = [[0],[2]]

    APSDic={
        0:[[P_NB,P_HP]],
        1:[[P_NB,P_NB]],
        2:[[P_HP,P_END]],
        3:[[P_HP, P_HP]]
    }
    def __init__(self,called_by="explore"):
        self.called_by=called_by
        self.CuPs_limits = [lc.specific_param.LNB, lc.LHP]

        assert lc.specific_param.raw_AV_shape==(lc.specific_param.LNB+1+lc.LHP+1,)
    def reset(self):
        self.CuP=self.P_NB
        self.CuPs_idx = [0 for _ in range(self.P_Num)]

    def check_need_force_state(self, action):
        if self.called_by=="explore":
            #if self.CuPs_idx[self.CuP]>=self.CuPs_limits[self.CuP]:
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
                #self.CuPs_idx[self.CuP]+=1
                Done_flag=support_view_dic["last_day_flag"]
        else:
            assert actual_action == 2
            self.CuPs_idx[self.P_HP] = 0  # this is set the holding period to 0 after successful sell
            Done_flag = True
        return Done_flag

    def fabricate_av_and_update_support_view(self, state, support_view_dic):
        assert not lc.flag_multi_buy, "{0} not support multi buy".format(self.__class__.__name__)
        idx_HP=self.CuPs_idx[self.P_HP]
        idx_NB=self.CuPs_idx[self.P_NB] + lc.LHP + 1
        lav=[1 if idx in [idx_HP,idx_NB] else 0 for idx in list(range(lc.specific_param.raw_AV_shape[0]))]
        #state.append(np.array(lav).reshape(1, -1))
        nav=np.array(lav).reshape(1, -1)
        if self.CuP in [self.P_HP,self.P_END]:
            nav[0, lc.LHP + 1:] = 0
        state.append(nav)
        support_view_dic["holding"] = 1 if state[2][0][0] == 0 else 0

        support_view_dic["flag_force_sell"] = True if self.CuP==self.P_END and \
                            self.CuPs_idx[self.P_HP] == self.CuPs_limits[self.P_HP] else False

    def get_OS_av(self,av):
        assert av.shape[1]==lc.specific_param.raw_AV_shape[0]
        return av[:,:lc.LHP + 1]


    def get_OB_av(self,av):
        assert av.shape[1]==lc.specific_param.raw_AV_shape[0]
        return av[:, lc.LHP + 1:]

    def check_holding_item(self,av_item):
        return False if av_item[0] == 1 else True


class Phase_State_V8:
    P_Num = 3  # total Number of Phases
    P_NT = 0  # Phase no trans
    P_NB = 1  # Phase no buy
    P_HP = 2  # phase holding period
    P_END =3  # finished

    CuPs_explore_force_actions = [[1],[0],[2]]

    APSDic={
        0:[[P_NT,P_HP],[P_NB,P_HP]],
        1:[[P_NT,P_NB],[P_NB,P_NB]],
        2:[[P_HP,P_END]],
        3:[[P_HP, P_HP]],
        4:[[P_NT, P_NT]]
    }
    def __init__(self,called_by="explore"):
        self.called_by=called_by
        self.CuPs_limits = [lc.specific_param.LNT, lc.specific_param.LNB, lc.LHP]

        assert lc.specific_param.raw_AV_shape[0]==lc.specific_param.LNT+1+lc.specific_param.LNB+1+lc.LHP+1
    def reset(self):
        self.CuP=self.P_NT
        self.CuPs_idx = [0 for _ in range(self.P_Num)]

    def check_need_force_state(self, action):
        if self.called_by=="explore":
            if self.CuPs_idx[self.CuP]>=self.CuPs_limits[self.CuP]-1:
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
                Done_flag=support_view_dic["last_day_flag"]
        else:
            assert actual_action == 2
            self.CuPs_idx[self.P_HP] = 0  # this is set the holding period to 0 after successful sell
            Done_flag = True
        return Done_flag

    def fabricate_av_and_update_support_view(self, state, support_view_dic):
        assert not lc.flag_multi_buy, "{0} not support multi buy".format(self.__class__.__name__)
        idx_HP=self.CuPs_idx[self.P_HP]
        idx_NT=self.CuPs_idx[self.P_NT] + lc.LHP + 1
        idx_NB=self.CuPs_idx[self.P_NB] + lc.LHP + 1 + lc.specific_param.LNT +1
        lav=[1 if idx in [idx_HP,idx_NT,idx_NB] else 0 for idx in list(range(lc.specific_param.raw_AV_shape[0]))]
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

        support_view_dic["flag_force_sell"] = True if self.CuP==self.P_END and \
                            self.CuPs_idx[self.P_HP] == self.CuPs_limits[self.P_HP] else False

    def get_OS_av(self,av):
        assert av.shape[1]==lc.specific_param.raw_AV_shape[0]
        return av[:,:lc.LHP + 1]


    def get_OB_av(self,av):
        assert av.shape[1]==lc.specific_param.raw_AV_shape[0]
        av_OB=np.concatenate([av[:,:1],av[:,lc.LHP + 1:]], axis=-1)
        return av_OB

    def check_holding_item(self,av_item):
        return False if av_item[0] == 1 else True