import numpy as np

class Phase_Data:
    def __init__(self, lc):
        self.lc=lc
        self.P_Num = 2  # total Number of Phases
        self.P_NB = 0  # Phase no buy
        self.P_HP = 1  # phase holding period
        self.P_END = 2  # finished
        self.P_init = self.P_NB
        self.CuPs_exit_actions = [0, 2]
        self.CuPs_limits = [self.lc.LNB, self.lc.LHP]

class Phase_State(Phase_Data):
    def __init__(self,lc,calledby):
        Phase_Data.__init__(self,lc)
        assert calledby in ["Explore", "Eval"]
        self.Flag_Calledby_Explore = True if calledby =="Explore" else False
        self.i_av=globals()[lc.CLN_AV](lc)
        for tag in ["V2","V3"]:
            if tag in self.lc.system_type :
                getattr(self,"Init_Phase_State_{0}_OS_{1}".format("V2",calledby))()
                break
        else:
            assert False, "{0} only support V2 V3".format(self.__class__.__name__)

    def Init_Phase_State_V2_OS_Explore(self):
        self.CuPs_force_flag = [True, True]
        assert self.lc.LNB==0

    def Init_Phase_State_V2_OS_Eval(self):
        self.CuPs_force_flag = [True, False]
        assert self.lc.LNB==0

    def Init_Phase_State_V3_OB_Explore(self):
        self.CuPs_force_flag = [True, True]
        assert self.lc.LNB != 0
    def Init_Phase_State_V3_OB_Eval(self):
        self.CuPs_force_flag = [False, False]
        assert self.lc.LNB == 0
        # TODO this is avoid the buy action always on the same day, especially for DBTP_DayByDay_reader

    def reset(self):
        self.CuP=self.P_init
        self.CuPs_idx = [0 for _ in range(self.P_Num)]
        self.raw_av = [0 for _ in range(self.lc.raw_AV_shape[0])]

    def old_check_need_force_state(self, action):
        if self.CuPs_idx[self.CuP] == self.CuPs_limits[self.CuP]-1:
            if self.CuPs_force_flag[self.CuP]:
                adj_action=self.CuPs_exit_actions[self.CuP]
                return adj_action
        return action

    def check_need_force_state(self, action):
        if self.Is_Phase_Last_Step() and self.CuPs_force_flag[self.CuP]:
            adj_action=self.CuPs_exit_actions[self.CuP]
            return adj_action
        else:
            return action

    def update_phase_state(self, action, return_message):
        if self.Is_Phase_Last_Step():
            return self.Phase_Last_Step(action, return_message)
        else:
            return self.Phase_Normal_Step(action, return_message)

    def Is_Phase_Last_Step(self):
        return self.CuPs_idx[self.CuP] == self.CuPs_limits[self.CuP] - 1

    def Phase_Last_Step(self,action, return_message):
        if action == self.CuPs_exit_actions[self.CuP] and return_message == "Success":
            self.CuP+=1
            self.CuPs_idx[self.CuP] += 1
            raw_av = self.i_av.fabricate_av(self.CuPs_idx, self.CuP, flag_CuP_finished=True, flag_CuP_Successed=True)
            Flag_Done= True if self.CuP==self.P_END else False  #next phase is end True or continue in next phase False
        else:
            self.CuPs_idx[self.CuP] += 1
            raw_av = self.i_av.fabricate_av(self.CuPs_idx, self.CuP, flag_CuP_finished=True, flag_CuP_Successed=False)
            Flag_Done = True   #exceed limitation  Error
        return Flag_Done, raw_av

    def Phase_Normal_Step(self, action, return_message):
        if action == self.CuPs_exit_actions[self.CuP] and return_message == "Success":
            self.CuP+=1
            self.CuPs_idx[self.CuP] += 1
            raw_av = self.i_av.fabricate_av(self.CuPs_idx, self.CuP, flag_CuP_finished=True, flag_CuP_Successed=True)
            Flag_Done= True if self.CuP==self.P_END else False  #next phase is end True or continue in next phase False
        else:
            self.CuPs_idx[self.CuP] += 1
            raw_av = self.i_av.fabricate_av(self.CuPs_idx, self.CuP, flag_CuP_finished=False, flag_CuP_Successed=False)
            Flag_Done = False   #continue in current phase
        return Flag_Done, raw_av

class AV_Handler(Phase_Data):
    # LNB  number of not buy days
    # +1   normal buy/ force_buy on last day (need also show status in raw av)
    # +2   code  0,0 means this phase not start  1,0 means this phase Success finished 0,1 this phase finish with error
    #     ( last day buy/force buy not sucess due to tinpai)
    # LHP  number of holding days
    # +1   normal sell/ force_buy on last day (need also show status in raw av)
    # +2   code  0,0 means this phase not start  1,0 means this phase Success finished 0,1 this phase finish with error
    #     ( last day sell/force sell not success due to tinpai)
    def __init__(self,lc):
        Phase_Data.__init__(self,lc)
        assert self.lc.raw_AV_shape == (self.lc.LNB + 1 +2+ self.lc.LHP + 1+2,)
        self.PAVStart_idx=[0,self.lc.LNB + 1 +2]
        self.PResStart_idx = [self.lc.LNB + 1, self.lc.LNB + 1 +2+ self.lc.LHP + 1]

    def Fresh_Raw_AV(self):
        return [0 for _ in range(self.lc.raw_AV_shape[0])]

    def fabricate_av(self, CuPs_idx,CuP,flag_CuP_finished, flag_CuP_Successed):
        Raw_AV=self.Fresh_Raw_AV()
        for Pid in list(range(len(CuPs_idx))):
            if Pid<CuP:
                Raw_AV[self.PAVStart_idx[Pid]+CuPs_idx[Pid]]=1
            elif Pid==CuP:
                Raw_AV[self.PAVStart_idx[Pid] + CuPs_idx[Pid]] = 1
                if flag_CuP_finished:
                    Raw_AV[self.PResStart_idx[Pid]+0 if flag_CuP_Successed else 1]=1
                else:
                    pass #phase result keep 0,0 as phase is unfinished
            else:
                pass # phase av keep init status
        return Raw_AV

    def Is_Phase_Error_Finished(self, Pid, raw_av):
        assert Pid <self.P_END
        return True if raw_av[self.PResStart_idx[Pid]]==0 and raw_av[self.PResStart_idx[Pid]+1]==1 else False

    def Is_Phase_Success_finished(self, Pid, raw_av):
        assert Pid < self.P_END
        return True if raw_av[self.PResStart_idx[Pid]]==1 and raw_av[self.PResStart_idx[Pid]+1]==0 else False

    def get_OS_av(self,raw_av):
        return raw_av[self.PAVStart_idx[self.P_HP]:self.PAVStart_idx[self.P_HP]+self.lc.OS_AV_shape[0]]

    def get_OB_av(self,raw_av):
        return raw_av[self.PAVStart_idx[self.P_NB]:self.PAVStart_idx[self.P_NB] + self.lc.OB_AV_shape[0]]

    def check_holding_item(self,raw_av):
        if any(raw_av[self.PResStart_idx[self.P_HP]:self.PResStart_idx[self.P_HP]+2]): #HP phase finished
            return False
        else:
            return any(raw_av[self.PAVStart_idx[self.P_HP]:self.PResStart_idx[self.P_HP]])
        
