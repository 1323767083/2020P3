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
        self.i_av=globals()[lc.CLN_AV_Handler](lc)
        for tag in ["V2","V3"]:
            if tag in self.lc.system_type :
                getattr(self,"Init_Phase_State_{0}_{1}".format(tag,calledby))()
                break
        else:
            assert False, "{0} only support V2 V3".format(self.__class__.__name__)

    def Init_Phase_State_V2_Explore(self):
        self.CuPs_force_flag = [True, True]
        assert self.lc.LNB==1

    def Init_Phase_State_V2_Eval(self):
        self.CuPs_force_flag = [True, False]
        assert self.lc.LNB==1

    def Init_Phase_State_V3_Explore(self):
        self.CuPs_force_flag = [True, True]
        assert self.lc.LNB == 1
    def Init_Phase_State_V3_Eval(self):
        self.CuPs_force_flag = [False, False]
        assert self.lc.LNB == 1
        # TODO this is avoid the buy action always on the same day, especially for DBTP_DayByDay_reader
        #TODO here is set alwas LNB==1 and train buy like a binary classification

    def check_need_force_state(self, action):
        if self.Is_Phase_Last_Step() and self.CuPs_force_flag[self.CuP]:
            adj_action=self.CuPs_exit_actions[self.CuP]
            return adj_action
        else:
            return action

    def reset_phase_state(self):
        self.CuP=self.P_init
        self.CuPs_idx = [0 for _ in range(self.P_Num)]
        raw_av = self.i_av.Fresh_Raw_AV()
        return np.array(raw_av).reshape(1, -1)
    def update_phase_state(self, action, return_message):
        if self.Is_Phase_Last_Step():
            return self.Update_Phase_Last_Step(action, return_message)
        else:
            return self.Update_Phase_Normal_Step(action, return_message)

    def Is_Phase_Last_Step(self):
        return self.CuPs_idx[self.CuP] == self.CuPs_limits[self.CuP] - 1

    def Update_Phase_Last_Step(self,action, return_message):
        if action == self.CuPs_exit_actions[self.CuP] and return_message == "Success":
            raw_av = self.i_av.fabricate_av(self.CuPs_idx, self.CuP, flag_CuP_finished=True, flag_CuP_Successed=True)
            self.CuP += 1  # the fabricate_av should always have current working CuP as input change to next phase use two flag to identify
            Flag_Done= True if self.CuP==self.P_END else False  #next phase is end True or continue in next phase False
        else:
            self.CuPs_idx[self.CuP] += 1
            raw_av = self.i_av.fabricate_av(self.CuPs_idx, self.CuP, flag_CuP_finished=True, flag_CuP_Successed=False)
            Flag_Done = True   #exceed limitation  Error
            action = 3 if self.CuP==self.P_HP else 1
        return Flag_Done, raw_av,action

    def Update_Phase_Normal_Step(self, action, return_message):
        if action == self.CuPs_exit_actions[self.CuP] and return_message == "Success":
            raw_av = self.i_av.fabricate_av(self.CuPs_idx, self.CuP, flag_CuP_finished=True, flag_CuP_Successed=True)
            self.CuP += 1
            Flag_Done= True if self.CuP==self.P_END else False  #next phase is end True or continue in next phase False
        else:
            self.CuPs_idx[self.CuP] += 1
            raw_av = self.i_av.fabricate_av(self.CuPs_idx, self.CuP, flag_CuP_finished=False, flag_CuP_Successed=False)

            Flag_Done = False   #continue in current phase
            action = 3 if self.CuP == self.P_HP else 1
        return Flag_Done, raw_av, action

class AV_Handler(Phase_Data):
    # LNB  number of not buy days
    # +1   normal buy/ force_buy on last day (need also show status in raw av)
    # +2   code  0,0 means this phase not start  1,0 means this phase Success finished 0,1 this phase finish with error
    #     ( last day buy/force buy not sucess due to tinpai)
    # LHP  number of holding days
    # +1   normal sell/ force_buy on last day (need also show status in raw av)
    # +2   code  0,0 means this phase not start  1,0 means this phase Success finished 0,1 this phase finish with error
    #     ( last day sell/force sell not success due to tinpai)
    #+1   Whether this is final record for optimize pupose
    def __init__(self,lc):
        Phase_Data.__init__(self,lc)
        assert self.lc.raw_AV_shape == (self.lc.LNB + 1 +2+ self.lc.LHP + 1+2 +1,)
        self.PAVStart_idx=[0,self.lc.LNB + 1 +2]
        self.PResStart_idx = [self.lc.LNB + 1, self.lc.LNB + 1 +2+ self.lc.LHP + 1]
        self.PFinal_idx= self.lc.LNB + 1 +2+ self.lc.LHP + 1 +2
    #These two function are batch
    def get_OS_AV(self,raw_av):
        #return raw_av[self.PAVStart_idx[self.P_HP]:self.PAVStart_idx[self.P_HP]+self.lc.OS_AV_shape[0]]
        return raw_av[:,self.PAVStart_idx[self.P_HP]:self.PResStart_idx[self.P_HP]]

    def get_OB_AV(self,raw_av):
        return raw_av[:,self.PAVStart_idx[self.P_NB]:self.PResStart_idx[self.P_NB]]

    # Following functions are single
    def Fresh_Raw_AV(self):
        return [0 for _ in range(self.lc.raw_AV_shape[0])]

    def fabricate_av(self,CuPs_idx,CuP,flag_CuP_finished, flag_CuP_Successed):
        Raw_AV = self.Fresh_Raw_AV()
        for Pid in list(range(len(CuPs_idx))):
            if Pid<CuP:
                Raw_AV[self.PAVStart_idx[Pid]+CuPs_idx[Pid]]=1
                Raw_AV[self.PResStart_idx[Pid] + 0 ] = 1  # since Pid < CuP already enter next phase this phase should finished successfully
            elif Pid == CuP:
                Raw_AV[self.PAVStart_idx[Pid] + CuPs_idx[Pid]] = 1
                if flag_CuP_finished:
                    if flag_CuP_Successed:
                        Raw_AV[self.PResStart_idx[Pid] + 0] = 1
                        if Pid+1!=self.P_END:
                            Raw_AV[self.PAVStart_idx[Pid+1] + CuPs_idx[Pid+1]] = 1  #while change phase success should set next phase status in av
                    else:
                        Raw_AV[self.PResStart_idx[Pid] + 1] = 1
        return np.array(Raw_AV).reshape(1, -1)

    def set_final_record_AV(self,raw_av):
        raw_av[self.PFinal_idx]=1
    def check_final_record_AV(self,raw_av):
        return raw_av[self.PFinal_idx]==1

    def Is_Phase_Finished(self,raw_av, phase):
        return True if any(raw_av[self.PResStart_idx[phase]:self.PResStart_idx[phase]+2]) else False

    def Is_Phase_Error_Finished(self, raw_av,Pid ):
        assert Pid <self.P_END
        return True if raw_av[self.PResStart_idx[Pid]]==0 and raw_av[self.PResStart_idx[Pid]+1]==1 else False

    def Is_Phase_Success_finished(self, raw_av,Pid ):
        assert Pid < self.P_END
        return True if raw_av[self.PResStart_idx[Pid]]==1 and raw_av[self.PResStart_idx[Pid]+1]==0 else False

    def Is_Holding_Item(self, raw_av):
        if self.Is_Phase_Success_finished(raw_av, self.P_NB): #P_NB Success finished
            if self.Is_Phase_Finished(raw_av, self.P_HP):  # P_HP phase success or error finished
                return False
            else:
                assert any(raw_av[self.PAVStart_idx[self.P_HP]:self.PResStart_idx[self.P_HP]-1]) , raw_av # not include the last, which is error finished
                return True
        else:
            return False # P_NB error finished or still in P_NB phase

    def get_HP_status_On_S_(self, raw_av):
        if self.Is_Phase_Success_finished(raw_av, self.P_NB): #P_NB Success finished
            found_idxs=np.where(raw_av[self.PAVStart_idx[self.P_HP]:self.PResStart_idx[self.P_HP]] == 1)[0]
            assert len(found_idxs)==1
            if self.Is_Phase_Finished(raw_av, self.P_HP):  # P_HP phase success or error finished
                if self.Is_Phase_Success_finished(raw_av, self.P_HP):
                    return False, found_idxs[0]
                else:
                    return False, found_idxs[0]-1  # Fail finish phase increase 1 the (LHP+1)
            else:
                return True, found_idxs[0]
        else:# P_NB error finished or still in P_NB phase
            return False, -1

    def get_NB_status_On_S_(self, raw_av):
        if not self.Is_Phase_Finished(raw_av,self.P_NB): #NB phase not finished
            return False, -1
        else:
            found_idxs=np.where(raw_av[self.PAVStart_idx[self.P_NB]:self.PResStart_idx[self.P_NB]] == 1)[0]
            assert len(found_idxs)==1
            if self.Is_Phase_Success_finished(raw_av, self.P_NB):
                return False, found_idxs[0]
            else:
                return False, found_idxs[0] - 1  # Fail finish phase increase 1 the (LNB+1)



