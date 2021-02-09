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

    def Init_Phase_State_V2_Eval(self):
        self.CuPs_force_flag = [True, False]

    def Init_Phase_State_V3_Explore(self):
        #self.CuPs_force_flag = [True, True]
        self.CuPs_force_flag = [False, True]
    def Init_Phase_State_V3_Eval(self):
        #self.CuPs_force_flag = [False, False]
        self.CuPs_force_flag = [False, True]  # V3 is to evaluate buy agent, so need to forece sell all holding if possible

    def check_need_force_state(self, action):
        if self._Is_Phase_Last_Step() and self.CuPs_force_flag[self.CuP]:
            adj_action=self.CuPs_exit_actions[self.CuP]
            return adj_action
        else:
            return action


    def reset_phase_state(self,l_av_inform,Flag_Force_Next_Reset):
        self.CuP=self.P_init
        self.CuPs_idx = [0 for _ in range(self.P_Num)]
        raw_av=self.i_av.fabricate_av(self.CuPs_idx, self.CuP, l_av_inform, np.NaN, Flag_Force_Next_Reset,False, False)
        return np.array(raw_av).reshape(1, -1)

    def update_phase_state(self, action, return_message, l_inform, PSS_action,Flag_Force_Next_Reset):
        assert PSS_action in [0,1], "Only support PSS_action has value 0,1 not {0}".format(PSS_action)
        if PSS_action==1:
            assert action == 0, "PSS_action=1 only workwith action=0 means multibuy, action cannot {0}".format(action)
            if return_message=="Success":
                self.CuP = self.P_init
                self.CuPs_idx = [0 for _ in range(self.P_Num)] ##TODO check??

        if self._Is_Phase_Last_Step():
            return self._Update_Phase_Last_Step(action, return_message,l_inform,Flag_Force_Next_Reset)
        else:
            return self._Update_Phase_Normal_Step(action, return_message,l_inform,Flag_Force_Next_Reset)


    def _Is_Phase_Last_Step(self):
        return self.CuPs_idx[self.CuP] == self.CuPs_limits[self.CuP] - 1

    def _Update_Phase_Last_Step(self,action, return_message,l_inform,Flag_Force_Next_Reset):
        if action == self.CuPs_exit_actions[self.CuP] and return_message == "Success":
            raw_av = self.i_av.fabricate_av(self.CuPs_idx, self.CuP, l_inform,action,Flag_Force_Next_Reset,
                                            flag_CuP_finished=True,flag_CuP_Successed=True)
            self.CuP += 1  # the fabricate_av should always have current working CuP as input change to next phase use two flag to identify
            Flag_Done= True if self.CuP==self.P_END else False  #next phase is end True or continue in next phase False
        else:
            self.CuPs_idx[self.CuP] += 1
            action = 3 if self.CuP==self.P_HP else 1
            raw_av = self.i_av.fabricate_av(self.CuPs_idx, self.CuP, l_inform,action, Flag_Force_Next_Reset,
                                            flag_CuP_finished=True,flag_CuP_Successed=False)
            Flag_Done = True   #exceed limitation  Error
            # Following is to ensure while HP forece flag set, only unsuccess finish HP is tinpai
            if self.CuP==self.P_END and self.CuPs_force_flag[self.P_HP]:
                assert return_message=="Tinpai"
                self.i_av.set_Tinpai_huaizhang(raw_av[0])  #this is to put tinpai can not sell invest to Tinpai_huaizhang
        return Flag_Done, raw_av,action

    def _Update_Phase_Normal_Step(self, action, return_message,l_inform,Flag_Force_Next_Reset):
        if action == self.CuPs_exit_actions[self.CuP] and return_message == "Success":
            raw_av = self.i_av.fabricate_av(self.CuPs_idx, self.CuP, l_inform,action,Flag_Force_Next_Reset,
                                            flag_CuP_finished=True,flag_CuP_Successed=True)
            self.CuP += 1
            Flag_Done= True if self.CuP==self.P_END else False  #next phase is end True or continue in next phase False
        else:
            self.CuPs_idx[self.CuP] += 1
            action = 3 if self.CuP == self.P_HP else 1
            raw_av = self.i_av.fabricate_av(self.CuPs_idx, self.CuP, l_inform,action,Flag_Force_Next_Reset,
                                            flag_CuP_finished=False,flag_CuP_Successed=False)
            Flag_Done = False   #continue in current phase

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
    #+1   Flag_Force_Next_Reset especially for CC Eval force next is reset flag
    #+1   Whether this is final record for optimize pupose
    #+len(self.lc.account_inform_titles)
    #+len(self.lc.simulator_inform_titles)
    #+len(self.lc.PSS_inform_titles)
    def __init__(self,lc):
        Phase_Data.__init__(self,lc)
        self.len_inform=len(self.lc.account_inform_titles) + len(self.lc.simulator_inform_titles) + len(self.lc.PSS_inform_titles)
        assert self.lc.raw_AV_shape == (self.lc.LNB + 1 +2+ self.lc.LHP + 1+2 +1+1+self.len_inform,)
        self.PAVStart_idx=[0,self.lc.LNB + 1 +2]
        self.PResStart_idx = [self.lc.LNB + 1, self.lc.LNB + 1 +2+ self.lc.LHP + 1]
        self.PFlag_Force_Next_Reset = self.lc.LNB + 1 + 2 + self.lc.LHP + 1 + 2
        self.PFinal_idx= self.lc.LNB + 1 +2+ self.lc.LHP + 1 +2+1
        for idx, title in enumerate(self.lc.account_inform_titles + self.lc.simulator_inform_titles + self.lc.PSS_inform_titles):
            setattr(self, "P{0}".format(title),  self.PFinal_idx+1+idx)

    #These two function are batch
    def get_OS_AV(self,raw_av):
       return raw_av[:,self.PAVStart_idx[self.P_HP]:self.PResStart_idx[self.P_HP]]

    def get_OB_AV(self,raw_av):
        return raw_av[:,self.PAVStart_idx[self.P_NB]:self.PResStart_idx[self.P_NB]]

    # Following functions are single
    def Fresh_Raw_AV(self):
        return [0 for _ in range(self.lc.raw_AV_shape[0])]

    def fabricate_av(self,CuPs_idx,CuP,l_av_inform,actual_action,Flag_Force_Next_Reset,flag_CuP_finished, flag_CuP_Successed):
        assert len(l_av_inform)==len(self.lc.account_inform_titles)+len(self.lc.simulator_inform_titles)
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
        Raw_AV[self.PFlag_Force_Next_Reset]=Flag_Force_Next_Reset
        Raw_AV[self.PFinal_idx + 1:self.PFinal_idx + 1+len(l_av_inform)]=l_av_inform
        Raw_AV[self.PFinal_idx + 1 + len(l_av_inform):self.PFinal_idx + 1 + len(l_av_inform)+1] = [actual_action]
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

    def get_inform_item(self, raw_av,title):
        if not hasattr(self, "P{0}".format(title)):
            return np.NaN
        return raw_av[getattr(self, "P{0}".format(title))]

    def get_inform_in_all(self, raw_av):
        return raw_av[self.PFinal_idx+1:self.PFinal_idx+1+self.len_inform]

    def Is_Force_Next_Reset(self,raw_av):
        return raw_av[self.PFlag_Force_Next_Reset]

    def set_Tinpai_huaizhang(self,raw_av): #this is to store the the Holding_Invest due to tinpai and alsoholding period end
        raw_av[self.PTinpai_huaizhang]=raw_av[self.PHolding_Invest]

class AV_Handler_AV1(AV_Handler):
    def get_OS_AV(self,raw_av):
       return raw_av[:,self.PAVStart_idx[self.P_HP]:self.PAVStart_idx[self.P_HP]+1]

    def get_OB_AV(self,raw_av):
        return raw_av[:,self.PFinal_idx:self.PFinal_idx+1]
