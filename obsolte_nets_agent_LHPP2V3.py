'''
import numpy as np
from nets_agent_base import *
from action_comm import actionOBOS
class LHPP2V3_Agent(net_aget_base):
    def __init__(self,lc):
        net_aget_base.__init__(self,lc)

    def choose_action(self,state,calledby):
        assert self.lc.P2_current_phase == "Train_Buy"
        lv, sv, av = state
        buy_probs, buy_SVs = self.predict([lv, sv,av])
        if not hasattr(self, "OS_agent"):
            self.OS_agent = V2OS_4_OB_agent(self.lc,self.lc.P2_sell_system_name, self.lc.P2_sell_model_tc)
            self.i_OS_action=actionOBOS("OS")
        sel_probs, sell_SVs = self.OS_agent.predict(state)
        l_a = []
        l_ap = []
        l_sv = []
        for buy_prob, sell_prob, buy_sv, sell_sv, av_item in zip(buy_probs,sel_probs,buy_SVs,sell_SVs,av):
            assert len(buy_prob)==2
            assert len(sell_prob) == 2
            flag_holding=self.i_cav.Is_Holding_Item(av_item)
            if flag_holding:
                #action = np.random.choice([2, 3], p=sell_prob)
                action = self.i_OS_action.I_nets_choose_action(sell_prob)
                l_a.append(action)
                l_ap.append(np.zeros_like(sell_prob))  # this is add zero and this record will be removed by TD_buffer before send to server for train
                l_sv.append(sell_sv[0])
            else: # not have holding
                if calledby=="Explore":
                    #TODO need to find whether configure in config needed
                    Flag_random_explore=np.random.choice([0, 1], p=[0.5,0.5])

                    if Flag_random_explore:
                        action=0
                    else:
                        action = self.i_action.I_nets_choose_action(buy_prob)
                elif calledby=="Eval":
                    action = self.i_action.I_nets_choose_action(buy_prob)
                else:
                    assert False, "Only support Explore and Eval as calledby not support {0}".format(calledby)
                l_a.append(action)
                l_ap.append(buy_prob)
                l_sv.append(buy_sv[0])
        return l_a, l_ap,l_sv

    def choose_action_CC(self,state,calledby):
        assert self.lc.P2_current_phase == "Train_Buy"
        lv, sv, av = state
        buy_probs, buy_SVs = self.predict([lv, sv,av])
        if not hasattr(self, "OS_agent"):
            self.OS_agent = V2OS_4_OB_agent(self.lc,self.lc.P2_sell_system_name, self.lc.P2_sell_model_tc)
            self.i_OS_action=actionOBOS("OS")
        sel_probs, sell_SVs = self.OS_agent.predict(state)
        #TODO check whether need to convert to list and if there is effecient way to convert to list
        l_buy_a  = [self.i_action.I_nets_choose_action(buy_prob) for buy_prob in buy_probs ]
        l_sell_a  = [self.i_OS_action.I_nets_choose_action(sell_prob) for sell_prob in sel_probs ]
        return l_buy_a,l_sell_a  # This only used in CC eval ,so AP and sv information in not necessary

'''