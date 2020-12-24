'''
from nets_agent_base import *
from action_comm import actionOBOS
class LHPP2V2_Agent(net_aget_base):
    def __init__(self,lc):
        net_aget_base.__init__(self,lc)

    def choose_action(self, state, calledby="Eval"):
        assert self.lc.P2_current_phase == "Train_Sell"
        lv, sv, av = state
        actions_probs, SVs = self.predict(state)
        l_a = []
        l_ap = []
        l_sv = []
        for sell_prob, SV, av_item in zip(actions_probs, SVs, av):
            assert len(sell_prob) == 2, sell_prob
            flag_holding=self.i_cav.Is_Holding_Item(av_item)
            if flag_holding:
                #action = np.random.choice([2, 3], p=action_probs)
                action =self.i_action.I_nets_choose_action(sell_prob)
                l_a.append(action)
                l_ap.append(sell_prob.ravel())
            else:  # not have holding
                action = 0
                l_a.append(action)
                l_ap.append(sell_prob.ravel())
            l_sv.append(SV[0])
        return l_a, l_ap, l_sv
'''