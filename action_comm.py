import numpy as np
class actionOBOS:
    Only_Action_Dimention = 2
    def __init__(self,action_type):
        assert action_type in ["OB", "OS","B32"],"{0} does not support {1}".format(self.__class__.__name__, self.action_type)
        self.action_type=action_type

    #used in config
    def sanity_check_action_config(self,lc):
        if self.action_type in ["OB", "OS"]:
            assert lc.action_type_dict[0] == "Buy"
            assert lc.action_type_dict[1] == "No_action"
            assert lc.action_type_dict[2] == "Sell"
            assert lc.action_type_dict[3] == "No_action"
        elif self.action_type in ["B32"]:
            assert lc.action_type_dict[0] == "Buy"
            assert lc.action_type_dict[1] == "No_action"
            assert lc.action_type_dict[2] == "Sell"
            assert lc.action_type_dict[3] == "No_action"
            assert lc.action_type_dict[4] == "No_trans"   # used in OB training


    #used in nets choose action
    def I_nets_choose_action(self,inputs):
        assert self.action_type in ["OB", "OS","B32"]
        if self.action_type=="OS":
            prob=inputs
            return np.random.choice([2, 3], p=prob)
        elif self.action_type=="OB":
            prob = inputs
            return np.random.choice([0, 1], p=prob)
            #Flag_Random_Explore=np.random.choice([0, 1], p=[0.8,0.2])
            #if Flag_Random_Explore:
            #    return 0
            #else:
            #    return np.random.choice([0, 1], p=prob)
        elif self.action_type == "B32":
            prob, av_item, lc_LNB = inputs
            if av_item[-lc_LNB-1]!=1:
                #action = np.random.choice([0, 1], p=prob[:2])
                psum=sum(prob[:2])
                adj_prob=[pi/psum for pi in prob[:2]]
                action = np.random.choice([0, 1], p=adj_prob)
            else:
                action = np.random.choice([0, 1, 4], p=prob)
            return action
        else:
            assert  False, "Only support OB OS B32 in I_nets_choose_action not support {0}".format(self.action_type)



    #used in TD_buffer,  from TD_buffer to train
    def I_TD_buffer(self,actionarray):
        if self.action_type in ["OB", "OS"]:
            assert actionarray.shape[1]==4
            return actionarray[:,:2] if self.action_type == "OB" else actionarray[:,2:]
        elif self.action_type in ["B32"]:
            assert actionarray.shape[1]==5
            adj_aarry= np.concatenate((actionarray[:,:2], actionarray[:,4:]), axis=-1)
            assert adj_aarry.shape[1]==3
            return adj_aarry
        else:
            assert False

    # used in A3C_worker
    def _action_2_actionarray(self, action):
        a_onehot_shape= (1,4) if self.action_type in ["OB", "OS"] else (1, 5)  # else self.action_type in ["B32"]
        a_onehot = np.zeros(a_onehot_shape)
        a_onehot[0, action] = 1
        return a_onehot


    def _get_prob_from_AParray(self,AParray, action):
        if self.action_type == "OS":
            assert len(AParray) == 2
            return -1 if action<2 else AParray[action-2]  # -1  record will be removed by TD_buffer and sainity cheked the remove at optimize_com
        elif self.action_type == "OB":
            assert len(AParray) == 2
            return -1 if action>=2 else AParray[action]  # -1  record will be removed by TD_buffer and sainity cheked the remove at optimize_com
        elif self.action_type in ["B32"]:
            assert len(AParray)==3
            if action in [0,1]:
                return AParray[action]
            elif action==4:
                return AParray[2]
            else:
                assert action in [2,3]
                return -1
        else:
            assert False

    def I_A3C_worker_explorer(self, actual_action,ap):
        a_onehot = self._action_2_actionarray(actual_action)
        old_ap = self._get_prob_from_AParray(ap, actual_action)
        if self.action_type in ["OB", "OS"]:
            assert a_onehot.shape==(1,4)
            assert type(old_ap) is np.float32 or type(old_ap) is int
        elif self.action_type in ["B32"]:

            assert a_onehot.shape == (1, 5)
            assert type(old_ap) is np.float32 or type(old_ap) is int,"{0}, {1}".format(type(old_ap),old_ap)
        else:
            assert False
        return a_onehot, old_ap
