import numpy as np
class actionOBOS:
    Only_Action_Dimention = 2
    def __init__(self,action_type):
        assert action_type in ["OB", "OS"], "{0} does not support {1}".format(self.__class__.__name__,self.action_type)
        self.action_type=action_type
    #used in config
    def sanity_check_action_config(self,lc):
        if self.action_type in ["OB", "OS"]:
            assert lc.action_type_dict[0] == "Buy"
            assert lc.action_type_dict[1] == "No_action"
            assert lc.action_type_dict[2] == "Sell"
            assert lc.action_type_dict[3] == "No_action"
        else:
            assert False
    #used in nets choose action
    def I_nets_choose_action(self,inputs):
        assert self.action_type in ["OB", "OS"]
        if self.action_type=="OS":
            prob=inputs
            return np.random.choice([2, 3], p=prob)
        elif self.action_type=="OB":
            prob = inputs
            return np.random.choice([0, 1], p=prob)
        else:
            assert  False, "Only support OB OS in I_nets_choose_action not support {0}".format(self.action_type)

    #used in TD_buffer,  from TD_buffer to train
    def I_TD_buffer(self,actionarray):
        if self.action_type in ["OB", "OS"]:
            assert actionarray.shape[1]==4
            return actionarray[:,:2] if self.action_type == "OB" else actionarray[:,2:]
        else:
            assert False

    # used in A3C_worker
    def _action_2_actionarray(self, action):
        a_onehot_shape = (1, 4)
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
        else:
            assert False

    def I_A3C_worker_explorer(self, actual_action,ap):
        a_onehot = self._action_2_actionarray(actual_action)
        old_ap = self._get_prob_from_AParray(ap, actual_action)
        if self.action_type in ["OB", "OS"]:
            assert a_onehot.shape==(1,4)
            assert type(old_ap) is np.float32 or type(old_ap) is int
        else:
            assert False
        return a_onehot, old_ap
