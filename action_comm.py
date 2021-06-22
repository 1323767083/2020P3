import numpy as np
class actionOBOS:
    Only_Action_Dimention = 2
    def __init__(self,action_type):
        assert action_type in ["OB", "OS"], "{0} does not support {1}".format(self.__class__.__name__,self.action_type)
        self.action_type=action_type

    #used in config
    def sanity_check_action_config(self,lc):
        assert lc.action_type_dict[0] == "Buy"
        assert lc.action_type_dict[1] == "No_action"
        assert lc.action_type_dict[2] == "Sell"
        assert lc.action_type_dict[3] == "No_action"

    #used in nets choose action
    def I_nets_choose_action(self,probs):
        return np.random.choice([2, 3], p=probs) if self.action_type=="OS" else np.random.choice([0, 1], p=probs)

    #used in TD_buffer,  from TD_buffer to train
    def I_TD_buffer(self,actionarray):
        assert actionarray.shape[1]==4
        assert self.action_type == "OB"
        return actionarray[:, :2]

    def I_A3C_worker_explorer(self, actual_action,ap):
        assert self.action_type=="OB"
        a_onehot = np.zeros((1, 4))
        a_onehot[0, actual_action] = 1
        old_ap=-1 if actual_action >= 2 else ap[actual_action]
        assert type(old_ap) is np.float32 or type(old_ap) is int
        return a_onehot,old_ap

