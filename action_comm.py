import numpy as np
class actionOBOS:
    Only_Action_Dimention = 2
    def __init__(self,action_type):
        assert action_type in ["OB", "OS", "B3","B4"],"{0} does not support {1}".format(self.__class__.__name__, self.action_type)
        self.action_type=action_type

    #used in config
    def sanity_check_action_config(self,lc):
        if self.action_type in ["OB", "OS"]:
            assert lc.action_type_dict[0] == "buy"
            assert lc.action_type_dict[1] == "no_action"
            assert lc.action_type_dict[2] == "sell"
            assert lc.action_type_dict[3] == "no_action"
        elif self.action_type in ["B3","B4"]:
            assert lc.action_type_dict[0] == "buy"
            assert lc.action_type_dict[1] == "no_action"
            assert lc.action_type_dict[2] == "sell"
            assert lc.action_type_dict[3] == "no_action"
            assert lc.action_type_dict[4] == "no_trans"   # used in OB training


    #used in nets choose action
    def I_nets_choose_action(self,prob):
        if self.action_type in ["OB", "OS"]:
            assert len(prob)==2
            return np.random.choice([0, 1], p=prob) if self.action_type == "OB" else np.random.choice([2, 3], p=prob)
        elif self.action_type == "B3":
            assert len(prob) == 3
            return  np.random.choice([0, 1,4], p=prob)
        elif self.action_type == "B4":
            assert len(prob) == 4
            action_trans=np.random.choice([0, 1], p=prob[:2])
            if action_trans==0:
                return np.random.choice([0, 1], p=prob[2:])
            else:
                return 4

    #used in TD_buffer,  from TD_buffer to train
    def I_TD_buffer(self,actionarray):
        if self.action_type in ["OB", "OS"]:
            assert actionarray.shape[1]==4
            return actionarray[:,:2] if self.action_type == "OB" else actionarray[:,2:]
        elif self.action_type in ["B3","B4"]:
            assert actionarray.shape[1]==5
            adj_aarry= np.concatenate((actionarray[:,:2], actionarray[:,4:]), axis=-1)
            assert adj_aarry.shape[1]==3
            return adj_aarry


    # used in A3C_worker
    def _action_2_actionarray(self, action):
        a_onehot_shape= (1,4) if self.action_type in ["OB", "OS"] else (1, 5)  # else self.action_type in ["B3", "B4"]
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
        elif self.action_type == "B3":
            assert len(AParray)==3
            if action in [0,1]:
                return AParray[action]
            elif action==4:
                return AParray[2]
            else:
                assert action in [2,3]
                return -1
        elif self.action_type == "B4":
            assert len(AParray)==4
            if action in [0,1]:
                return [AParray[0],AParray[2+action]]
            elif action==4:
                return [AParray[1],0.5]  # not use 0 but 0.5 is to avoid very large number ap/0ld_ap
            else:
                assert action in [2,3]
                return [-1,-1]  # should be removed at TD buffer

    def _get_actual_action(self,support_view_dic):
        if self.action_type in ["OB", "OS"]:
            if support_view_dic["action_taken"]=="Sell" and support_view_dic["action_return_message"]=="Success":
                return 2 #"Sell"
            elif support_view_dic["action_taken"]=="Buy" and support_view_dic["action_return_message"]=="Success":
                return 0  #"Buy"
            else:  # this is for unsuccessful buy or sell and no action
                return 3 if support_view_dic["holding"] != 0 else 1  # "No_action"  # Failed buy and sell for simulator is not action
        elif self.action_type in ["B3","B4"]:
            if support_view_dic["action_taken"]=="Sell" and support_view_dic["action_return_message"]=="Success":
                return 2 #"Sell"
            elif support_view_dic["action_taken"]=="Buy" and support_view_dic["action_return_message"]=="Success":
                return 0  #"Buy"
            elif support_view_dic["action_taken"]=="No_trans" and support_view_dic["action_return_message"]=="No_trans":
                return 4  #"No Trans"
            else:  # this is for unsuccessful buy or sell and no action
                return 3 if support_view_dic[
                                "holding"] != 0 else 1  # "No_action"  # Failed buy and sell for simulator is not action

    def I_A3C_worker_explorer(self,support_view_dic, ap):
        actual_action = self._get_actual_action(support_view_dic)
        a_onehot = self._action_2_actionarray(actual_action)
        old_ap = self._get_prob_from_AParray(ap, actual_action)
        if self.action_type in ["OB", "OS"]:
            assert a_onehot.shape==(1,4)
            assert type(old_ap) is float
        elif self.action_type == "B3":
            assert a_onehot.shape == (1, 5)
            assert type(old_ap) is float
        elif self.action_type == "B4":
            assert a_onehot.shape == (1, 5)
            assert type(old_ap) is list
        return a_onehot, old_ap


