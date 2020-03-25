import numpy as np
import os, pickle,re
import datetime as dt
from data_T5 import R_T5,R_T5_scale, R_T5_skipSwh,R_T5_skipSwh_balance
import config as sc
import pandas as pd

class record_variable:
    def __init__(self, lc, checked_name="loss"):
    #def __init__(self, dirwp, checked_name="loss", checked_threadhold=10):
        self.dirwp=lc.record_variable_dir
        #self.dirwp=dirwp
        if not os.path.exists(self.dirwp):os.makedirs(self.dirwp)
        self.checked_name = checked_name
        self.checked_threadhold = lc.record_checked_threahold
        self.F_need_record = False
        self.reset()

    def reset(self):
        self.F_trainer_recorded, self.F_brain_recorded, self.F_process_recorded = False, False, False
        self.RD_trainer, self.RD_brain,self.RD_process=[],[],[]

    def check_need_record(self, inputs):
        lm_names, lm_valuess=inputs

        if len (lm_names)==1:
            assert lm_names==self.checked_name
            self.F_need_record=True if lm_valuess>self.checked_threadhold else False
            if self.F_need_record: self.reset()

        else:
            idx=lm_names.index(self.checked_name)
            self.F_need_record=True if lm_valuess[idx]>self.checked_threadhold else False
            if self.F_need_record: self.reset()
            return self.F_need_record

    def recorder_trainer(self, inputs):
        if self.F_need_record:
            self.RD_trainer = inputs
            self.F_trainer_recorded = True
            return True
        else:
            return False

    def recorder_brainer(self, inputs):
        if self.F_need_record:
            self.RD_brain= inputs
            self.F_brain_recorded = True
            return True
        else:
            return False

    def recorder_process(self, inputs):
        if self.F_need_record:
            self.RD_process = inputs
            self.F_process_recorded = True
            return True
        else:
            return False

    def saver(self):
        if self.F_need_record and self.F_trainer_recorded and self.F_brain_recorded and self.F_process_recorded:
            fn=dt.datetime.utcnow().strftime('%Y-%m-%d_%H:%M:%S_%f')[:-3]+".pkl"
            fnwp=os.path.join(self.dirwp, fn)

            with open(fnwp, 'w') as f:  # Python 3: open(..., 'wb')
                pickle.dump([self.RD_trainer, self.RD_brain,self.RD_process], f)
            self.F_need_record = False


    def loader(self, fn):
        fnwp=os.path.join(self.dirwp, fn)
        with open(fnwp) as f:  # Python 3: open(..., 'rb')
            return pickle.load(f)


    def make_test_variables(self):
        s_lv=np.ones((500,20,17), dtype=float)
        s_sv = np.ones((500, 20, 25,2), dtype=float)*2
        s_av = np.ones((500, 8), dtype=float)*3
        a=np.ones((500, 1), dtype=float)*4
        r=np.ones((500, 1), dtype=float)*5
        s__lv=np.ones((500,20,17), dtype=float)*6
        s__sv = np.ones((500, 20, 25,2), dtype=float)*7
        s__av = np.ones((500, 8), dtype=float)*8
        Done=np.ones((500,1), dtype=bool)
        Support_view=np.array([{"this_trade_day_Nprice":idx, "this_trade_day_hfq_ratio":idx*1.0/100,
                                "stock":"SH6000000","date":"20180401"} for idx in range(500)])
        train_count=100
        saved_trc=20

        lm_values=range (4)
        lm_names =["loss", "M_policy_loss","M_value_loss","M_entropy","M_entropy"]

        return [[s_lv,s_sv,s_av,a,r,s__lv,s__sv,s__av, Done, Support_view],[train_count, saved_trc]],lm_names,lm_values
class record_variable2(record_variable):
    def __init__(self,lc, checked_name="loss"):
        record_variable.__init__(self,lc, checked_name=checked_name)
        self.lc=lc
        self.system_name = lc.RL_system_name
        self.data_name=lc.data_name
        self.class_env_read_data=globals()[lc.CLN_env_read_data]
        self.checked_threadhold = np.NaN
        self.dic_dh_index={}
        self.l_sdh=[]


    def _compress_content(self, inputs):
        C_RD_trainer, C_RD_brain, C_RD_process=inputs
        s_lv, s_sv, s_av, a, r, s__lv, s__sv, s__av, done_flag, l_support_view=C_RD_trainer
        trainer_content=[]
        for idx in range(len(s_lv)):
            to_saved_item_dic={
                "stock":  l_support_view[idx][0,0]["stock"],
                "s_period_idx":l_support_view[idx][0,0]["period_idx"],
                "s_idx_in_period":l_support_view[idx][0,0]["idx_in_period"],
                "s_date":l_support_view[idx][0,0]["date"],
                "s__period_idx":l_support_view[idx][0,0]["_support_view_dic"]["period_idx"],
                "s__idx_in_period":l_support_view[idx][0,0]["_support_view_dic"]["idx_in_period"],
                "s__date":l_support_view[idx][0,0]["_support_view_dic"]["date"],
                "action_taken":l_support_view[idx][0,0]["action_taken"],
                "action_return_message":l_support_view[idx][0,0]["action_return_message"],
                "action":a[idx][0],
                "TD_reward":r[idx][0],
                "s_av":s_av[idx][0],
                "s__av": s__av[idx][0],
                "flag_force_sell":l_support_view[idx][0,0]['flag_force_sell'],
                "old_ap": l_support_view[idx][0, 0]['old_ap']
            }
            to_saved_item_dic["SdisS_"]=l_support_view[idx][0, 0]['SdisS_']
            trainer_content.append(to_saved_item_dic)

        tr_dic={
            "current_train_count":C_RD_process[0],
            "saved_train_count":C_RD_process[1]
        }
        compressed_inputs=[trainer_content,C_RD_brain, tr_dic]
        return compressed_inputs

    def _get_fnwp(self,sc, cc):
        return  os.path.join(self.dirwp, "SC{0}_CC{1}.pkl".format(sc,cc))


    def check_need_record(self, inputs):
        lm_names, lm_valuess=inputs
        self.F_need_record = True
        self.reset()
        return self.F_need_record

    def saver(self):
        if self.F_need_record and self.F_trainer_recorded and self.F_brain_recorded and self.F_process_recorded:
            compressed_inputs = self._compress_content([self.RD_trainer, self.RD_brain, self.RD_process])
            fnwp = self._get_fnwp(compressed_inputs[2]["saved_train_count"],compressed_inputs[2]["current_train_count"])
            with open(fnwp, 'w') as f:  # Python 3: open(..., 'wb')
                pickle.dump(compressed_inputs, f)
            self.F_need_record = False

    def read_SC_CC_compressed_data(self, sc, cc):
        raw_states, raw_loss, raw_sccc=pickle.load(open(self._get_fnwp(sc, cc),"r"))
        return raw_states, raw_loss, raw_sccc


    def read_SC_CC_data(self, sc, cc):
        raw_states, raw_loss, raw_sccc=self.read_SC_CC_compressed_data(sc,cc)
        assert raw_sccc["current_train_count"]==cc
        assert raw_sccc["saved_train_count"] == sc
        l_s_lv = []
        l_s_sv = []
        l_s_av = []
        l_a    = []
        l_r    = []
        l_s__lv = []
        l_s__sv = []
        l_s__av = []
        l_support_view = []

        for raw_state in raw_states:
            stock=raw_state["stock"]
            try:
                sdf_idx=self.dic_dh_index[stock]
            except:
                sdf=self.class_env_read_data(self.data_name, stock)
                if not sdf.flag_prepare_data_ready:
                    assert False, "unexpected error {0} data not availble".format(stock)
                self.l_sdh.append(sdf)
                self.dic_dh_index[stock]=len(self.l_sdh)-1
                sdf_idx = self.dic_dh_index[stock]

            period_idx=raw_state["s_period_idx"]
            idx_in_period=raw_state["s_idx_in_period"]
            [s_lv, s_sv], s_support_view = self.l_sdh[sdf_idx].read_one_day_data_by_index(period_idx, idx_in_period-1)
            assert s_support_view["stock"]==stock
            period_idx=raw_state["s__period_idx"]
            idx_in_period=raw_state["s__idx_in_period"]
            [s__lv, s__sv], s__support_view = self.l_sdh[sdf_idx].read_one_day_data_by_index(period_idx, idx_in_period)

            l_s_lv.append(s_lv)
            l_s_sv.append(s_sv)
            l_s_av.append(np.expand_dims(raw_state["s_av"], axis=0))
            l_a.append(np.expand_dims(raw_state["action"], axis=0))
            l_r.append(np.array([raw_state["TD_reward"]]))
            l_s__lv.append(s__lv)
            l_s__sv.append(s__sv)
            l_s__av.append(np.expand_dims(raw_state["s__av"], axis=0))
            support_view={}
            support_view["stock"] = raw_state ["stock"]
            support_view["date"] = raw_state ["s_date"]
            support_view["action_return_message"] = raw_state ["action_return_message"]
            support_view["action_taken"] = raw_state ["action_taken"]
            support_view["period_idx"] = raw_state ["s_period_idx"]
            support_view["idx_in_period"] = raw_state ["s_idx_in_period"]
            support_view["flag_force_sell"] = raw_state["flag_force_sell"]
            support_view["old_ap"] = raw_state["old_ap"]
            support_view["SdisS_"] = raw_state["SdisS_"]
            l_support_view.append(np.array([[support_view]]))

        RD_trainer = [l_s_lv, l_s_sv, l_s_av, l_a, l_r, l_s__lv, l_s__sv, l_s__av,False,l_support_view]
        RD_brain = raw_loss
        RD_process = [raw_sccc["current_train_count"],raw_sccc["saved_train_count"]]
        return RD_trainer, RD_brain,RD_process

    def read_SC_CC_data_loss_sccc(self, sc, cc):
        fnwp = self._get_fnwp(sc, cc)
        raw_data=pickle.load(open(fnwp,"r"))
        compressed_RD_trainer, raw_loss, raw_sccc=raw_data
        return compressed_RD_trainer,raw_loss, raw_sccc

    def read_SC_CC_data_raw(self, sc, cc):
        assert False, "legacy debug purpose, not support in class {0}".format(self.__class__.__name__)

class record_variable3(record_variable2):
    def __init__(self,lc, checked_name="loss"):
        record_variable2.__init__(self,lc, checked_name=checked_name)
    def _get_fnwp(self,sc, cc):
        current_dir = os.path.join(self.dirwp, "SC{0}".format(sc))
        if not os.path.exists(current_dir): os.mkdir(current_dir)
        return  os.path.join(current_dir, "CC{0}.pikle".format(cc))


class record_send_to_server:
    '''
    pickle data structure
    a[0]------------------how many record
    a[0][0]---------------state
    a[0][1]---------------action
    a[0][2]---------------reward
    a[0][3]---------------state_
    a[0][4]---------------last_flag
    a[0][5]---------------support_view
    '''

    def __init__(self,dirwp,flag_recorder=True):
        self.flag_recorder=flag_recorder
        self.dirwp=dirwp
        if not os.path.exists(self.dirwp): os.makedirs(dirwp)
    def saver(self, inputs):
        fn = dt.datetime.utcnow().strftime('%Y-%m-%d_%H:%M:%S_%f')[:-3] + ".pkl"
        fnwp = os.path.join(self.dirwp, fn)
        with open(fnwp, 'w') as f:  # Python 3: open(..., 'wb')
            pickle.dump(inputs, f)
        idxfn = fn[:-4]+"_idx" + ".csv"
        idxfnwp = os.path.join(self.dirwp, idxfn)
        l_index=[[item[5][0,0]["stock"],item[5][0,0]["date"]]for item in inputs]
        df=pd.DataFrame(l_index, columns=["stock","date"])
        df.to_csv(idxfnwp, index=False)

class record_sim_stock_data:
    def __init__(self,dirwp,stock,flag_recorder=True):
        self.flag_recorder=flag_recorder
        if not os.path.exists(dirwp): os.makedirs(dirwp)
        self.dir_base=dirwp
        self.dirwp=os.path.join(self.dir_base, stock)
        if not os.path.exists(self.dirwp): os.makedirs(self.dirwp)
    def saver(self, inputs, date):
        fn = date + ".pkl"
        fnwp = os.path.join(self.dirwp, fn)
        with open(fnwp, 'w') as f:  # Python 3: open(..., 'wb')
            pickle.dump(inputs, f)
    def loader(self, stock, date):
        fnwp=os.path.join(self.dir_base, stock, "{0}.pkl".format(date))
        if not os.path.exists(fnwp):
            return None
        [lv,sv],support_view=pickle.load(open(fnwp,"r"))
        return lv, sv, support_view


class get_recorder_OS_losses:
    def __init__(self,system_name):
        param_fnwp = os.path.join(sc.base_dir_RL_system, system_name, "config.json")
        if not os.path.exists(param_fnwp):
            raise ValueError("{0} does not exisit".format(param_fnwp))
        self.lgc = sc.gconfig()
        self.lgc.read_from_json(param_fnwp)
        assert self.lgc.P2_current_phase=="Train_Sell"
        self.ir = globals()[self.lgc.CLN_record_variable](self.lgc)


        '''
        self.L_tcs = [int(re.findall(r'\w+T(\d+).h5py', fn)[0]) for fn in os.listdir(self.lgc.brain_model_dir)
                 if fn.startswith("train_model_AIO_")]
        self.L_tcs.sort()
        self.L_tcs.pop(-1)  # th unfinished trained tcs model
        '''
        LETs= [int(re.findall(r'\w+T(\d+).h5py', fn)[0]) for fn in os.listdir(self.lgc.brain_model_dir)
           if fn.startswith("train_model_AIO_")]
        LETs.sort()
        LETs.pop(-1)
        self.L_tcs= [idx*self.lgc.num_train_to_save_model for idx in range(LETs[-1]/self.lgc.num_train_to_save_model+1)]
        #self.L_tcs.pop(0)


        des_dir = self.lgc.system_working_dir
        for sub_dir in ["analysis","pre_losses"]:
            des_dir=os.path.join(des_dir,sub_dir)
            if not os.path.exists(des_dir): os.mkdir(des_dir)
        self.des_dir=des_dir
        if not os.path.exists(self.des_dir):
            os.makedirs(self.des_dir)

    def get_losses_per_stc(self, stc):
        fnwp = os.path.join(self.des_dir, "loss_stc_{0}.csv".format(stc))
        if os.path.exists(fnwp):
            print "{0} already exist".format(fnwp)
            df = pd.read_csv(fnwp)
            return df
        df = pd.DataFrame(columns=['stc', 'tc', "valid_count","valid_sell_count", "invalid_count",
                    'loss', 'M_policy_loss', 'M_value_loss', 'M_entropy', 'M_state_value',"M_reward"])
        for tci in range(self.lgc.num_train_to_save_model):
            tc = tci + int(stc)
            print "handling saved_tc: {0}  tc:{1}".format(stc, tc)
            compressed_RD_trainer,raw_loss, raw_sccc = self.ir.read_SC_CC_data_loss_sccc(stc, tc)
            count_valid_record_sell = 0
            count_valid_record_sell_no_action =0
            count_invalid_record = 0
            lr=[]
            for record in compressed_RD_trainer:
                if record["action"][0] == 1:
                    if record["action_return_message"] in ["No_holding", "Tinpai"]:
                        count_invalid_record += 1
                    else:
                        count_valid_record_sell += 1
                elif record["action"][1] == 1:
                    count_valid_record_sell_no_action += 1
                else:
                    count_invalid_record += 1
                lr.append(record["TD_reward"][0])
            npr=np.array(lr)
            count_valid_record=count_valid_record_sell+count_valid_record_sell_no_action

            #row_to_add = [stc, tc,count_valid_record, count_valid_record_sell, count_invalid_record] + raw_loss[1]+[npr.mean()]

            row_to_add = [stc, tc, count_valid_record, count_valid_record_sell, count_invalid_record]
            for idx,item_name in enumerate(['loss', 'M_policy_loss', 'M_value_loss', 'M_entropy', 'M_state_value']):
                assert raw_loss[0][idx]==item_name, "{0} {1} miss match".format(raw_loss[0][idx],item_name)
                row_to_add.append(raw_loss[1][idx])
            row_to_add.append(npr.mean())

            df.loc[len(df)] = row_to_add
        df.to_csv(fnwp, index=False)
        return df

    def get_losses(self):
        for stc in self.L_tcs:
            self.get_losses_per_stc(stc)

import miscellaneous
class record_data_verify:
    dir_buffer="record_send_buffer"
    dir_sim="record_sim"
    dir_tstate="record_state"
    def __init__(self, system_name):
        #load index of "record_send_buffer"
        self.system_name=system_name
        lc=miscellaneous.load_config_from_system(self.system_name)
        difwp=os.path.join(sc.base_dir_RL_system, system_name,self.dir_buffer)
        lfn=[fn for fn in os.listdir(difwp) if fn.endswith(".csv")]
        self.dfi=pd.DataFrame()
        for fn in lfn:
            fnwp=os.path.join(difwp, fn)
            df=pd.read_csv(fnwp)
            df["fn"]=fn[:-8]+".pkl"
            self.dfi=self.dfi.append(df, ignore_index=True)
        self.i_sim=record_sim_stock_data(os.path.join(sc.base_dir_RL_system, system_name,self.dir_sim),"SH600036")
        self.i_state = globals()[lc.CLN_record_variable](lc)
    def read_buffer_data(self,stock, date):
        dfr=self.dfi[(self.dfi.stock==stock) &(self.dfi.date==int(date))]
        if len(dfr)!=1:
            return None
        elif len(dfr)==1:
            buffer_items=pickle.load(open(os.path.join(sc.base_dir_RL_system, self.system_name, self.dir_buffer, dfr.iloc[0].fn), "r"))
            for item in buffer_items:
                if item[5][0,0]["stock"]==stock and item[5][0,0]["date"]==date:
                    state, action, reward, state_, last_flag, support_view = item
                    return state, action, reward, state_, last_flag, support_view
            else:
                raise ValueError("Unexpected Error {0} {1} cound not found in {2}".format(stock, date, dfr.fn))

    def analysis(self):
        #import recorder
        #i = recorder.record_analysis("LHP2")
        i=self
        cs = i.i_state.read_SC_CC_data(0, 0)
        rs = i.i_state.read_SC_CC_data_raw(0,0)

        #check cs first data
        idx=1
        cs_state_lv=cs[0][0][idx]

        rs_state_lv=rs[0][0][idx]

        date  = cs[0][8][idx][0, 0]["date"]
        stock = cs[0][8][idx][0, 0]["stock"]

        bs=i.read_buffer_data(stock, date)

        print (bs[0][0] == rs_state_lv).all()

        ss=i.i_sim.loader(stock, date)
        #ss[0]==cs_state_lv

        cs_state_sv = cs[0][1][idx]
        rs_state_sv = rs[0][1][idx]
        print (cs_state_sv == rs_state_sv).all()

        cs_state_lv_ = cs[0][5][idx]
        rs_state_lv_ = rs[0][5][idx]
        print (cs_state_lv_ == rs_state_lv_).all()

        cs_state_sv_ = cs[0][6][idx]
        rs_state_sv_ = rs[0][6][idx]
        print (cs_state_sv_ == rs_state_sv_).all()

        cs_state_av = cs[0][2][idx]
        rs_state_av = rs[0][2][idx]
        print cs_state_av
        print rs_state_av

        cs_state_av_ = cs[0][7][idx]
        rs_state_av_ = rs[0][7][idx]
        print cs_state_av_
        print rs_state_av_

        cs_action = cs[0][3][idx]
        rs_action = rs[0][3][idx]
        print cs_action
        print rs_action

        cs_reward = cs[0][4][idx]
        rs_reward = rs[0][4][idx]
        print cs_reward
        print rs_reward

