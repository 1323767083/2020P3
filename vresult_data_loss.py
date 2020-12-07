import os,re
from recorder import get_recorder_OS_losses
import config as sc
import pandas as pd
import tensorflow as tf

class loss_summary_recorder:
    def __init__(self,system_name):
        self.system_name=system_name
        param_fnwp = os.path.join(sc.base_dir_RL_system, self.system_name, "config.json")
        if not os.path.exists(param_fnwp):
            raise ValueError("{0} does not exisit".format(param_fnwp))
        self.lgc=sc.gconfig()
        self.lgc.read_from_json(param_fnwp)
        self.loss_field=["loss","M_policy_loss","M_value_loss","M_entropy_loss","M_state_value"]
        self.lfield = self.loss_field + ["M_reward"]
        #self.lfield=["loss","M_policy_loss","M_value_loss","M_entropy","M_state_value","M_reward"]
        self.lstat=["mean","std","min","max","25%","50%","75%"]

    def data_init(self):
        self.i_get_recorder_OS_losses = get_recorder_OS_losses(self.system_name)
        self.i_get_recorder_OS_losses.get_losses()
        des_dir=self.lgc.system_working_dir
        for sub_dir in ["analysis", "pre_losses"]:
            des_dir=os.path.join(des_dir, sub_dir)
            if not os.path.exists(des_dir): os.mkdir(des_dir)
        self.des_dir = des_dir
        self.lstc=[int(re.findall(r"loss_stc_(\d+)", fn)[0]) for fn in os.listdir(self.des_dir) if fn.startswith("loss_stc_")]
        self.lstc.sort()

    def get_column_choices(self):
        return self.lfield, self.lstat

    def get_loss_summary(self):
        if not hasattr(self,"i_get_recorder_OS_losses"):
            self.data_init()
        fnwp_result = os.path.join(self.lgc.system_working_dir, "analysis",
                                   "loss_summary_{0}_{1}.csv".format(self.lstc[0], self.lstc[-1]))
        if os.path.exists(fnwp_result):
            df=pd.read_csv(fnwp_result, header=0)
            return df
        lcolumn=["stc"]
        for field in self.lfield:
            for stat in self.lstat:
                lcolumn.append(field+"__"+stat)
        dfr=pd.DataFrame(columns=lcolumn)
        for stc in self.lstc:
            print("handling ", stc)
            df=self.i_get_recorder_OS_losses.get_losses_per_stc(stc)
            lr=[stc]
            for field in self.lfield:
                sr=df[field].describe()
                lr = lr + [sr[item] for item in self.lstat]
            dfr.loc[len(dfr)]=lr
        dfr.to_csv(fnwp_result,index=False)
        return dfr
    def get_losses_per_stc(self,stc):
        if not hasattr(self,"i_get_recorder_OS_losses"):
            self.data_init()
        df = self.i_get_recorder_OS_losses.get_losses_per_stc(stc)
        return df
class ana_loss_data_tb:
    def __init__(self,system_name):
        self.system_name=system_name
        param_fnwp = os.path.join(sc.base_dir_RL_system, self.system_name, "config.json")
        if not os.path.exists(param_fnwp):
            raise ValueError("{0} does not exisit".format(param_fnwp))
        self.lgc=sc.gconfig()
        self.lgc.read_from_json(param_fnwp)

        self.lfield = ["loss", "M_policy_loss", "M_value_loss", "M_entropy_loss", "M_state_value"]
        self.lstat=["mean","std","min","max","25%","50%","75%"]

    def data_init(self):
        des_dir=self.lgc.system_working_dir
        for sub_dir in ["analysis","pre_losses_from_tb"]:
            des_dir =os.path.join(des_dir,sub_dir)
            if not os.path.exists(des_dir): os.mkdir(des_dir)
        self.des_dir=des_dir

        dir_model=os.path.join(self.lgc.system_working_dir,"model")
        self.lstc = [int(re.findall(r'\w+T(\d+).h5', fn)[0]) for fn in os.listdir(dir_model) if "train_model_AIO_" in fn]
        self.lstc.sort()
        self.lstc.pop(-1)
        if not self.check_loss_from_tb_avalaible(self.lstc):
            self.get_loss_from_tb()

    def get_column_choices(self):
        return self.lfield, self.lstat

    def get_loss_summary(self):
        if not hasattr(self,"lstc"):
            self.data_init()
        fnwp_result=os.path.join(self.lgc.system_working_dir, "analysis","loss_summary_{0}_{1}.csv".format(self.lstc[0], self.lstc[-1]))
        if os.path.exists(fnwp_result):
            df=pd.read_csv(fnwp_result, header=0)
            return df
        lcolumn=["stc"]
        for field in self.lfield:
            for stat in self.lstat:
                lcolumn.append(field+"__"+stat)
        dfr=pd.DataFrame(columns=lcolumn)
        for stc in self.lstc:
            print("handling ", stc)
            df=self.get_losses_per_stc(stc)
            if len(df)==0:
                continue
            lr=[stc]
            for field in self.lfield:
                sr=df[field].describe()
                lr = lr + [sr[item] for item in self.lstat]
            dfr.loc[len(dfr)]=lr
        dfr.to_csv(fnwp_result,index=False)
        return dfr
    def get_losses_per_stc(self,stc):
        if not hasattr(self,"lstc"):
            self.data_init()
        fnwp = os.path.join(self.des_dir, "loss_stc_{0}.csv".format(stc))
        if os.path.exists(fnwp):
            df = pd.read_csv(fnwp)
            return df
        else:
            return pd.DataFrame([])

    def check_loss_from_tb_avalaible(self, lstc):
        for stc in lstc:
            fnwp_to_check = os.path.join(self.des_dir, "loss_stc_{0}.csv".format(stc))
            if not os.path.exists(fnwp_to_check):
                return False
        return True
    def get_loss_from_tb(self):
        lfn=[fn for fn in os.listdir(self.lgc.tensorboard_dir) if fn.startswith("events")]
        if len(lfn)!=1:
            return False
        else:
            fnwp_tblog=os.path.join(self.lgc.tensorboard_dir,lfn[0])
        target_dir=self.des_dir
        #target_dir=os.path.join(self.lgc.system_working_dir,"analysis","pre_losses_from_tb")
        if not os.path.exists(target_dir): os.mkdir(target_dir)

        current_tc=0
        current_stc=0
        print("Handling saved tc ", current_stc)
        flag_one_record_ready=False
        Ditem={}
        df = pd.DataFrame(columns=["stc", "tc"] + self.lfield)
        fnwp_to_save = os.path.join(target_dir, "loss_stc_{0}.csv".format(current_stc))
        for event in tf.train.summary_iterator(fnwp_tblog):
            if len(event.summary.value) == 0:
                continue
            working_tc=event.step
            working_stc=event.step/self.lgc.num_train_to_save_model*self.lgc.num_train_to_save_model
            if working_tc!=current_tc:
                flag_one_record_ready=True
                try:
                    litem = [current_stc, current_tc] + [Ditem[key] for key in self.lfield]
                except Exception as e:
                    print("ET {0} not have all loss data quit {1}".format(working_stc, Ditem))
                    break
                current_tc=working_tc
                Ditem = {}
                for item in event.summary.value:
                    Ditem[item.tag] = item.simple_value
            else:
                for item in event.summary.value:
                    Ditem[item.tag] = item.simple_value
                continue

            if flag_one_record_ready:
                df.loc[len(df)]=litem
                flag_one_record_ready=False
            if working_stc!=current_stc:
                df.to_csv(fnwp_to_save, index=False)
                current_stc= working_stc
                print("Handling saved tc ", current_stc)
                df = pd.DataFrame(columns=["stc", "tc"] + self.lfield)
                fnwp_to_save = os.path.join(target_dir, "loss_stc_{0}.csv".format(current_stc))



class ana_loss:
    def __init__(self,system_name):
        self.system_name=system_name
        param_fnwp = os.path.join(sc.base_dir_RL_system, self.system_name, "config.json")
        if not os.path.exists(param_fnwp):
            raise ValueError("{0} does not exisit".format(param_fnwp))
        self.lgc=sc.gconfig()
        self.lgc.read_from_json(param_fnwp)


        self.i_loss_summary_recorder=loss_summary_recorder(self.system_name)
        self.i_loss_summary_tb=ana_loss_data_tb(self.system_name)

    def show_loss(self,fig, ET,LEvalT, l_content,i_get_data, fun_plot_reward_count):
        allaxes = fig.get_axes()
        for axe in allaxes:
            axe.remove()
        fig.add_subplot(221)
        fig.add_subplot(222)
        fig.add_subplot(223)
        fig.add_subplot(224)

        allaxes = fig.get_axes()
        fig.suptitle("Summary on/at {0} EvalT{1}".format(self.system_name,ET), fontsize=14)
        fig.subplots_adjust(bottom=0.05, top=0.9, left=0.03, right=0.97, wspace=0.1, hspace=0.3)
        ax=allaxes[0]
        fun_plot_reward_count(ax,ET)
        ymin, ymax = ax.get_ylim()

        '''
        stc = ET / i_get_data.lgc.num_train_to_save_model - 1
        ax.plot([stc, stc], [ymin, ymax])

        stc=ET-self.lgc.num_train_to_save_model
        df = i_get_data.get_losses_per_stc(stc)
        '''
        stc = ET / i_get_data.lgc.num_train_to_save_model
        ax.plot([stc, stc], [ymin, ymax])
        df = i_get_data.get_losses_per_stc(ET)


        if len(df)==0:
            return
        df.reset_index(inplace=True)
        if len(l_content)==2 and i_get_data == self.i_loss_summary_recorder:
            l_content.append("valid_count")
        for idx, content in enumerate(l_content):
            if i_get_data == self.i_loss_summary_tb and content == "M_reward":
                continue
            ax=allaxes[1+idx]
            ax.set_title(content)
            ax.plot(df.index,df[content],label=content)
            ax.set_title("{0} @ ET {1}".format(content, ET))
            ax.legend(loc='upper right')
            ax.tick_params(axis='x', rotation=90)
            ax.set_xticks(list(range(len(df.index) + 1)))
            ax.set_xticklabels(df.index.values, fontsize=7)

    def show_loss_summary(self, fig, l_content, l_sfun,ET,fun_plot_reward_count,LEvalT,i_get_data):
        assert len(l_content)<=3
        x_tick_label=[0]+LEvalT
        df = i_get_data.get_loss_summary()
        if len(df)==0:
            return

        allaxes = fig.get_axes()
        for axe in allaxes:
            axe.remove()
        fig.add_subplot(221)
        fig.add_subplot(222)
        fig.add_subplot(223)
        fig.add_subplot(224)

        allaxes = fig.get_axes()

        ax=allaxes[0]
        fun_plot_reward_count(ax,ET)
        ymin, ymax = ax.get_ylim()
        #stc = ET / i_get_data.lgc.num_train_to_save_model - 1
        stc = ET / i_get_data.lgc.num_train_to_save_model
        ax.plot([stc, stc], [ymin, ymax])

        for idx,content in enumerate(l_content):
            if i_get_data==self.i_loss_summary_tb and content=="M_reward":
                continue
            for sf in l_sfun:
                ax=allaxes[1+idx]
                cname="{0}__{1}".format(content,sf)
                ax.plot(df[cname],label=cname)
                ymin,ymax=ax.get_ylim()
                stc = ET / i_get_data.lgc.num_train_to_save_model
                ax.plot([stc,stc],[ymin,ymax])
            ax.set_title("{0}".format(content))
            ax.legend(loc='upper right')
            ax.tick_params(axis='x', rotation=90)
            ax.set_xticks(list(range(len(x_tick_label) + 1)))
            ax.set_xticklabels(x_tick_label, fontsize=7)

