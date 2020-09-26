import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk #NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import tkinter as tk
import re,os
import vresult_data_reward
import config as sc
#from data_common import API_HFQ_from_file,hfq_toolbox
from DBI_Base import DBI_init
from vcomm import Checkbar,label_entry,img_tool,one_label_entry,one_radion_box,param_input
class vEreward(tk.Frame):
    def __init__(self, container, param):
        tk.Frame.__init__(self, container)
        self.system_name=param["system_name"]
        self.eval_process_name=param["eval_process_name"]
        self.threadhold=-20
        self.compare_opration="less"
        self.Lstock=param["Lstock"]
        self.LEvalT=param["LEvalT"]
        self.LYM=param["LYM"]
        self.lgc=param["lgc"]

        self.i_anaExtreR=anaExtreR(self.system_name, self.eval_process_name, self.threadhold,self.compare_opration,self.Lstock, self.LEvalT, self.LYM,self.lgc)

        self.lcc, self.lncc=self.i_anaExtreR.lcc, self.i_anaExtreR.lncc

        self.Cidx_lcc, self.Cidx_lncc=0,0

        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        #self.canvas.show()
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        #toolbar = NavigationToolbar2TkAgg(self.canvas, self)
        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.frame_update("")
        self.init_control_frame()


    def frame_update(self, Event):
        if Event=="":  #this only for the init start
            #init fake data
            cc_tid=self.lcc[self.Cidx_lcc]
            ncc_tid=self.lncc[self.Cidx_lncc]
        else:
            cc_tid = self.input_cc.entry()
            self.Cidx_lcc = self.lcc.index(cc_tid)

            ncc_tid = self.input_ncc.entry()
            self.Cidx_lncc = self.lncc.index(ncc_tid)

        self.i_anaExtreR.show(self.fig,cc_tid,ncc_tid)
        #call show fun
        self.canvas.draw()



    def init_control_frame(self):
        layout_column=0
        xiamian=tk.Frame(self)
        xiamian.pack(anchor=tk.W)

        cfc0 = tk.Frame(xiamian,highlightbackground="green", highlightcolor="green", highlightthickness=2,
                              width=300, height=200, bd= 0)
        cfc0.grid(column=layout_column,row=0,sticky=tk.NW)

        #add control

        tk.Label(cfc0, text = "Extreme Reward {0} {1}".format(self.compare_opration,self.threadhold)).grid(column=0, row=0, sticky=tk.W)
        cfc0r1= tk.Frame(cfc0)
        cfc0r1.grid(column=0, row=1, sticky=tk.W)
        self.input_cc = label_entry(cfc0r1, "ratio changed", self.lcc[self.Cidx_lcc] )

        cfc0r2= tk.Frame(cfc0)
        cfc0r2.grid(column=0, row=2, sticky=tk.W)
        self.input_ncc = label_entry(cfc0r2, "ratio unchanged", self.lncc[self.Cidx_lncc])

        layout_column += 1
        # update buttom
        cf = tk.Frame(xiamian,highlightbackground="green", highlightcolor="green", highlightthickness=2,
                              width=200, height=200, bd= 0)
        cf.grid(column=layout_column,row=0,sticky=tk.NW)
        self.update_button = tk.Button(cf, text="update",width=10)
        self.update_button.pack(anchor=tk.W)
        layout_column += 1



        # LINK BUTTOM TO FUN
        #prilimary SEY
        self.stock_pre_button=self.pre_button_common(self.lcc, "Cidx_lcc",self.input_cc)
        self.stock_next_button=self.next_button_common(self.lcc, "Cidx_lcc",self.input_cc)
        self.input_cc.get_prev_button().bind("<Button-1>", func=self.stock_pre_button)
        self.input_cc.get_next_button().bind("<Button-1>", func=self.stock_next_button)


        self.month_pre_button=self.pre_button_common(self.lncc, "Cidx_lncc",self.input_ncc)
        self.month_next_button = self.next_button_common(self.lncc, "Cidx_lncc", self.input_ncc)
        self.input_ncc.get_prev_button().bind("<Button-1>", func=self.month_pre_button)
        self.input_ncc.get_next_button().bind("<Button-1>", func=self.month_next_button)



    def pre_button_common(self, List_content, list_index_name,control):
        def pre_button_core(Event):
            Cidx=getattr(self,list_index_name)
            if Cidx == 0:
                Cidx = len(List_content) - 1
            else:
                Cidx -= 1
            selected_content = List_content[Cidx]
            control.set_entry(selected_content)
            self.frame_update(Event)
        return pre_button_core

    def next_button_common(self,List_content, list_index_name,control):
        def next_button_core(Event):
            Cidx = getattr(self, list_index_name)
            if Cidx == len(List_content) - 1:
                Cidx = 0
            else:
                Cidx += 1
            selected_content = List_content[Cidx]
            control.set_entry(selected_content)
            self.frame_update(Event)
        return next_button_core


class anaExtreR(DBI_init):
    def __init__(self,system_name, process_name, threadhold,compare_operation,Lstock, LEvalT, LYM,lgc):
        DBI_init.__init__(self)
        #system_name="LHPP2V2_PPO3_LOSSV05P3R10_SV_Deep_SG_round_6_reward_same_2_Eval3_3"
        #process_name="Eval_0"
        #threadhold=-20
        #compare_operation="less"  #["less", "greater"]

        self.system_name, self.process_name, self.threadhold, self.compare_operation=\
            system_name, process_name, threadhold,compare_operation

        self.i_are_summary = vresult_data_reward.ana_reward_data(self.system_name, self.process_name,Lstock, LEvalT, LYM,lgc)
        self.dfare,_,_ = self.i_are_summary.get_are_summary()

        self.lcc, lncc=self.get_cc_ncc_tid_list(self.threadhold, self.compare_operation)

        lncc_hle, self.lncc =self.get_hle(lncc)  # remove the hightest  equal to lowest days, which seem abnormal trading day

        if len(self.lcc)==0:
            self.lcc.append("Empty")
        if len(self.lncc)==0:
            self.lncc.append("Empty")

    def get_cc_ncc_tid_list(self,threadhold,compare_operation):
        if compare_operation=="less":
            df=self.dfare[self.dfare["reward"]<threadhold]
        else:
            assert compare_operation=="greater"
            df = self.dfare[self.dfare["reward"] > threadhold]

        df.sort_values(by=['stock'])
        lstock=list(set(df["stock"].tolist()))
        l_tid_cc=[]
        l_tid_ncc=[]
        for stock in lstock:
            print("handling ", stock)
            #dfh = API_HFQ_from_file().get_df_HFQ(stock)
            _,dfh = self.get_hfq_df(self.get_DBI_hfq_fnwp(stock))
            dfr = df[df["stock"]==stock]
            for idx, row in dfr.iterrows():
                dfhr=dfh[(dfh["date"]>=str(row["trans_start"])) & (dfh["date"]<=str(row["trans_end"]))]
                if len(list(set(dfhr["coefficient_fq"].tolist())))>1:
                    l_tid_cc.append(row["trans_id"])
                else:
                    l_tid_ncc.append(row["trans_id"])
        return l_tid_cc, l_tid_ncc

    def get_hle(self,l_tid_ncc):
        current_stock=""
        dfh=pd.DataFrame([])
        lncc_hle=[]
        lncc_nhle = []
        for tid in l_tid_ncc:
            working_stock=re.findall(r"(\w+)_\w+", tid)[0]
            if working_stock!=current_stock:
                #dfh = API_HFQ_from_file().get_df_HFQ(working_stock)
                _,dfh = self.get_hfq_df(self.get_DBI_hfq_fnwp(working_stock))
                dfh["Open_Nprice"] = dfh["open_price"] / dfh["coefficient_fq"]
                dfh["highest_Nprice"] = dfh["highest_price"] / dfh["coefficient_fq"]
                dfh["lowest_Nprice"] = dfh["lowest_price"] / dfh["coefficient_fq"]
                dfh["close_Nprice"] = dfh["close_price"] / dfh["coefficient_fq"]
                current_stock=working_stock
            df=self.dfare[self.dfare["trans_id"]==tid]
            assert len(df)==1
            dfhr=dfh[(dfh["date"]>=str(df.iloc[0]["trans_start"]))&(dfh["date"]<=str(df.iloc[0]["trans_end"]))]
            flag_hle=False
            for idx, row in dfhr.iterrows():
                if row["lowest_price"]==row["highest_price"]:
                    flag_hle=True
                    break
            if flag_hle:
                lncc_hle.append(tid)
            else:
                lncc_nhle.append(tid)
        return lncc_hle, lncc_nhle

    def CmdTool_check_Hratio_change(self,stock):
        #dfh = API_HFQ_from_file().get_df_HFQ(stock)
        _,dfh = self.get_hfq_df(self.get_DBI_hfq_fnwp(stock))
        dfh["Scoefficient_fq"] = dfh["coefficient_fq"].shift(1)
        dfh.bfill(inplace=True)
        dfh["change_flag_Hratio"] = dfh["Scoefficient_fq"] - dfh["coefficient_fq"]
        return dfh[dfh["change_flag_Hratio"] != 0]

    def CmdTool_check_date_in_are(self,system_name, process_name, stock, date_s):
        src_dir = os.path.join(sc.base_dir_RL_system, system_name, process_name, stock)
        lfn = [fn for fn in os.listdir(src_dir) if "log_a_r_e" in fn]
        for fn in lfn:
            fnwp = os.path.join(src_dir, fn)
            df = pd.read_csv(fnwp, header=0)
            df["day"] = df["day"].astype(str)
            if len(df[df["day"] == date_s]) > 0:
                print("Found ", fnwp, df[df["day"] == date_s])
                break
        else:
            print("Not Found ")


    def CmdTool_print_ltid_start_end_date(self,l_tid):
        for tid in l_tid:
            df=self.dfare[self.dfare["trans_id"]==tid]
            assert len(df)==1
            if ("201507" in str(df["trans_start"].iloc[0])) or ("201507" in str(df["trans_end"].iloc[0])):
                pass
            else:
                print("{0}  {1}  {2}".format(df["trans_id"].iloc[0], df["trans_start"].iloc[0] , df["trans_end"].iloc[0]))

    def debug_show(self):
        fig, ax_array = plt.subplots(2, 1)
        self.plot_td_data(ax_array[0], "SH600218_T6339")
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        mng.resize(*mng.window.maxsize())
        fig.canvas.draw()

    def show(self, fig, cc_tid, ncc_tid):
        allaxes = fig.get_axes()
        for axe in allaxes:
            axe.remove()
        fig.add_subplot(211)
        fig.add_subplot(212)
        allaxes = fig.get_axes()


        if cc_tid!="Empty":
            ax = allaxes[0]
            self.plot_td_data(ax, cc_tid)
            ax.set_title("ratio changed total {0} Tid {1}".format(len(self.lcc),cc_tid))
        if ncc_tid!="Empty":
            ax = allaxes[1]
            self.plot_td_data(ax, ncc_tid)
            ax.set_title("ratio un changed total {0} Tid {1}".format(len(self.lncc),ncc_tid))

    def plot_td_data(self, ax,tid):
        df=self.dfare[self.dfare["trans_id"]==tid]
        assert len(df)==1
        stock=re.findall(r'(\w+)_\w+', tid)[0]
        #dfh = API_HFQ_from_file().get_df_HFQ(stock)
        _, dfh = self.get_hfq_df(self.get_DBI_hfq_fnwp(stock))
        dfh["Open_Nprice"]=dfh["open_price"]/dfh["coefficient_fq"]
        dfh["highest_Nprice"] = dfh["highest_price"] / dfh["coefficient_fq"]
        dfh["lowest_Nprice"] = dfh["lowest_price"] / dfh["coefficient_fq"]
        dfh["close_Nprice"] = dfh["close_price"] / dfh["coefficient_fq"]

        dfhr=dfh[(dfh["date"]>=str(df.iloc[0]["trans_start"]))&(dfh["date"]<=str(df.iloc[0]["trans_end"]))]

        ax.plot(list(range(len(dfhr))),dfhr["highest_Nprice"].values, color="b", label="highest_Nprice")
        ax.plot(list(range(len(dfhr))),dfhr["lowest_Nprice"].values, color="g", label="lowest_Nprice")

        ax.plot([0,len(dfhr)-1],[df.iloc[0]["buy_price"],df.iloc[0]["sell_price"]], color="r",label="transaction_price")
        ax2=ax.twinx()
        ax2.plot(list(range(len(dfhr))), dfhr["coefficient_fq"].values, color="k", label="coefficient_fq")
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        #ax.tick_params(axis='x', rotation=90)
        ax.set_xticks(list(range(len(dfhr))))
        ax.set_xticklabels(dfhr["date"].tolist(), fontsize=7)
