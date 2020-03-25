import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from vcomm import Checkbar,label_entry,img_tool,one_label_entry,one_radion_box,param_input
from vresult_data_reward import ana_reward
from vresult_data_loss import ana_loss
from vresult_data_pbsv import ana_pbsv, ana_pbsv_detail
import tkinter as tk
import recorder

class vresult(tk.Frame):
    def __init__(self, container, param):
        self.system_name=param["system_name"]
        self.eval_process_name=param["eval_process_name"]
        self.Lstock=param["Lstock"]
        self.LEvalT=param["LEvalT"]
        self.LYM=param["LYM"]
        self.lgc=param["lgc"]

        tk.Frame.__init__(self, container)
        self.Cidx_stock,self.Cidx_EvalT,self.Cidx_LYM=0,0,0
        self.Cidx_stock2, self.Cidx_EvalT2, self.Cidx_LYM2 = 0, 0, 0

        self.i_ana_reward           = ana_reward(self.system_name, self.eval_process_name,self.Lstock, self.LEvalT, self.LYM,self.lgc)

        self.i_ana_loss= ana_loss(self.system_name)


        self.lfield, self.lstat=self.i_ana_loss.i_loss_summary_recorder.get_column_choices()
        self.i_ana_pbsv          =   ana_pbsv(self.system_name, self.eval_process_name,self.Lstock, self.LEvalT, self.LYM,self.lgc)
        self.i_ana_pbsv_detail     =   ana_pbsv_detail(self.system_name, self.eval_process_name,self.Lstock, self.LEvalT, self.LYM,self.lgc)

        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2TkAgg(self.canvas, self)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.frame_update("")
        self.init_control_frame()


    def frame_update(self, Event):
        if Event=="":  #this only for the init start
            show_type = "reward_summary"
            EvalT=1000
            stock2, EvalT2, YM2="","",""
            Cidx_stock2,Cidx_LYM2,Cidx_EvalT2=0,0,0
            rd_param = []
            hist_param = []

            ETP_param=[]
            stock=""
            YM=""
            selected_sv_pb=[]
            reward_dist_sub_choice=""
            Dreward_img_customized=[]

            l_selected_loss_summary =""
            l_selected_loss_summary_fun =""

        else:
            #prilimary SEY
            stock= self.input_stock.entry()
            self.Cidx_stock=self.Lstock.index(stock)

            YM=self.input_month.entry()
            self.Cidx_LYM = self.LYM.index(YM)

            EvalT=int(self.input_eval_count.entry())
            self.Cidx_EvalT=self.LEvalT.index(EvalT)

            # secondary SEY
            stock2 = self.input_stock2.entry()
            self.Cidx_stock2 = self.Lstock.index(stock2)

            YM2 = self.input_month2.entry()
            self.Cidx_LYM2 = self.LYM.index(YM2)

            EvalT2 = int(self.input_eval_count2.entry())
            self.Cidx_EvalT2 = self.LEvalT.index(EvalT2)

            show_type=self.show_type.get()#self.cb.selcted_item()

            selected_sv_pb=self.sv_pb.selcted_item()

            rd_param=self.i_rd_param.get_param()

            hist_param=self.i_hist_param.get_param()

            ETP_param=self.i_ETP_param.get_param()

            reward_dist_sub_choice=self.i_reward_dist_sub_choice.get()

            Dreward_img_customized=self.i_reward_img_customize.get_param()


            l_selected_loss_summary=self.loss_content.selcted_item()
            if len(l_selected_loss_summary)>3:
                return
            l_selected_loss_summary_fun=self.loss_fun.selcted_item()
            if len(l_selected_loss_summary_fun)<1:
                return
        if show_type=="reward_summary":
            self.i_ana_reward.show_reward(self.fig,EvalT,rd_param=rd_param, hist_param=hist_param)
        elif show_type=="reward_detail":
            self.i_ana_reward.show_reward_detail(self.fig, stock, EvalT, YM,Dimgc=Dreward_img_customized)
        elif show_type=="reward_distribution":
            self.i_ana_reward.show_reward_distribution(self.fig, stock, EvalT, Dimgc=Dreward_img_customized,
                                                       sub_choice=reward_dist_sub_choice)
        elif show_type == "reward_compare_ET":
            lpriliminary_choice= [stock, EvalT,YM]
            lsecondary_choice =  [stock2, EvalT2,YM2]
            self.i_ana_reward.show_reward_compare_ET(self.fig,lpriliminary_choice, lsecondary_choice,hist_param,
                                                     Dimgc=Dreward_img_customized,sub_choice=reward_dist_sub_choice)
        elif show_type == "reward_compare_ET(stock)":
            lpriliminary_choice= [stock, EvalT,YM]
            lsecondary_choice =  [stock2, EvalT2,YM2]
            if stock!=stock2:
                return
            self.i_ana_reward.show_reward_compare_ET__stock(self.fig, lpriliminary_choice, lsecondary_choice,
                                            hist_param, Dimgc=Dreward_img_customized,sub_choice=reward_dist_sub_choice)
        elif show_type=="pbsv_summary":
            if len(selected_sv_pb) <2:
                return
            self.i_ana_pbsv.show(self.fig, stock,selected_sv_pb, ETP_param=ETP_param, flag_trend=False)
        elif show_type=="pbsv_summary_trend":
            if len(selected_sv_pb) <2:
                return
            self.i_ana_pbsv.show(self.fig, stock,selected_sv_pb, ETP_param=ETP_param, flag_trend=True)
        elif show_type == "pbsv_detail":
            if "state_value" in selected_sv_pb:
                selected_sv_pb.remove("state_value")
            if len(selected_sv_pb) ==0:
                return
            self.i_ana_pbsv_detail.show(self.fig, stock, EvalT,selected_sv_pb)
        elif show_type == "loss_detail_ET":
            self.i_ana_loss.show_loss(self.fig,EvalT,self.LEvalT,l_selected_loss_summary,
                                      self.i_ana_loss.i_loss_summary_recorder,self.i_ana_reward.plot_reward_count)
        elif show_type == "loss_summary":
            self.i_ana_loss.show_loss_summary(self.fig,l_selected_loss_summary, l_selected_loss_summary_fun,EvalT,
                                self.i_ana_reward.plot_reward_count, self.LEvalT,self.i_ana_loss.i_loss_summary_recorder)

        elif show_type == "loss_detail_ET_tb(quick)":
            self.i_ana_loss.show_loss(self.fig,EvalT,self.LEvalT,l_selected_loss_summary,
                                      self.i_ana_loss.i_loss_summary_tb,self.i_ana_reward.plot_reward_count)
        elif show_type == "loss_summary_tb(quick)":
            self.i_ana_loss.show_loss_summary(self.fig,l_selected_loss_summary, l_selected_loss_summary_fun,EvalT,
                                  self.i_ana_reward.plot_reward_count, self.LEvalT, self.i_ana_loss.i_loss_summary_tb)

        else:
            assert False, "unexpect show_type choice {0}".format(show_type)

        self.canvas.draw()



    def init_control_frame(self):
        layout_column=0
        xiamian=tk.Frame(self)
        xiamian.pack(anchor=tk.W)

        cfc0 = tk.Frame(xiamian,highlightbackground="green", highlightcolor="green", highlightthickness=2,
                              width=200, height=200, bd= 0)
        cfc0.grid(column=layout_column,row=0,sticky=tk.NW)


        tk.Label(cfc0, text = "Pliminary Choice").grid(column=0, row=0, sticky=tk.W)
        cfc0r1= tk.Frame(cfc0)
        cfc0r1.grid(column=0, row=1, sticky=tk.W)
        self.input_stock = label_entry(cfc0r1, "Stock", self.Lstock[self.Cidx_stock] )

        cfc0r2= tk.Frame(cfc0)
        cfc0r2.grid(column=0, row=2, sticky=tk.W)
        self.input_month = label_entry(cfc0r2, "Month", self.LYM[self.Cidx_LYM])

        cfc0r3= tk.Frame(cfc0)
        cfc0r3.grid(column=0, row=3, sticky=tk.W)
        self.input_eval_count = label_entry(cfc0r3, "EvalT", self.LEvalT[self.Cidx_EvalT])
        layout_column+=1

        cfc0 = tk.Frame(xiamian,highlightbackground="green", highlightcolor="green", highlightthickness=2,
                              width=200, height=200, bd= 0)
        cfc0.grid(column=layout_column,row=0,sticky=tk.NW)
        tk.Label(cfc0, text="Secondary Choice").grid(column=0, row=0, sticky=tk.W)

        cfc0r1= tk.Frame(cfc0)
        cfc0r1.grid(column=0, row=1, sticky=tk.W)
        self.input_stock2 = label_entry(cfc0r1, "Stock", self.Lstock[self.Cidx_stock2] )

        cfc0r2= tk.Frame(cfc0)
        cfc0r2.grid(column=0, row=2, sticky=tk.W)
        self.input_month2 = label_entry(cfc0r2, "Month", self.LYM[self.Cidx_LYM2])

        cfc0r3= tk.Frame(cfc0)
        cfc0r3.grid(column=0, row=3, sticky=tk.W)
        self.input_eval_count2 = label_entry(cfc0r3, "EvalT", self.LEvalT[self.Cidx_EvalT2])
        layout_column+=1


        # radio box show type
        title_show_type = [
            ("reward_summary",                  "reward_summary"),
            ("reward_detail",                   "reward_detail"),
            ("reward_distribution",             "reward_distribution"),
            ("reward_compare_ET",                "reward_compare_ET"),
            ("reward_compare_ET(stock)",          "reward_compare_ET(stock)"),
            ("pbsv_summary",                    "pbsv_summary"),
            ("pbsv_summary_trend",              "pbsv_summary_trend"),
            ("pbsv_detail",                     "pbsv_detail"),
            ("loss_detail_ET",                  "loss_detail_ET"),
            ("loss_summary",                    "loss_summary"),
            ("loss_detail_ET_tb(quick)",        "loss_detail_ET_tb(quick)"),
            ("loss_summary_tb(quick)",           "loss_summary_tb(quick)")
        ]
        cfc1 = tk.Frame(xiamian,highlightbackground="green", highlightcolor="green", highlightthickness=2,
                              width=200, height=200, bd= 0)
        cfc1.grid(column=layout_column,row=0,sticky=tk.NW)
        self.show_type=one_radion_box(cfc1,title_show_type, main_title="main menu")
        layout_column+=1

        # check box sv ap choice
        title_sv_pb = ["state_value"]
        for idx in range(self.lgc.train_num_action):
            label="Action_Prob_{0}".format(idx)
            title_sv_pb.append(label)
        cf = tk.Frame(xiamian,highlightbackground="green", highlightcolor="green", highlightthickness=2,
                              width=200, height=200, bd= 0)
        cf.grid(column=layout_column,row=0,sticky=tk.NW)
        self.sv_pb=Checkbar(cf, title_sv_pb,[title_sv_pb[0],title_sv_pb[1]],flag_divide=False,main_title="choice_pbsv" )
        layout_column += 1

        cf = tk.Frame(xiamian,highlightbackground="green", highlightcolor="green", highlightthickness=2,
                              width=200, height=200, bd= 0)
        cf.grid(column=layout_column,row=0,sticky=tk.NW)
        title_reward_duration = [
            ("rd(origin)",                "reward vs duration(origin)"),
            ("rd(threadhold)",             "reward vs duration(threadhold)")
        ]
        self.i_rd_param=param_input(cf,title_reward_duration,["reward_max","reward_min","duration_max"],main_title="reward_summary_rd" )
        self.i_rd_param.show_elements()
        layout_column += 1

        title_hist = [
            ("hist(origin)",                "hist(origin)"),
            ("hist(threadhold)",             "hist(threadhold)")
        ]
        cf = tk.Frame(xiamian,highlightbackground="green", highlightcolor="green", highlightthickness=2,
                              width=200, height=200, bd= 0)
        cf.grid(column=layout_column,row=0,sticky=tk.NW)
        self.i_hist_param=param_input(cf,title_hist,["hist_max","hist_min","hist_step"],main_title="reward_summary_hist" )
        self.i_hist_param.show_elements()
        layout_column += 1

        title_EvalT_period = [
            ("ET_period(origin)",                "ET_period(origin)"),
            ("ET_period(threadhold)",             "ET_period(threadhold)")
        ]
        cf = tk.Frame(xiamian,highlightbackground="green", highlightcolor="green", highlightthickness=2,
                              width=200, height=200, bd= 0)
        cf.grid(column=layout_column,row=0,sticky=tk.NW)
        self.i_ETP_param=param_input(cf,title_EvalT_period,["ET_max","ET_min"],main_title="pbsv_clip" )
        self.i_ETP_param.show_elements()
        layout_column += 1

        # radio box show type
        title_sub_choice = [
            ("mean",    "mean"),
            ("median",  "median"),
            ("count",   "count"),
            ("std",     "std")
        ]
        cf = tk.Frame(xiamian,highlightbackground="green", highlightcolor="green", highlightthickness=2,
                              width=200, height=200, bd= 0)
        cf.grid(column=layout_column,row=0,sticky=tk.NW)
        self.i_reward_dist_sub_choice=one_radion_box(cf,title_sub_choice, main_title="reward_dist_sub_choice")
        layout_column+=1


        title_hist = [
            ("Origin",             "Origin"),
            ("Greater E",          "Greater"),
            ("Less E",             "Less")
        ]
        cf = tk.Frame(xiamian,highlightbackground="green", highlightcolor="green", highlightthickness=2,
                              width=200, height=200, bd= 0)
        cf.grid(column=layout_column,row=0,sticky=tk.NW)
        self.i_reward_img_customize=param_input(cf,title_hist,["threadhold"],main_title="custom_reward_img" )
        self.i_reward_img_customize.show_elements()
        layout_column += 1


        # check box sv ap choice
        title_loss = ["loss_summary_choice"]
        cf = tk.Frame(xiamian,highlightbackground="green", highlightcolor="green", highlightthickness=2,
                              width=200, height=200, bd= 0)
        cf.grid(column=layout_column,row=0,sticky=tk.NW)


        cfsub = tk.Frame(cf, width=200, height=200, bd= 0)
        cfsub.grid(column=0,row=0,sticky=tk.NW)
        self.loss_content = Checkbar(cfsub, self.lfield,[self.lfield[0],self.lfield[1],self.lfield[2]], flag_divide=False,
                              main_title="choice_content")

        cfsub = tk.Frame(cf,width=200, height=200, bd= 0)
        cfsub.grid(column=0,row=1,sticky=tk.NW)
        self.loss_fun = Checkbar(cfsub, self.lstat,[self.lstat[0],self.lstat[2],self.lstat[3]] ,flag_divide=True,
                              main_title="choice_fun")

        layout_column += 1

        # update buttom
        cf = tk.Frame(xiamian,highlightbackground="green", highlightcolor="green", highlightthickness=2,
                              width=200, height=200, bd= 0)
        cf.grid(column=layout_column,row=0,sticky=tk.NW)
        self.update_button = tk.Button(cf, text="update",width=10)
        self.update_button.pack(anchor=tk.W)
        self.prepare_loss_button = tk.Button(cf, text="prepare_loss",width=10)
        self.prepare_loss_button.pack(anchor=tk.W)

        layout_column += 1




        #prilimary SEY
        self.stock_pre_button=self.pre_button_common(self.Lstock, "Cidx_stock",self.input_stock)
        self.stock_next_button=self.next_button_common(self.Lstock, "Cidx_stock",self.input_stock)
        self.input_stock.get_prev_button().bind("<Button-1>", func=self.stock_pre_button)
        self.input_stock.get_next_button().bind("<Button-1>", func=self.stock_next_button)


        self.month_pre_button=self.pre_button_common(self.LYM, "Cidx_LYM",self.input_month)
        self.month_next_button = self.next_button_common(self.LYM, "Cidx_LYM", self.input_month)
        self.input_month.get_prev_button().bind("<Button-1>", func=self.month_pre_button)
        self.input_month.get_next_button().bind("<Button-1>", func=self.month_next_button)

        self.eval_count_pre_button=self.pre_button_common(self.LEvalT, "Cidx_EvalT",self.input_eval_count)
        self.eval_count_next_button = self.next_button_common(self.LEvalT, "Cidx_EvalT", self.input_eval_count)
        self.input_eval_count.get_prev_button().bind("<Button-1>", func=self.eval_count_pre_button)
        self.input_eval_count.get_next_button().bind("<Button-1>", func=self.eval_count_next_button)

        # secondary SEY
        self.stock_pre_button2=self.pre_button_common(self.Lstock, "Cidx_stock2",self.input_stock2)
        self.stock_next_button2=self.next_button_common(self.Lstock, "Cidx_stock2",self.input_stock2)
        self.input_stock2.get_prev_button().bind("<Button-1>", func=self.stock_pre_button2)
        self.input_stock2.get_next_button().bind("<Button-1>", func=self.stock_next_button2)


        self.month_pre_button2=self.pre_button_common(self.LYM, "Cidx_LYM2",self.input_month2)
        self.month_next_button2 = self.next_button_common(self.LYM, "Cidx_LYM2", self.input_month2)
        self.input_month2.get_prev_button().bind("<Button-1>", func=self.month_pre_button2)
        self.input_month2.get_next_button().bind("<Button-1>", func=self.month_next_button2)

        self.eval_count_pre_button2=self.pre_button_common(self.LEvalT, "Cidx_EvalT2",self.input_eval_count2)
        self.eval_count_next_button2 = self.next_button_common(self.LEvalT, "Cidx_EvalT2", self.input_eval_count2)
        self.input_eval_count2.get_prev_button().bind("<Button-1>", func=self.eval_count_pre_button2)
        self.input_eval_count2.get_next_button().bind("<Button-1>", func=self.eval_count_next_button2)

        self.update_button.bind("<Button-1>", func=self.frame_update)

        self.prepare_loss_button.bind("<Button-1>", func=self.prepare_loss)


    def prepare_loss(self,Event):
        i=recorder.get_recorder_OS_losses(self.system_name)
        i.get_losses()

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
