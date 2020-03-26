import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
from nets import Explore_Brain, init_gc
from vtrainer_comm import *
from vcomm import *
def init_global(system_name):
    global lgc
    param_fnwp = os.path.join(sc.base_dir_RL_system, system_name, "config.json")
    if not os.path.exists(param_fnwp):
        raise ValueError("{0} does not exisit".format(param_fnwp))
    lgc = sc.gconfig()
    lgc.read_from_json(param_fnwp)
    init_gc(lgc)
    comm_init_lgc(lgc)

class vtrainer_tk(tk.Frame):
    def __init__(self, parent, param):
        tk.Frame.__init__(self, parent)
        self.system_name  =   param["system_name"]
        init_global(self.system_name)
        self.fig = Figure(figsize=(5, 5), dpi=100)

        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2TkAgg(self.canvas, self)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.L_DModel=[{"Tmodel":None, "Pmodel": None, "train_count":None,"ib":visual_trainer_brain()}  for _ in range(2)]
        self.L_ln, _ = self.L_DModel[0]["ib"].get_trainable_layer_list_from_config_file(self.system_name)
        brain_model_dir=os.path.join(sc.base_dir_RL_system,self.system_name,"model")
        self.L_tcs=[re.findall(r'\w+T(\d+).h5py',fn)[0] for fn in os.listdir(brain_model_dir) if fn.startswith("train_model_AIO_")]
        self.L_tcs.sort()
        self.L_control=[]
        self.L_table_control=[]
        self.train_data_surf_control=""

        self.init_control_frame()
        self.i_show_op=show_op()

        self.i_rd = record_variable2(lgc)

        self.td_current_sc = -1
        self.td_tc = -1
    def init_control_frame(self):
        xiamian=tk.Frame(self)
        xiamian.pack(anchor=tk.W)

        current_column=0
        container = tk.Frame(xiamian, highlightbackground="green", highlightcolor="green", highlightthickness=2,
                              width=200, height=200, bd= 0)
        container.grid(column=0, row=0, sticky=tk.NW)
        param={}
        param["title"]="base brain control"
        param["tcs"]=self.L_tcs
        param["layerNames"]=self.L_ln
        param["plot_type"]=[("2D img","2D img"),("hist","hist")]
        self.L_control.append(self._one_model_control_layout(container, "", param))

        current_column += 1
        container = tk.Frame(xiamian, highlightbackground="green", highlightcolor="green", highlightthickness=2,
                              width=200, height=200, bd= 0)
        container.grid(column=current_column, row=0, sticky=tk.NW)
        param["title"]="operation brain control"
        self.L_control.append(self._one_model_control_layout(container, "", param))

        current_column += 1
        container = tk.Frame(xiamian, highlightbackground="green", highlightcolor="green", highlightthickness=2,
                              width=200, height=200, bd= 0)
        container.grid(column=current_column, row=0, sticky=tk.NW)
        param["title"]="weight/bias difference"
        param.pop('tcs', None)
        self.L_control.append(self._layer_diff_control_layout(container, param))


        current_column += 1
        container = tk.Frame(xiamian, highlightbackground="green", highlightcolor="green", highlightthickness=2,
                              width=200, height=200, bd= 0)
        container.grid(column=current_column, row=0, sticky=tk.NW)
        sub_container = tk.Frame(container)
        sub_container.grid(column=0, row=0, sticky=tk.NW)
        self.L_table_control.append(self._tk_table_layout(sub_container, 6))
        param={}
        param["table_title"]="layer XX"
        param["column_titles"] = ["","min","25\%","50\%","75\%","max"]
        param["row_titles"] = ["weight", "bias"]
        param["data"]       =[[0,0,0,0,0],[0,0,0,0,0]]
        self._tk_table_update(self.L_table_control[0], param)
        sub_container = tk.Frame(container)
        sub_container.grid(column=0, row=1, sticky=tk.NW)
        self.L_table_control.append(self._tk_table_layout(sub_container, 6))
        self._tk_table_update(self.L_table_control[1], param)

        current_column +=1
        container = tk.Frame(xiamian, highlightbackground="green", highlightcolor="green", highlightthickness=2,
                              width=200, height=200, bd= 0)
        container.grid(column=current_column, row=0, sticky=tk.NW)
        param = {}
        param["title"]="Train data surf"
        param["l_sc"]=self.L_tcs
        param["max_step"]=lgc.num_train_to_save_model
        self.train_data_surf_control=self._tk_train_data_surf_layout(container,param )


        current_column += 1
        Btry = ttk.Button(xiamian, text="try")
        Btry.grid(column=current_column, row=0, sticky=tk.NW)
        Btry.bind("<Button-1>", func=self._try_show_data(self.fig, 0, 0))


        self._one_model_control_bind(self.L_control,self.L_ln, self.L_DModel, idx=0 )
        self._one_model_control_bind(self.L_control, self.L_ln, self.L_DModel, idx=1)
        self._layer_diff_control_bind(self.L_control, self.L_ln, self.L_DModel, idx=2)
        self._tk_train_data_surf_bind(self.train_data_surf_control)
    def _try_show_data(self, fig, sc, tc):
        def _show_data_base(event):
            self.show_train_datas(fig,sc, tc,"first half")
        return _show_data_base

    def show_layer(self,DModel,ln,show_type):
        assert type(DModel) is dict, "this is dirty solution, as shower_layer send one, show layer_diff send a list"
        if (DModel["Tmodel"] is None) or (DModel["Pmodel"] is None) or (DModel["saved_tc"] is None):
            return
        wbs=DModel["ib"].get_layer_wb(DModel["Pmodel"], ln)
        lresult = self.i_show_op._wb_stats(wbs)
        param = {}
        param["table_title"] = "layer: {0} @ TC {1}".format(ln, DModel["saved_tc"])
        param["column_titles"] = ["", "min", "25\%", "50\%", "75\%", "max"]
        param["row_titles"] = ["weight", "bias"]
        param["data"] = lresult
        self._tk_table_update(self.L_table_control[0], param)

        self.fig.suptitle('layer: {0} @ train count {1} weight/bias stastic'.format(ln, DModel["saved_tc"]))
        if show_type=="hist":

            allaxes = self.fig.get_axes()
            for axe in allaxes:
                axe.remove()
            self.fig.suptitle("layer: {0} @ TC {1}".format(ln, DModel["saved_tc"]))
            self.fig.add_subplot(221)
            self.fig.add_subplot(222)
            axes=self.fig.get_axes()
            self.i_show_op._wb_hist_op(wbs,axes,lresult)
        elif show_type=="2D img":
            self.fig.suptitle(
                'layer: {0} @ train count {1} weight/bias value'.format(ln, DModel["saved_tc"]))
            self.i_show_op._wbs_imshow(self.fig, wbs)
        else:
            return
        self.canvas.draw()
    def show_layer_diff(self,L_DModel,ln,show_type):
        assert len(L_DModel)==2, "this is dirty solution, as shower_layer send one, show layer_diff send a list"
        L_wb=[]
        for DModel in L_DModel:
            if (DModel["Tmodel"] is None) or (DModel["Pmodel"] is None) or (DModel["saved_tc"] is None):
                return
            L_wb.append(DModel["ib"].get_layer_wb(DModel["Pmodel"],ln))

        wbd=[]
        wbd.append(L_wb[0][0] - L_wb[1][0])
        wbd.append(L_wb[0][1] - L_wb[1][1])

        lresult = self.i_show_op._wb_stats(wbd)
        param = {}
        param["table_title"] = "layer: {0} @ TC {1} vs {2} weight/bias value".\
            format(ln, L_DModel[0]["saved_tc"],L_DModel[1]["saved_tc"])
        param["column_titles"] = ["", "min", "25\%", "50\%", "75\%", "max"]
        param["row_titles"] = ["weight", "bias"]
        param["data"] = lresult
        self._tk_table_update(self.L_table_control[1], param)

        if show_type=="hist":
            allaxes = self.fig.get_axes()
            for axe in allaxes:
                axe.remove()
            self.fig.suptitle("layer: {0} @ TC {1} vs {2} weight/bias stastic".
                    format(ln, L_DModel[0]["saved_tc"],L_DModel[1]["saved_tc"]))
            self.fig.add_subplot(221)
            self.fig.add_subplot(222)
            axes=self.fig.get_axes()
            self.i_show_op._wb_hist_op(wbd,axes,lresult)
        elif show_type=="2D img":
            self.fig.suptitle("layer: {0} @ TC {1} vs {2} weight/bias stastic".
                              format(ln, L_DModel[0]["saved_tc"],L_DModel[1]["saved_tc"]))
            self.i_show_op._wbs_imshow(self.fig, wbd)
        else:
            return
        self.canvas.draw()
    def show_train_datas(self, fig,saved_tc, tc, selected_half):

        if self.td_current_sc!=saved_tc or self.td_tc!=tc:
            RD_trainer, RD_brain, RD_process = self.i_rd.read_SC_CC_data(int(saved_tc), int(tc))
            self.td_current_sc = saved_tc
            self.td_tc = tc
            self.show_records = []
            for idx in range(len(RD_trainer[0])):
                rr = '%.2e' % float(RD_trainer[4][idx][0, 0])
                rs = RD_trainer[9][idx][0, 0]
                record = [rs["stock"], rs["date"], rs["action_taken"], rs["action_return_message"], rr]
                self.show_records.append(record)
            #self.show_records = [['%.2f' % float(j) for j in i] for i in self.show_records]
        base_point= 0 if selected_half=="first half" else 125
        allaxes = self.fig.get_axes()
        for axe in allaxes:
            axe.remove()
        fig.subplots_adjust(bottom=0.4, top=0.6, left=0.01, right=0.99, wspace=0.01, hspace=0.01)
        self.fig.suptitle("Saved Tc {0} TC {1} {2}".format(self.td_current_sc,self.td_tc, selected_half ))
        self.fig.add_subplot(151)
        self.fig.add_subplot(152)
        self.fig.add_subplot(153)
        self.fig.add_subplot(154)
        self.fig.add_subplot(155)
        axes=self.fig.get_axes()
        for i in range(0, 5):

            the_table = axes[i].table(
                cellText=self.show_records[base_point+i*25:base_point+(i+1)*25],
                #rowLabels=["stock", "data", "action","result","reward"],
                colLabels=["stock", "data", "action","result","reward"],
                loc='center',
            )
            axes[i].axis("off")
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(8)
            #the_table.scale(50, 5)

        self.canvas.draw()
    #operation fun
    def left_layer(self,dc, lc, DModel, idx):
        def left_layer_in(Event):
            ln = dc["layer_name"].get()
            cidx = lc.index(ln)
            cidx = cidx - 1 if cidx > 0 else len(lc) - 1
            ln = lc[cidx]
            dc["layer_name"].set(ln)
            show_type = dc["show_type"].get()
            if idx in [0,1]:
                self.show_layer(DModel, ln, show_type)
            else:
                assert idx ==2
                self.show_layer_diff(DModel, ln, show_type)
        return left_layer_in
    def right_layer(self,dc, lc, DModel, idx):
        def right_layer_in(Event):
            ln = dc["layer_name"].get()
            # print lc
            # print ln
            cidx = lc.index(ln)
            cidx = cidx + 1 if cidx < len(lc) - 1 else 0
            ln = lc[cidx]
            dc["layer_name"].set(ln)
            show_type = dc["show_type"].get()
            if idx in [0,1]:
                self.show_layer(DModel, ln, show_type)
            else:
                assert idx ==2
                self.show_layer_diff(DModel, ln, show_type)
        return right_layer_in
    def update_plot(self,dc, DModel, idx):
        def update_plot_in(Event):
            ln = dc["layer_name"].get()
            show_type = dc["show_type"].get()
            if idx in [0,1]:
                self.show_layer(DModel, ln, show_type)
            else:
                assert idx ==2
                self.show_layer_diff(DModel, ln, show_type)
        return update_plot_in
    def load_models(self,dc, Dmodel):
        model_config_fnwp = os.path.join(lgc.brain_model_dir, "config.json")
        def load_models_in(Event):
            tc = dc["tcs"].get()
            fns = [fn for fn in os.listdir(lgc.brain_model_dir) if
                   re.match(r'train_model_AIO_\w+T{0}.h5py'.format(tc), fn)]
            assert len(fns) == 1,"{0} train count {1} not have model saved ".format(self.system_name, tc)
            AIO_fnwp = os.path.join(lgc.brain_model_dir, fns[0])
            if AIO_fnwp!="":
                Dmodel["Tmodel"], Dmodel["Pmodel"]=Dmodel["ib"].load_model(AIO_fnwp,model_config_fnwp)
                Dmodel["saved_tc"]=tc
                dc["load_result"].set("Success load {0}".format(tc))
            else:
                Dmodel["Tmodel"], Dmodel["Pmodel"],Dmodel["saved_tc"] = None, None, None
                dc["load_result"].set("Fail load {0}".format(tc))
        return load_models_in
    def optimize_step(self,dc, Dmodel):
        def optimize_step_in(Event):
            i_rd = record_variable2(lgc)
            steps_to_take=int(dc["step"].get())
            if steps_to_take==0:
                dc["step_result"].set("0 step no optimize")
                return
            if (Dmodel["Tmodel"] is None) or (Dmodel["Pmodel"] is None) or (Dmodel["saved_tc"] is None):
                dc["step_result"].set("model not loaded yet")
                return
            saved_tc = int(Dmodel["saved_tc"])
            end_tc = saved_tc + steps_to_take
            for tc in range(saved_tc, end_tc):
                RD_trainer, RD_brain, RD_process = i_rd.read_SC_CC_data_raw(saved_tc, tc)
                num_record_to_train, loss_this_round = Dmodel["ib"].optimize(Dmodel["Tmodel"], Dmodel["Pmodel"], RD_trainer)
                assert all([recoved_lm == saved_lm for recoved_lm, saved_lm in
                            zip(loss_this_round, RD_brain[1])]), "{0}, {1}".format(loss_this_round, RD_brain)
                assert num_record_to_train == len(RD_trainer[0]), "{0} {1} ".format(num_record_to_train,
                                                                                    len(RD_trainer[0]))
                print("optimized on train count {0}".format(tc))
                Dmodel["saved_tc"] = tc + 1
            dc["step_result"].set("optimize to {0}".format(end_tc))
        return optimize_step_in
    #train data surf
    def train_data_update(self,dc):
        def train_data_update(Even):
            tc = int(dc["step"].get())
            sc = int(dc["save_tc"].get())
            selected_half = dc["R_selected_half"].get()
            self.show_train_datas(self.fig,sc, int(sc)+int(tc),selected_half)
        return train_data_update
    def left_step(self,dc):
        def left_step_in(Event):
            tc = int(dc["step"].get())
            tc = tc - 1 if tc != 0 else lgc.num_train_to_save_model - 1
            dc["step"].set(tc)
            sc = int(dc["save_tc"].get())
            selected_half = dc["R_selected_half"].get()
            self.show_train_datas(self.fig,sc, sc+tc,selected_half)
        return left_step_in
    def right_step(self,dc):
        def right_step_in(Event):
            tc = int(dc["step"].get())
            tc = tc + 1 if tc != lgc.num_train_to_save_model - 1 else 0
            dc["step"].set(tc)
            sc = int(dc["save_tc"].get())
            selected_half = dc["R_selected_half"].get()
            self.show_train_datas(self.fig,sc, sc+tc,selected_half)
        return right_step_in


    #layout fun
    def _one_model_control_layout(self, container, DictM, param ):
        width_left = 15
        width_right = 20
        current_row = 0
        tk.Label(container, text=param["title"], width=width_left+width_right, anchor=tk.CENTER,pady = 10 ).\
            grid(row = current_row, columnspan = 2,sticky=tk.W)
        #padx = 10,

        current_row += 1
        tk.Label(container,text="TC to load:", width=width_left,anchor=tk.W,pady = 3).\
            grid(column=0, row=current_row, sticky=tk.W)
        CBtcs = ttk.Combobox(container, values=param["tcs"], width=width_right)
        CBtcs.set(param["tcs"][0])
        CBtcs.grid(column=1, row=current_row, sticky=tk.W)

        current_row+=1
        tk.Label(container, text="optimize step:", width=width_left,anchor=tk.W,pady = 3).\
            grid(column=0, row=current_row, sticky=tk.W)
        Estep = tk.Entry(container, width=width_right)
        Estep.grid(column=1, row=current_row, sticky=tk.W)

        current_row += 1
        tk.Label(container, text="layer name:", width=width_left,anchor=tk.W,pady = 3).\
            grid(column=0, row=current_row, sticky=tk.W)
        CBln = ttk.Combobox(container, values=param["layerNames"], width=width_right)
        CBln.set(param["layerNames"][0])
        CBln.grid(column=1, row=current_row, sticky=tk.W)

        current_row += 1
        tk.Label(container, text="plot to show:", width=width_left,anchor=tk.W,pady = 3).\
            grid(column=0, row=current_row, sticky=tk.W)
        show_type = tk.StringVar()
        show_type.set(param["plot_type"][0][1])  # initialize
        for idx in range(len(param["plot_type"])):
            tk.Radiobutton(container, text=param["plot_type"][idx][0], variable=show_type,
                           value=param["plot_type"][idx][1],pady = 2).grid(column=1, row=current_row+idx, sticky=tk.W)

        current_row += len(param["plot_type"])
        Bload=tk.Button(container, text="load Model", width=width_left,anchor=tk.CENTER,pady = 3)
        Bload.grid(column=0, row=current_row, sticky=tk.W)
        Sload_result=tk.StringVar()
        tk.Label(container, textvariable=Sload_result, width=width_left, anchor=tk.W, pady=3).\
            grid(column=1, row=current_row,sticky=tk.W)

        current_row += 1
        Bupdate=tk.Button(container, text="update chart", width=width_left,anchor=tk.CENTER,pady = 3)
        Bupdate.grid(column=0, row=current_row, sticky=tk.W)


        current_row += 1
        Bstep=tk.Button(container, text="step optimize", width=width_left,anchor=tk.CENTER,pady = 3)
        Bstep.grid(column=0, row=current_row, sticky=tk.W)
        Sstep_result=tk.StringVar()
        tk.Label(container, textvariable=Sstep_result, width=width_left, anchor=tk.W, pady=3).\
            grid(column=1, row=current_row,sticky=tk.W)


        current_row += 1
        Bleft=tk.Button(container, text="Layer<<", width=width_left,anchor=tk.CENTER,pady = 3)
        Bleft.grid(column=0, row=current_row, sticky=tk.W)
        Bright=tk.Button(container, text=">>Layer", width=width_left,anchor=tk.CENTER,pady = 3)
        Bright.grid(column=1, row=current_row, sticky=tk.W)


        control={}
        control["tcs"]=CBtcs
        control["step"] = Estep
        control["layer_name"]= CBln
        control["show_type"] = show_type
        control["B_load"]=Bload
        control["B_update"] = Bupdate
        control["B_step"] = Bstep
        control["step_result"] = Sstep_result
        control["load_result"] = Sload_result
        control["B_layer_left"] = Bleft
        control["B_layer_right"] = Bright
        return control
    def _one_model_control_bind(self,L_control,L_ln,L_DModel,idx ):
        L_control[idx]["B_layer_left"].bind("<Button-1>", func=self.left_layer(L_control[idx], L_ln, L_DModel[idx], idx))
        L_control[idx]["B_layer_right"].bind("<Button-1>", func=self.right_layer(L_control[idx], L_ln, L_DModel[idx], idx))
        L_control[idx]["B_update"].bind("<Button-1>", func=self.update_plot(L_control[idx],L_DModel[idx], idx))
        L_control[idx]["B_load"].bind("<Button-1>", func=self.load_models(L_control[idx],L_DModel[idx]))
        L_control[idx]["B_step"].bind("<Button-1>", func=self.optimize_step(L_control[idx],L_DModel[idx]))
    def _layer_diff_control_layout(self, container, param):
        width_left = 15
        width_right = 20
        current_row = 0
        tk.Label(container, text=param["title"], width=width_left+width_right, anchor=tk.CENTER,pady = 10 ).\
            grid(row = current_row, columnspan = 2,sticky=tk.W)

        current_row += 1
        tk.Label(container, text="layer name:", width=width_left,anchor=tk.W,pady = 3).\
            grid(column=0, row=current_row, sticky=tk.W)
        CBln = ttk.Combobox(container, values=param["layerNames"], width=width_right)
        CBln.set(param["layerNames"][0])
        CBln.grid(column=1, row=current_row, sticky=tk.W)

        current_row += 1
        tk.Label(container, text="plot to show:", width=width_left,anchor=tk.W,pady = 3).\
            grid(column=0, row=current_row, sticky=tk.W)
        show_type = tk.StringVar()
        show_type.set(param["plot_type"][0][1])  # initialize
        for idx in range(len(param["plot_type"])):
            tk.Radiobutton(container, text=param["plot_type"][idx][0], variable=show_type,
                           value=param["plot_type"][idx][1],pady = 2).grid(column=1, row=current_row+idx, sticky=tk.W)


        current_row += len(param["plot_type"])
        Bleft=tk.Button(container, text="Layer<<", width=width_left,anchor=tk.CENTER,pady = 3)
        Bleft.grid(column=0, row=current_row, sticky=tk.W)
        Bright=tk.Button(container, text=">>Layer", width=width_left,anchor=tk.CENTER,pady = 3)
        Bright.grid(column=1, row=current_row, sticky=tk.W)

        current_row += 1
        Bupdate=tk.Button(container, text="update chart", width=width_left,anchor=tk.CENTER,pady = 3)
        Bupdate.grid(column=0, row=current_row, sticky=tk.W)

        control={}
        control["layer_name"]= CBln
        control["show_type"] = show_type
        control["B_update"] = Bupdate
        control["B_layer_left"] = Bleft
        control["B_layer_right"] = Bright
        return control
    def _layer_diff_control_bind(self,L_control,L_ln, L_DModel,  idx ):
        L_control[idx]["B_layer_left"].bind("<Button-1>", func=self.left_layer(L_control[idx], L_ln, L_DModel, idx))
        L_control[idx]["B_layer_right"].bind("<Button-1>", func=self.right_layer(L_control[idx], L_ln, L_DModel, idx))
        L_control[idx]["B_update"].bind("<Button-1>", func=self.update_plot(L_control[idx],L_DModel,idx))
    def _tk_table_layout(self, container,num_column):

        control={}
        control["table_title"]=tk.StringVar()
        control["column_titles"] = [tk.StringVar() for _ in range(num_column)]
        control["row_0"] = [tk.StringVar()  for _ in range(num_column)]
        control["row_1"] = [tk.StringVar()  for _ in range(num_column)]

        width_cell = 10

        current_row = 0
        tk.Label(container, textvariable=control["table_title"], width=width_cell*num_column, anchor=tk.CENTER,pady = 10).\
            grid(row = current_row, columnspan = num_column,sticky=tk.W)
        for si in [control["column_titles"], control["row_0"], control["row_1"]]:
            current_row += 1
            for idx in range(num_column):
                tk.Label(container, textvariable=si[idx], width=width_cell, anchor=tk.CENTER,pady = 10).\
                    grid(row = current_row, column = idx,sticky=tk.W)

        return control

        '''
        param={}
        param["table_title"]="layer XX"
        param["column_titles"] = ["","min","25\%","50\%","75\%","max"]
        param["row_titles"] = ["weight", "bias"]
        param["data"]       =[[0,1,2,3,4],[10,11,12,13,14]]
        '''
    def _tk_table_update(self, control,param):
        def is_float(str):
            try:
                float(str)
                return True
            except ValueError:
                return False
        control["table_title"].set(param["table_title"])
        row0_content=[param["row_titles"][0]]+param["data"][0]
        row1_content = [param["row_titles"][1]] + param["data"][1]
        llcontent=[param["column_titles"],row0_content,row1_content  ]
        llcontrol=[control["column_titles"], control["row_0"], control["row_1"]]
        for lcontrol, lcontent in zip(llcontrol, llcontent):
            for control, content in zip (lcontrol, lcontent):
                control.set("{0:.2E}".format(float(content))if is_float(content) else content)
    def _tk_train_data_surf_layout(self, container,param):
        width_left = 15
        width_right = 20
        current_row = 0
        tk.Label(container, text=param["title"], width=width_left+width_right, anchor=tk.CENTER,pady = 10 ).\
            grid(row = current_row, columnspan = 2,sticky=tk.W)

        current_row += 1
        tk.Label(container, text="saved tc:", width=width_left,anchor=tk.W,pady = 3).\
            grid(column=0, row=current_row, sticky=tk.W)
        CBsc = ttk.Combobox(container, values=param["l_sc"], width=width_right)
        CBsc.set(param["l_sc"][0])
        CBsc.grid(column=1, row=current_row, sticky=tk.W)

        current_row += 1
        tk.Label(container, text="step:", width=width_left,anchor=tk.W,pady = 3).\
            grid(column=0, row=current_row, sticky=tk.W)
        CBstep = ttk.Combobox(container, values=list(range(param["max_step"])), width=width_right)
        CBstep.set(0)
        CBstep.grid(column=1, row=current_row, sticky=tk.W)

        current_row += 1
        Bleft=tk.Button(container, text="tc<<", width=width_left,anchor=tk.CENTER,pady = 3)
        Bleft.grid(column=0, row=current_row, sticky=tk.W)
        Bright=tk.Button(container, text=">>tc", width=width_left,anchor=tk.CENTER,pady = 3)
        Bright.grid(column=1, row=current_row, sticky=tk.W)

        current_row += 1
        selected_half = [
            ("first half",              "first half"),
            ("second half",             "second half")
        ]
        cf = tk.Frame(container)
        cf.grid(column=0,row=current_row,sticky=tk.W)
        Rselected_half=one_radion_box(cf,selected_half)

        #Bup=tk.Button(container, text="Up", width=width_left,anchor=tk.CENTER,pady = 3)
        #Bup.grid(column=0, row=current_row, sticky=tk.W)
        #Bdown=tk.Button(container, text="Down", width=width_left,anchor=tk.CENTER,pady = 3)
        #Bdown.grid(column=1, row=current_row, sticky=tk.W)

        current_row += 1
        Bupdate=tk.Button(container, text="update chart", width=width_left,anchor=tk.CENTER,pady = 3)
        Bupdate.grid(column=0, row=current_row, sticky=tk.W)


        control={}
        control["save_tc"]= CBsc
        control["step"] = CBstep
        control["B_update"] = Bupdate
        control["B_left"] = Bleft
        control["B_right"] = Bright
        control["R_selected_half"] = Rselected_half


        return control
    def _tk_train_data_surf_bind(self, control):
        control["B_left"].bind("<Button-1>", func=self.left_step(self.train_data_surf_control))
        control["B_right"].bind("<Button-1>", func=self.right_step(self.train_data_surf_control))
        control["B_update"].bind("<Button-1>", func=self.train_data_update(self.train_data_surf_control))

