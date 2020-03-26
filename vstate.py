from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tkinter as tk
from tkinter import ttk
from data_common import API_qz_data_source_related
from vcomm import Checkbar,label_button_entry
import nets
import env
import numpy as np
import re,os
import config as sc
import Stocklist_comm as scom
from data_T5 import R_T5,R_T5_scale,R_T5_balance,R_T5_skipSwh,R_T5_skipSwh_balance
from data_common import API_trade_date

class vstate(tk.Frame):
    def __init__(self, container, param):
        tk.Frame.__init__(self, container)
        self.data_name=param["data_name"]


        start_s, end_s=API_qz_data_source_related().get_data_state_end_time_s(self.data_name,"SH")
        td = API_trade_date().np_date_s
        #self.td=td[td<="20170731"]
        self.td = td[(td >= start_s)&(td <= end_s)]
        stock  =   "SH600001"
        if self.data_name=="T5":
            date_s="20170731"
        else:
            date_s =   "20180301"

        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ins = visual_state_data(self.data_name,stock, date_s)
        self.config_dic ={"stock": "SH600000", "date_up": date_s, "date_down": date_s, "l_mask_choice": [0,0,0,0],"l_threadhold": [0,0,0,0],"l_scale_choice": [0,0,0,0]}
        self.ins.show_all_in_one(self.fig, self.config_dic)
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2TkAgg(self.canvas, self)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        xiamian=tk.Frame(self)
        xiamian.pack(anchor=tk.W)

        xiamian_0 = tk.Frame(xiamian)
        xiamian_0.grid(column=0,row=0,sticky=tk.W)
        Fstock = tk.Frame(xiamian_0)
        Fstock.pack(anchor=tk.W)
        tk.Label(Fstock, text="stock",width=10).grid(column=0, row=0, sticky=tk.W)
        self.Estock = tk.Entry(Fstock, width=10)
        self.Estock.grid(column=1, row=0, sticky=tk.W)
        self.Estock.delete(0, tk.END)
        self.Estock.insert(0, self.config_dic["stock"])
        self.Edate_up, self.Bprev_up, self.Bnex_up =self._entry_pre_next_control(Fstock, "date(up)", self.config_dic["date_up"],2)
        self.Edate_down, self.Bprev_down, self.Bnex_down =self._entry_pre_next_control(Fstock, "date(down)", self.config_dic["date_down"], 4)


        l_mask_op = ["keep","less_than",  "greater_than"]
        l_set_y_scale=["preset min/max", "iamge min/max"]
        l_mask_title=["mask(lv up)","mask(sv up)","mask(lv down)","mask(sv down)"]
        l_scale_title=["scale(lv up)","scale(sv up)","scale(lv down)","scale(sv down)"]
        start_row=1
        self.l_DC=[{} for _ in range(4)]
        for idx in range(4):
            self._control_summary(xiamian, self.l_DC[idx], l_mask_op, l_mask_title[idx], l_set_y_scale, l_scale_title[idx],
                                  col=start_row+idx)

        xiamian_5 = tk.Frame(xiamian)
        xiamian_5.grid(column=5, row=0,sticky=tk.W)
        self.BAOI = tk.Button(xiamian_5, text="show_AIO",width=10)
        self.BAOI.grid(column=0, row=0, sticky=tk.W)

        self.BOD = tk.Button(xiamian_5, text="show one day",width=10)
        self.BOD.grid(column=0, row=1, sticky=tk.W)

        self.BDiff = tk.Button(xiamian_5, text="show Diff",width=10)
        self.BDiff.grid(column=0, row=2, sticky=tk.W)

        self.BAOI.bind("<Button-1>", func=self.frame_AIO_update)
        self.BOD.bind("<Button-1>", func=self.frame_OD_update)
        self.BDiff.bind("<Button-1>", func=self.frame_diff_update)

        self.Bprev_up.bind("<Button-1>", func=self.up_pre_date)
        self.Bnex_up.bind("<Button-1>", func=self.up_nex_date)
        self.Bprev_down.bind("<Button-1>", func=self.down_pre_date)
        self.Bnex_down.bind("<Button-1>", func=self.down_nex_date)

    def _get_all_input(self):
        self.config_dic.clear()

        self.config_dic["stock"]=self.Estock.get()
        self.config_dic["date_up"] = str(self.Edate_up.get())
        self.config_dic["date_down"] = str(self.Edate_down.get())
        self.config_dic["l_mask_choice"] =[]
        self.config_dic["l_threadhold"] = []
        self.config_dic["l_scale_choice"] = []
        for idx in range(4):
            self.config_dic["l_mask_choice"].append(self.l_DC[idx]["mask_choice"].get())
            self.config_dic["l_threadhold"].append(float(self.l_DC[idx]["threadhold"].get()))
            self.config_dic["l_scale_choice"].append(self.l_DC[idx]["scale_choice"].get())

    def _control_summary(self, frame, DC, mask_op_content,mask_op_title, scale_content, scale_title, col ):
        show_frame = tk.Frame(frame)
        show_frame.grid(column=col, row=0,sticky=tk.W)
        DC["mask_choice"],DC["threadhold"]=self._mask_control(show_frame,mask_op_content, 0, 0,mask_op_title)
        DC["scale_choice"],_= self._mask_control(show_frame,scale_content, 0, None,scale_title)

    def _entry_pre_next_control(self, frame, label, value, start_row):
        tk.Label(frame, text=label,width=10).grid(column=0, row=start_row, sticky=tk.W)
        Edate = tk.Entry(frame, width=10)
        Edate.grid(column=1, row=start_row, sticky=tk.W)
        Edate.delete(0, tk.END)
        Edate.insert(0, value)
        Bprev = tk.Button(frame, text="<<",width=5)
        Bprev.grid(column=0, row=start_row+1, sticky=tk.E)
        Bnext = tk.Button(frame, text=">>",width=5)
        Bnext.grid(column=1, row=start_row+1, sticky=tk.W)
        return Edate,Bprev,Bnext

    def _mask_control(self, frame, l_choice, choice_value, threadhold_value, label):
        Fsvmask = tk.Frame(frame)
        Fsvmask.pack(anchor=tk.W)
        tk.Label(Fsvmask,text=label, justify=tk.LEFT).grid(column=0, row=0, sticky=tk.W,padx=20)
        mask_choice = tk.IntVar()
        mask_choice.set(choice_value)  # initializing the choice, i.e. Python
        for val, mask_op in enumerate(l_choice):
            tk.Radiobutton(Fsvmask, text=mask_op, variable=mask_choice,value=val,padx=20).grid(column=0, row=1+val,
                                                                                               sticky=tk.W)
        if not (threadhold_value is None):
            Emask = tk.Entry(Fsvmask, width=10)
            Emask.grid(column=0, row=5, sticky=tk.W, padx=40)
            Emask.delete(0, tk.END)
            Emask.insert(0, threadhold_value)
        else:
            Emask = None
        return mask_choice,Emask

    def frame_AIO_update(self, Event):
        self._get_all_input()
        self.ins.show_all_in_one(self.fig, self.config_dic)
        self.canvas.draw()

    def frame_OD_update(self, Event):
        self._get_all_input()
        self.ins.show_one_day(self.fig, self.config_dic)
        self.canvas.draw()

    def frame_diff_update(self, Event):
        self._get_all_input()
        self.ins.show_diff(self.fig, self.config_dic)
        self.canvas.draw()

    def up_pre_date(self, event):
        self._pre_date(self.Edate_up)
    def up_nex_date(self, event):
        self._next_date(self.Edate_up)
    def down_pre_date(self, event):
        self._pre_date(self.Edate_down)
    def down_nex_date(self, event):
        self._next_date(self.Edate_down)
    def _pre_date(self, entry):
        current_date=entry.get()
        found_idx=np.where(self.td == current_date)
        if len(found_idx[0])==1:
            idx=found_idx[0][0]
            if idx>0:
                idx-=1
                entry.delete(0, tk.END)
                entry.insert(0, self.td[idx])

    def _next_date(self, entry):
        current_date = entry.get()
        found_idx = np.where(self.td == current_date)
        if len(found_idx[0]) == 1:
            idx = found_idx[0][0]
            if idx < len(self.td)-1:
                idx += 1
                entry.delete(0, tk.END)
                entry.insert(0, self.td[idx])

class visual_state_data:
    #data_name="T5"
    def __init__(self,data_name,stock, date_s, fun_R_T5="R_T5"): #fun_R_T5 in ["R_T5","R_T5_scale","R_T5_balance"]
        self.data_name=data_name
        self.stock=stock
        self.date_s = date_s
        self.fun_R_T5=fun_R_T5
        #self.ins = R_T5(self.data_name, self.stock)
        self.ins = globals()[fun_R_T5](self.data_name, self.stock)

    def _get_data(self, stock, date_s):
        if stock!=self.stock:
            self.stock=stock
            #self.ins = R_T5(self.data_name, self.stock)
            self.ins = globals()[self.fun_R_T5](self.data_name, self.stock)

        state, support_view = self.ins.read_one_day_data(date_s)
        lv, sv = state

        return lv, sv, support_view

    def _init_axes(self, fig, rows, cols):
        allaxes = fig.get_axes()
        if len(allaxes) != rows*cols*2:
            for axe in allaxes:
                axe.remove()
            for idx in range (rows*cols):
                ax=fig.add_subplot(rows,cols,idx+1)
                divider3 = make_axes_locatable(ax)
                cax3 = divider3.append_axes("right", size="5%", pad=0.05)
            allaxes = fig.get_axes()
        return allaxes

    def _clear_setting_ax_cax(self, ax, cax):
        # ax.grid('on', linestyle='--')
        ax.clear()
        cax.clear()

        ax.grid('off')
        ## ax.set_adjustable('box-forced')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        # turn off ticks
        #ax.xaxis.set_ticks_position('none')
        #ax.yaxis.set_ticks_position('none')
        #l_shape=list(image_shape)
        #print l_shape
        #ax.set_xlim([0, l_shape[1]])
        #ax.set_ylim([0, l_shape[0]])
        #ax.set_xticklabels(range(l_shape[1]))
        #ax.set_yticklabels(range(l_shape[0]))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        return ax, cax

    def mask_array(self, npao,op, threadhold=0):
        npa=npao.copy()
        if op==1:#"less_than":
            npa_idx=npa>=threadhold
            npa[npa_idx]=threadhold
        elif op==2:#"greater_than":
            npa_idx=npa<=threadhold
            npa[npa_idx]=threadhold
        else:
            assert op==0 #"keep"
            assert threadhold==0
        return npa

    def show_one_day_data_with_axes(self,fig,l_axes,inputs, config, part):
        #allaxes = fig.get_axes()
        lv,sv,support_view=inputs
        lv_mask_op=config["l_mask_choice"][2*part]
        lv_mask_threadhold=config["l_threadhold"][2*part]
        lv_scale_op=config["l_scale_choice"][2*part]
        sv_mask_op=config["l_mask_choice"][2*part+1]
        sv_mask_threadhold=config["l_threadhold"][2*part+1]
        sv_scale_op=config["l_scale_choice"][2*part+1]

        sub_title=self._generate_sub_title(support_view)
        for idx in range(2):
            if idx==0:
                image_shape=(20,17)
                ax, cax = l_axes[idx*2], l_axes[idx*2+1]
                self._clear_setting_ax_cax(ax, cax)

                ax.set_title("{0}_lv_{1}_{2}".format(sub_title,lv_mask_op, lv_mask_threadhold), fontdict={"fontsize": 10}, pad=2)
                if lv_scale_op==0:
                    im = ax.imshow(self.mask_array(lv,lv_mask_op, lv_mask_threadhold).reshape(image_shape), aspect='auto',vmin=-4, vmax=4)
                else:
                    assert lv_scale_op == 1
                    im = ax.imshow(self.mask_array(lv, lv_mask_op, lv_mask_threadhold).reshape(image_shape),aspect='auto')
                ax.set_xticks(list(range(17)))
                ax.set_xticklabels([str(idx+1) for idx in range(17)])
                ax.set_yticks(list(range(20)))
                ax.set_yticklabels([str(idx+1) for idx in range(20)])
                cax.tick_params(labelsize=8)
                cbar = fig.colorbar(im, cax=cax, format='%.0e')

            else:
                assert idx==1
                image_shape = (20,50)
                ax, cax = l_axes[idx * 2], l_axes[idx * 2 + 1]
                self._clear_setting_ax_cax(ax, cax)
                ax.set_title("sv_{0}_{1}".format(sv_mask_op, sv_mask_threadhold), fontdict={"fontsize": 10}, pad=2)
                if sv_scale_op==0:
                    im = ax.imshow(self.mask_array(sv,sv_mask_op, sv_mask_threadhold).reshape(image_shape), aspect='auto',vmin=-4, vmax=4)
                else:
                    assert sv_scale_op==1
                    im = ax.imshow(self.mask_array(sv, sv_mask_op, sv_mask_threadhold).reshape(image_shape),aspect='auto')
                ax.set_xticks(list(range(50)))
                ax.set_xticklabels([str(idx+1) for idx in range(50)])
                ax.set_yticks(list(range(20)))
                ax.set_yticklabels([str(idx+1) for idx in range(20)])

                cax.tick_params(labelsize=8)
                cbar = fig.colorbar(im, cax=cax, format='%.0e')

    def show_two_day_diff_with_axes(self,fig,l_axes, first_day_inputs, sencond_day_input):
        ulv, usv, usupport_view=first_day_inputs
        dlv, dsv, dsupport_view=sencond_day_input
        for idx in range(2):
            if idx==0:
                image_shape=(20,17)
                ax, cax = l_axes[idx*2], l_axes[idx*2+1]
                self._clear_setting_ax_cax(ax, cax)
                sub_title = self._generate_sub_title(usupport_view)
                ax.set_title("{0}_lvdiff".format(sub_title), fontdict={"fontsize": 10}, pad=2)
                im = ax.imshow(self.mask_array(ulv-dlv, 0, 0).reshape(image_shape),aspect='auto')
                cax.tick_params(labelsize=8)
                cbar = fig.colorbar(im, cax=cax, format='%.0e')
                ax.set_xticks(list(range(17)))
                ax.set_xticklabels([str(idx+1) for idx in range(17)])
                ax.set_yticks(list(range(20)))
                ax.set_yticklabels([str(idx+1) for idx in range(20)])

            else:
                assert idx==1
                image_shape = (20,50)
                ax, cax = l_axes[idx * 2], l_axes[idx * 2 + 1]
                self._clear_setting_ax_cax(ax, cax)
                sub_title = self._generate_sub_title(dsupport_view)
                ax.set_title("{0}_svdiff".format(sub_title),fontdict={"fontsize": 10}, pad=2)
                im = ax.imshow(self.mask_array(usv-dsv, 0, 0).reshape(image_shape),aspect='auto')
                cax.tick_params(labelsize=8)
                cbar = fig.colorbar(im, cax=cax, format='%.0e')
                ax.set_xticks(list(range(50)))
                ax.set_xticklabels([str(idx+1) for idx in range(50)])
                ax.set_yticks(list(range(20)))
                ax.set_yticklabels([str(idx+1) for idx in range(20)])


    def _generate_sub_title(self,support_view ):
        def adjust_key_name(origin_title):
            l_remove_prefix = ["this_trade_day_", "stock_"]
            for remove_prefix in l_remove_prefix:
                if remove_prefix in origin_title:
                    return origin_title[len(remove_prefix):]
            return origin_title
        l_key = list(support_view.keys())
        l_key.remove("stock")
        l_key.remove("date")
        l_key.remove("last_day_flag")
        title = "{0} {1}".format(support_view["stock"], support_view["date"])
        for key in l_key:
            if type(support_view[key]) is float:
                title = "{0} {1}:{2:.2f}".format(title, adjust_key_name(key), support_view[key])
            else:
                title = "{0} {1}:{2}".format(title, adjust_key_name(key), support_view[key])
        return title

    def show_all_in_one(self,fig, config_dic):
        stock=config_dic["stock"]
        udate = config_dic["date_up"]
        ddate = config_dic["date_down"]
        ulv, usv, usupport_view=self._get_data(stock, udate)
        dlv, dsv, dsupport_view = self._get_data(stock, ddate)
        allaxes = self._init_axes(fig, 3, 2)
        fig.subplots_adjust(bottom=0.05, top=0.98, left=0.02, right=0.97, wspace=0.1, hspace=0.2)
        self.show_one_day_data_with_axes(fig,allaxes[0:4],[ulv, usv, usupport_view], config_dic, part=0)
        self.show_one_day_data_with_axes(fig,allaxes[4:8],[dlv, dsv, dsupport_view], config_dic, part=1)
        self.show_two_day_diff_with_axes(fig,allaxes[8:12],[ulv, usv, usupport_view],[dlv, dsv, dsupport_view])

    def show_one_day(self,fig, config_dic):
        stock=config_dic["stock"]
        udate = config_dic["date_up"]
        ulv, usv, usupport_view=self._get_data(stock, udate)
        allaxes=self._init_axes(fig, 1, 2)
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.1, hspace=0.1)
        self.show_one_day_data_with_axes(fig,allaxes,[ulv, usv, usupport_view], config_dic, part=0)

    def show_diff(self,fig, config_dic):
        stock=config_dic["stock"]
        udate = config_dic["date_up"]
        ddate = config_dic["date_down"]
        ulv, usv, usupport_view=self._get_data(stock, udate)
        dlv, dsv, dsupport_view = self._get_data(stock, ddate)
        allaxes = self._init_axes(fig, 1, 2)
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.1, hspace=0.1)
        self.show_two_day_diff_with_axes(fig,allaxes,[ulv, usv, usupport_view],[dlv, dsv, dsupport_view])


'''
In [58]: x = np.array([1,3,-1, 5, 7, -1]) 
In [59]: mask = (x < 0) 
In [60]: mask 
Out[60]: array([False, False,  True, False, False,  True], dtype=bool) 
We can see from the preceding example that by applying the < logic sign that we applied scalars to a NumPy Array and the naming of a new array to mask, it's still vectorized and returns the True/False boolean with the same shape of the variable x indicated which element in x meet the criteria:

Copy
In [61]: x [mask] = 0 
In [62]: x 
Out[62]: array([1, 3, 0, 5, 7, 0]) 
Using the mask, we gain the ability to access or replace any element value in our...



import matplotlib.pyplot as plt
import imp
import visual_state
fig = plt.figure()
imp.reload(visual_state)
i=visual_state.visual_state_data("SH600177", "20170731")
i.show_state(fig, {"stock":"SH600177","date":"20170731","lv_mask_op":"keep","lv_mask_threadhold":0, "sv_mask_op":"keep","sv_mask_threadhold":0})
'''

