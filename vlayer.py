import os, re, pickle
import progressbar
import nets
import config as sc
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk #NavigationToolbar2TkAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

import tkinter as tk
from tkinter import ttk

class get_layer_wb:
    def __init__(self, system_name):

        param_fnwp = os.path.join(sc.base_dir_RL_system, system_name, "config.json")
        if not os.path.exists(param_fnwp):
            raise ValueError("{0} does not exisit".format(param_fnwp))
        self.lgc = sc.gconfig()
        self.lgc.read_from_json(param_fnwp)

        nets.init_gc(self.lgc)
        self.i_brain=nets.Explore_Brain(0.8,self.lgc.method_name_of_choose_action_for_eval)

        self.dir_analysis=os.path.join(sc.base_dir_RL_system,system_name, "analysis")
        if not os.path.exists(self.dir_analysis): os.makedirs(self.dir_analysis)

        l_ana_sub_dir=["pre_layers_weight"]
        for sub in l_ana_sub_dir:
            dir_sub = os.path.join(self.dir_analysis, sub)
            if not os.path.exists(dir_sub): os.makedirs(dir_sub)



        self.LEvalT=[int(re.findall(r'T(\d+)', fn)[0]) for fn in os.listdir(self.lgc.brain_model_dir)
                     if fn.startswith("weight_")]
        assert len(self.LEvalT)>=1
        self.LEvalT.sort()
        self.load_weight_one_evalT(self.LEvalT[0])
        self.Cidx_EvalT=0
        m = self.i_brain.Pmodel
        self.l_layer_name = []
        self.l_layer_output_shape = []
        for layer in m.layers:
            if self.check_layer_type(layer.name)!="Non_param_layer":
                self.l_layer_name.append(layer.name)
                self.l_layer_output_shape.append(layer.output_shape)

    def check_layer_type(self,layer_name):
        if len(re.findall(r'TD\w+_conv', layer_name)) == 1:
            return "TDConv"
        elif len(re.findall(r'\w+_conv', layer_name)) == 1:
            return "Conv"
        elif len(re.findall(r'\w+Dense\d+', layer_name)) == 1:
            return "Dense"
        elif layer_name in ["Act_prob", "State_value"]:
            return "Dense"
        else:
            return "Non_param_layer"


    def load_weight_one_evalT(self,EvalT):
        l_model_fn = [fn for fn in os.listdir(self.lgc.brain_model_dir) if "_T{0}.".format(EvalT) in fn and fn.startswith("weight")]
        if len(l_model_fn) == 1:
            weight_fn=l_model_fn[0]
        elif len(l_model_fn)==0:
            return False
        else:
            assert False,"no weight found for {0}".format(EvalT)
        self.i_brain.load_weight(os.path.join(self.lgc.brain_model_dir, weight_fn))
        return True

    def get_1layer_wbs(self, layer_name):
        temp_store_fn=os.path.join(self.dir_analysis,"pre_layers_weight", "weights_{0}_{1}_{2}.pickle".
                                   format(layer_name, self.LEvalT[0],self.LEvalT[-1] ) )
        if os.path.exists(temp_store_fn):
            l_W=pickle.load(open(temp_store_fn, "r"))
            assert len(l_W)==len(self.LEvalT)
            return l_W
        l_W=[]
        bar = progressbar.ProgressBar(maxval=len(self.LEvalT),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        for idx, EvalT in enumerate(self.LEvalT):
            if not self.load_weight_one_evalT(EvalT):
                assert False

            with self.i_brain.default_graph.as_default():
                with self.i_brain.session.as_default():
                    W=self.i_brain.Pmodel.get_layer(name=layer_name).get_weights()
            assert W is not None
            l_W.append(W)
            bar.update(idx)
        pickle.dump(l_W, open(temp_store_fn,"wb"))
        return l_W

class show_layer_wb:
    def __init__(self, system_name):
        self.system_name    =   system_name
        self.i_get_data=get_layer_wb(system_name)
        self.LEvalT=self.i_get_data.LEvalT
        self.layer_name=None
        self.l_W=None


    def debug_show(self,layer_name, channel=-1):
        #layer_name="P_LVSV_conv0_conv"
        fig, ax_array = plt.subplots(2, 1)
        self.show(fig,layer_name,channel)
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        #mng.resize(*mng.window.maxsize())
        fig.canvas.draw()

    def show(self, fig, layer_name,selected_output_channel):
        #if selected_output_channel==-1 means all channels

        allaxes = fig.get_axes()
        for axe in allaxes:
            axe.remove()
        fig.add_subplot(211)
        fig.add_subplot(212)
        allaxes = fig.get_axes()

        fig.suptitle("{0} layer:{1} weight".format(self.system_name,layer_name), fontsize=14)
        fig.subplots_adjust(bottom=0.05, top=0.9, left=0.03, right=0.97, wspace=0.1, hspace=0.3)
        self.show_layer_wb(fig, layer_name, selected_output_channel)

    def show_layer_wb(self,fig,layer_name,selected_output_channel ):

        #if selected_output_channel==-1 means all channels
        #selected_output_channel=10
        soc=selected_output_channel


        if self.layer_name is None:
            self.l_W = self.i_get_data.get_1layer_wbs(layer_name)
            self.layer_name=layer_name
        elif self.layer_name!=layer_name:
            self.l_W = self.i_get_data.get_1layer_wbs(layer_name)
            self.layer_name=layer_name
        else:
            pass

        layer_type = self.i_get_data.check_layer_type(layer_name)
        if layer_type in ["TDConv","Conv"]:
            flag_shape_3=True
        elif layer_type in ["Dense"]:
            flag_shape_3 = False
        else:
            assert layer_type == "Non_param_layer"
            return

        l_weight=[]
        l_bias=[]
        for W in  self.l_W:
            if flag_shape_3:
                l_weight.append(W[0][:,:,soc].reshape(-1,) if soc!=-1 else W[0].reshape(-1,))
            else:
                l_weight.append(W[0][:, soc].reshape(-1, ) if soc != -1 else W[0].reshape(-1, ))
            l_bias.append(W[1].reshape(-1,))

        np_weight=np.stack(l_weight, axis=1)
        np_bias=np.stack(l_bias, axis=1)

        allaxes = fig.get_axes()
        ax=allaxes[0]
        ax.clear()
        divider3 = make_axes_locatable(ax)
        cax = divider3.append_axes("right", size="1%", pad=0.05)
        ax.set_title("{0} weight {1}".format(layer_name, "channel {0}".format(soc if soc!=-1 else "all channels")))

        im=ax.imshow(np_weight.T, aspect='auto')
        cax.tick_params(labelsize=8)
        cbar = fig.colorbar(im, cax=cax, format='%.0e')

        ax = allaxes[1]
        ax.clear()
        ax=allaxes[1]
        divider3 = make_axes_locatable(ax)
        cax = divider3.append_axes("right", size="1%", pad=0.05)
        ax.set_title("{0} bias {1}".format(layer_name, "all channels"))
        im=ax.imshow(np_bias.T,  aspect='auto')
        cax.tick_params(labelsize=8)
        cbar = fig.colorbar(im, cax=cax, format='%.0e')


class vlayer(tk.Frame):
    def __init__(self, container,param ):

        tk.Frame.__init__(self, container)
        system_name=param["system_name"]
        self.ip=show_layer_wb(system_name)
        self.fig = Figure(figsize=(5, 5), dpi=100)

        self.LEvalT=self.ip.i_get_data.LEvalT
        self.Cidx_EvalT=self.ip.i_get_data.Cidx_EvalT

        self.l_layer_name=self.ip.i_get_data.l_layer_name
        self.l_layer_output_shape = self.ip.i_get_data.l_layer_output_shape

        layer_name_to_show=None
        for layer_name in self.l_layer_name:
            if self.ip.i_get_data.check_layer_type(layer_name)!="Non_param_layer":
                layer_name_to_show=layer_name
                break

        self.ip.show(self.fig, layer_name_to_show, -1)
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        #self.canvas.show()
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        #toolbar = NavigationToolbar2TkAgg(self.canvas, self)
        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


        xiamian=tk.Frame(self)
        xiamian.pack(anchor=tk.W)
        layout_column = 0
        cf = tk.Frame(xiamian)
        cf.grid(column=layout_column,row=0,sticky=tk.W)
        self.S_layer_name = tk.StringVar()
        self.C_layer_name = ttk.Combobox(cf, textvariable=self.S_layer_name)
        self.C_layer_name["values"] = self.l_layer_name
        self.C_layer_name["state"] = "readonly"
        self.C_layer_name.pack(anchor=tk.W)

        self.C_layer_name.current(0)

        layout_column += 1
        
        # update buttom
        cf = tk.Frame(xiamian)
        cf.grid(column=layout_column,row=0,sticky=tk.W)
        self.update_button = tk.Button(cf, text="update",width=10)
        self.update_button.pack(anchor=tk.W)
        layout_column += 1

        self.update_button.bind("<Button-1>", func=self.frame_update)

    def frame_update(self, Event):

        layer_name=self.C_layer_name.get()
        self.ip.show(self.fig, layer_name, -1)
        self.canvas.draw()



