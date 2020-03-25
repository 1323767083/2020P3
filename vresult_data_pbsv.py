import pandas as pd
import numpy as np
from vresult_data_com import are_esi_reader

from vcomm import img_tool
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable


class ana_pbsv_data(are_esi_reader):
    def __init__(self, system_name, process_name,Lstock, LEvalT, LYM,lgc):
        are_esi_reader.__init__(self,system_name, process_name )
        self.Lstock, self.LEvalT, self.LYM,self.lgc=Lstock, LEvalT, LYM,lgc

    def get_SV_PB_1stock(self, stock ):
        dfsv=pd.DataFrame()
        #dfpb=[pd.DataFrame() for _ in range(self.lgc.num_action)]### remove .num_action
        dfpb=[pd.DataFrame() for _ in range(self.lgc.train_num_action)]
        for evalT in self.LEvalT:
            flag_opt,dfi=self._read_stock_are(stock, evalT)
            if not flag_opt:
                continue
            dfsv["ET{0}".format(evalT)]=dfi["state_value"]
            #for pb_idx in range(self.lgc.num_action): ### remove .num_action
            for pb_idx in range(self.lgc.train_num_action):
                dfpb[pb_idx]["ET{0}".format(evalT)]=dfi["p{0}".format(pb_idx)]
        npsv=dfsv.values
        #l_nppb=[dfpb[pb_idx].values for pb_idx in range(self.lgc.num_action) ]### remove .num_action
        l_nppb=[dfpb[pb_idx].values for pb_idx in range(self.lgc.train_num_action) ]
        return npsv,l_nppb
class ana_pbsv:
    def __init__(self, system_name, process_name,Lstock, LEvalT, LYM,lgc):
        self.process_name = process_name
        self.system_name = system_name
        self.i_ARE_1stock = ana_pbsv_data(self.system_name, self.process_name,Lstock, LEvalT, LYM,lgc)

        self.Lstock = self.i_ARE_1stock.Lstock
        self.LEvalT = self.i_ARE_1stock.LEvalT
        self.Cidx_stock = 0
        self.npsv, self.l_nppb=self.i_ARE_1stock.get_SV_PB_1stock(self.Lstock[self.Cidx_stock])

    def debug_show(self):
        pass
        #fig, ax_array = plt.subplots(2, 1)
        #self.show(fig, "SH600606")
        #mng = plt.get_current_fig_manager()
        #mng.full_screen_toggle()
        #mng.resize(*mng.window.maxsize())
        #fig.canvas.draw()

    def extract_select_item(self, item):
        if 'state_value' in  item:
            return "sv", 0
        else:
            assert "Action_Prob" in item
            return "pb",int (item[-1])

    def show(self,fig, stock, selected_item, flag_trend,ETP_param):
        allaxes = fig.get_axes()
        for axe in allaxes:
            axe.remove()
        fig.add_subplot(211)
        fig.add_subplot(212)
        allaxes = fig.get_axes()
        fig.subplots_adjust(bottom=0.05, top=0.90, left=0.03, right=0.97, wspace=0.1, hspace=0.3)
        fig.suptitle("Summary at {0} {1}".format(self.system_name, stock), fontsize=14)
        if not(self.Lstock[self.Cidx_stock]==stock):
            self.Cidx_stock=self.Lstock.index(stock)
            self.npsv, self.l_nppb = self.i_ARE_1stock.get_SV_PB_1stock(self.Lstock[self.Cidx_stock])
        data_type, data_idx=self.extract_select_item(selected_item[0])
        self.image_sv_pb(fig, allaxes[0], stock, data_type, data_idx,ETP_param,flag_trend)
        data_type, data_idx = self.extract_select_item(selected_item[1])
        self.image_sv_pb(fig, allaxes[1], stock, data_type, data_idx,ETP_param,flag_trend)

    def image_sv_pb(self, fig, ax, stock, show_type, show_idx, ETP_param,flag_trend=False):
        ax.clear()
        divider3 = make_axes_locatable(ax)
        cax = divider3.append_axes("right", size="1%", pad=0.05)
        if show_type=="sv":
            array=self.npsv
        else:
            assert show_type =="pb"
            array=self.l_nppb[show_idx]
        if len(ETP_param)!=0:
            min_f, max_f=ETP_param
            min_i=int(min_f)
            max_i=int(max_f)
            array=array[:,min_i:max_i]

        ax.set_title("{0} {1}".format(show_type,show_idx))
        if not flag_trend:
            im=ax.imshow(array.T,aspect = 'auto')
        else:
            marked_arry=img_tool().mark_trend(array.T)
            im = ax.imshow(marked_arry, aspect='auto')
        cax.tick_params(labelsize=8)
        fig.colorbar(im, cax=cax, format='%.0e')

class ana_pbsv_detail:
    def __init__(self, system_name, process_name,Lstock, LEvalT, LYM,lgc):
        self.process_name = process_name
        self.system_name = system_name
        self.i_ARE_1stock = ana_pbsv_data(self.system_name, self.process_name,Lstock, LEvalT, LYM,lgc)

        self.Lstock = self.i_ARE_1stock.Lstock
        self.LEvalT = self.i_ARE_1stock.LEvalT
        self.Cidx_stock = 0
        self.npsv, self.l_nppb=self.i_ARE_1stock.get_SV_PB_1stock(self.Lstock[self.Cidx_stock])

    def extract_select_item(self, item):
        assert "Action_Prob" in item
        return int (item[-1])


    def show(self, fig, stock, EvalT,selected_sb):
        if stock !=self.Lstock[self.Cidx_stock]:
            #get data
            self.Cidx_stock=self.Lstock.index(stock)
            self.npsv, self.l_nppb = self.i_ARE_1stock.get_SV_PB_1stock(self.Lstock[self.Cidx_stock])


        allaxes = fig.get_axes()
        for axe in allaxes:
            axe.remove()
        fig.add_subplot(211)

        allaxes = fig.get_axes()
        fig.subplots_adjust(bottom=0.05, top=0.90, left=0.03, right=0.97, wspace=0.1, hspace=0.3)
        fig.suptitle("{0} {1} {2} action probability".format(self.system_name, stock, EvalT), fontsize=14)
        self.show_action_prob(allaxes[0], EvalT,selected_sb)

    def show_action_prob(self, ax, EvalT,selected_sb):
        show_index=self.LEvalT.index(EvalT)
        npr=np.array((1,))
        for idx , nppb in enumerate(self.l_nppb):
            if idx ==0:
                npr = nppb[:,show_index:show_index+1]
            else:
                npr = np.concatenate([npr, nppb[:,show_index:show_index+1]],axis=1)
        show_content,show_label= [], []
        for item in selected_sb:
            selecte_idx=self.extract_select_item(item)
            show_content.append(npr[:, selecte_idx])
            show_label.append(self.i_ARE_1stock.lgc.action_type_dict[selecte_idx])
        ax.stackplot(range(len(npr[:, 0])), show_content, labels=show_label)
        ax.legend()
        return npr
