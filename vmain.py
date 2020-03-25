import sys,os
import matplotlib
matplotlib.use("TkAgg")
import tkinter as tk
from miscellaneous import getselected_item_name
import recorder
LARGE_FONT = ("Verdana", 12)

from vresult import vresult
from vEreward import vEreward
from vlayer import vlayer
from vstate import vstate
from vtrainer_tk import vtrainer_tk
from vresult_data_com import get_addon_setting
def main(argv):

    if len(argv)==1:
        if argv[0] == "-h" or argv[0] == "--help":
            sys.exit()
    lfun_1=["vresult","vEreward"]   #input: RL system name; process name as input
    lfun_2=["vstate"]               #input: data name
    lfun_3=["vlayer","vtrainer_tk"]               #input: RL system name
    #lfun=["vresult","vEreward","vlayer","vstate"]
    lfun=lfun_1+lfun_2+lfun_3
    if len(lfun)==1:
        view_class =lfun[0]
        print ("{0} selected as default".format(view_class))
    else:
        view_class = getselected_item_name(lfun, colum_per_row=1)

    if view_class in lfun_1:
        system_name = getselected_item_name(os.listdir("/home/rdchujf/n_workspace/RL"))
        leval_process=[dn for dn in os.listdir(os.path.join("/home/rdchujf/n_workspace/RL",
                                                            system_name))if dn.startswith("Eval")]
        if len(leval_process)==1:
            eval_process_name=leval_process[0]
            print ("{0} selected as default".format(eval_process_name))
        else:
            eval_process_name = getselected_item_name(leval_process,colum_per_row=1)
        Lstock, LEvalT, LYM, lgc = get_addon_setting(system_name, eval_process_name)
        param = {"system_name":system_name,"eval_process_name":eval_process_name,
                 "Lstock":Lstock,"LEvalT":LEvalT,"LYM": LYM, "lgc":lgc}
    elif view_class in lfun_2:
        data_name = getselected_item_name(["T5","T5_V2_"], colum_per_row=1)
        param = {"data_name": data_name}
    else:
        assert view_class in lfun_3
        system_name = getselected_item_name(os.listdir("/home/rdchujf/n_workspace/RL"))
        param = {"system_name":system_name}


    page_classes = globals()[view_class]

    app =tk.Tk()
    tk.Tk.wm_title(app, "Visual_in_Tk")
    tk.Tk.attributes(app, '-zoomed', True)

    container = tk.Frame(app)
    container.pack(side="top", fill="both", expand=True)
    container.grid_rowconfigure(0, weight=1)
    container.grid_columnconfigure(0, weight=1)

    frame = page_classes(container, param)
    frame.grid(row=0, column=0, sticky="nsew")
    frame.tkraise()
    app.mainloop()

if __name__ == '__main__':
    # app =tk_main(sys.argv[1:])
    main(sys.argv[1:])


'''
import os
import config as sc
import nets

system_name="CNNCNN_PPO"
param_fnwp = os.path.join(sc.base_dir_RL_system, system_name, "config.json")
if not os.path.exists(param_fnwp):
    raise ValueError("{0} does not exisit".format(param_fnwp))
lgc=sc.gconfig()
lgc.read_from_json(param_fnwp)
nets.init_gc(lgc)
i_brain=nets.Explore_Brain(0.8,lgc.method_name_of_choose_action_for_eval)
'''