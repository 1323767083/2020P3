import tkinter as tk
import numpy as np
import pandas as pd
class label_button_entry(tk.Frame):
    def __init__(self, parent, label, entry_defualt,button_title,anchor=tk.W):
        tk.Frame.__init__(self, parent)
        self.pack(anchor=anchor)

        this_label = tk.Label(self, text=label,width=10)
        this_label.grid(column=0, row=0, sticky=anchor)

        self.this_entry = tk.Entry(self, width=10)
        self.this_entry.grid(column=1, row=0, sticky=anchor)
        self.this_entry.delete(0, tk.END)
        self.this_entry.insert(0, entry_defualt)

        self.this_button = tk.Button(self, text=button_title,width=10)
        self.this_button.grid(column=2, row=0, sticky=anchor)

        self.prev_button = tk.Button(self, text="<<",width=10)
        self.prev_button.grid(column=1, row=1, sticky=anchor)

        self.next_button = tk.Button(self, text=">>",width=10)
        self.next_button.grid(column=2, row=1, sticky=anchor)

    def entry(self):
        return self.this_entry.get()

    def set_entry(self, data):
        self.this_entry.delete(0, tk.END)
        self.this_entry.insert(0, data)


    def button(self):
        return self.this_button

    def get_prev_button(self):
        return self.prev_button

    def get_next_button(self):
        return self.next_button


class label_entry(tk.Frame):
    def __init__(self, parent, label, entry_default,anchor=tk.W):
        tk.Frame.__init__(self, parent)
        self.pack(anchor=anchor)

        this_label = tk.Label(self, text=label,width=10)
        this_label.grid(column=0, row=0, sticky=anchor)

        self.this_entry = tk.Entry(self, width=10)
        self.this_entry.grid(column=1, row=0, sticky=anchor)
        self.this_entry.delete(0, tk.END)
        self.this_entry.insert(0, entry_default)

        self.prev_button = tk.Button(self, text="<<",width=10)
        self.prev_button.grid(column=0, row=1, sticky=anchor)

        self.next_button = tk.Button(self, text=">>",width=10)
        self.next_button.grid(column=1, row=1, sticky=anchor)

    def entry(self):
        return self.this_entry.get()

    def set_entry(self, data):
        self.this_entry.delete(0, tk.END)
        self.this_entry.insert(0, data)

    def get_prev_button(self):
        return self.prev_button

    def get_next_button(self):
        return self.next_button

def one_radion_box(parent, title_show_type,main_title=""):
    if len(main_title)!=0:
        this_label = tk.Label(parent, text=main_title,width=10)
        this_label.grid(column=0, row=0, sticky=tk.W)
        radio_start_row = 1
    else:
        radio_start_row = 0
    Variable = tk.StringVar()
    Variable.set(title_show_type[0][1])  # initialize
    for idx in range(len(title_show_type)):
        # for text, short_value in title_show_type:
        b = tk.Radiobutton(parent, text=title_show_type[idx][0],
                           variable=Variable, value=title_show_type[idx][1])
        b.grid(column=0, row=radio_start_row+idx, sticky=tk.W)
    return Variable


def one_label_entry(parent, start_show_row, title, default_value):
    #tk.Label(parent, text=title, width=10).grid(column=0, row=start_show_row, sticky=tk.W)
    tk.Label(parent, text=title).grid(column=0, row=start_show_row, sticky=tk.W)
    entry = tk.Entry(parent, width=10)
    entry.grid(column=1, row=start_show_row, sticky=tk.W)
    entry.delete(0, tk.END)
    entry.insert(0, default_value)
    return entry


class Checkbar(tk.Frame):
    def __init__(self, parent, picks, default_picks,anchor=tk.W, flag_divide=True,main_title=""):
        tk.Frame.__init__(self, parent)
        self.pack(anchor=anchor)
        if len(main_title) != 0:
            this_label = tk.Label(self, text=main_title)#, width=20)
            this_label.grid(column=0, row=0, sticky=anchor)
            check_start_row = 1
        else:
            check_start_row = 0

        self.vars = []
        self.picks=picks
        for idx, pick in enumerate(self.picks):
            var = tk.IntVar()
            if pick in default_picks:
                var.set(1)
            else:
                var.set(0)
            chk = tk.Checkbutton(self, text=pick, variable=var)
            if flag_divide:
                chk.grid(column=idx % 2, row=check_start_row+idx / 2, sticky=anchor)
            else:
                chk.grid(column=0, row=check_start_row+idx, sticky=anchor)
            self.vars.append(var)

    def state(self):
        return map((lambda var: var.get()), self.vars)

    def selcted_item(self):
       selected_states=self.state()
       return [ pick for selected_state, pick in zip(selected_states, self.picks) if selected_state ]

class param_input:
    def __init__(self, parent_frame,radio_titles, entry_titles,main_title=""):
        self.main_title=main_title
        self.radio_titles=radio_titles
        self.entry_titles=entry_titles
        self.parent_frame=parent_frame
        assert len( self.entry_titles)!=0

    def show_elements(self):
        cf = tk.Frame(self.parent_frame)
        cf.pack(anchor=tk.W)
        #cf.grid(column=0, row=0, sticky=tk.W)

        if len(self.main_title) != 0:
            cfr0 = tk.Frame(cf)
            cfr0.grid(column=0, row=0, sticky=tk.W)
            this_label = tk.Label(cfr0, text=self.main_title)
            this_label.grid(column=0, row=0, sticky=tk.W)
            parm_start_row = 1
        else:
            parm_start_row = 0
        cfr1 = tk.Frame(cf)
        cfr1.grid(column=0, row=parm_start_row, sticky=tk.W)

        if len(self.radio_titles)!=0:
            self.radio_control = one_radion_box(cfr1, self.radio_titles)
        row_to_start=len(self.radio_titles)
        self.l_entrys=[]
        for idx,title in enumerate(self.entry_titles):
            self.l_entrys.append(one_label_entry(cfr1, row_to_start+idx, title, 0.0))

    def get_param(self):
        if len(self.radio_titles) == 0:                               #no selection only inputs
            rd_param = []
            for one_entry in self.l_entrys:
                rd_param.append(float(one_entry.get()))
            return rd_param
        elif len(self.radio_titles) == 2:
            selected_radio_control=self.radio_control.get()
            if selected_radio_control ==self.radio_titles[0][1]:     # select origin
                rd_param=[]
            else:                                                    #threahold is selected
                assert  selected_radio_control ==self.radio_titles[1][1]
                rd_param = []
                for one_entry in self.l_entrys:
                    rd_param.append(float(one_entry.get()))
            return rd_param
        elif len(self.radio_titles) >= 3:                            #means selection and inputs should all send
            rd_param = [self.radio_control.get()]
            for one_entry in self.l_entrys:
                rd_param.append(float(one_entry.get()))
            return rd_param

class img_tool:
    def _shift(self, arr, num, fill_value=np.nan):
        result = np.empty_like(arr)
        if num > 0:
            result[:num] = fill_value
            result[num:] = arr[:-num]
        elif num < 0:
            result[num:] = fill_value
            result[:num] = arr[-num:]
        else:
            result = arr
        return result

    def mark_trend(self, img):   #shape  (day, EvalTs)
        imgs=self._shift(img,1)
        imgs[:,0]=img[:,0]
        imgr=img-imgs
        imgr[imgr > 0] = 1
        imgr[imgr < 0] = -1
        return imgr

    def clip(self, img, greater_than=None, less_than=None):
        imgc = np.zeros(img.shape)
        if greater_than is None and less_than is None:
            return imgc
        else:
            np.clip(img, greater_than, less_than, out=imgc)
            return imgc

