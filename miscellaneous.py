import os,json,shutil,time
from collections import OrderedDict
import sys
import config as sc

def getselected_item_name(l_items, colum_per_row=3,flag_sort=True):
    if flag_sort:
        l_items.sort()
    num_column_last_row=len(l_items)%colum_per_row
    num_row = len(l_items) / colum_per_row
    row_per_column=[]
    for idx in range(colum_per_row):
        if idx<num_column_last_row:
            row_per_column.append(num_row+1)
        else:
            row_per_column.append(num_row)
    for row_idx in range(num_row):
        prt_str = ""
        shift_idx=0
        for column_idx in range(colum_per_row):
            prt_str += "({0:<3}){1:<60}".format(row_idx+shift_idx, l_items[row_idx+shift_idx])
            shift_idx+=row_per_column[column_idx]
        print(prt_str)
    if num_column_last_row!=0:
        prt_str = ""
        shift_idx = 0
        for column_idx in range(num_column_last_row):
            prt_str += "({0:<3}){1:<60}".format(num_row+shift_idx, l_items[num_row+shift_idx])
            shift_idx += row_per_column[column_idx]
        print(prt_str)

    selected_item = eval(input("Enter a number: "))
    return l_items[int(selected_item)]

def create_system(RL_systemn_dir):
    print("select the system to copy from")

    lfn=os.listdir(RL_systemn_dir)
    source_system_name=getselected_item_name(lfn)
    source_system_dir=os.path.join(RL_systemn_dir,source_system_name)
    source_fnwp=os.path.join(source_system_dir,"config.json")
    param = json.load(open(source_fnwp, "r"), object_pairs_hook=OrderedDict)
    flag_not_found=True
    new_system_dir=""
    new_system_name=""
    while flag_not_found:
        new_system_name=input("Enter the new system name: ")
        new_system_dir=os.path.join(RL_systemn_dir,new_system_name)
        if not os.path.exists(new_system_dir):
            flag_not_found=False
        else:
            decision=input("{0} already exists, delete it? Yes/No".format(new_system_name))
            if decision=="Yes":
                shutil.rmtree(new_system_dir)
                flag_not_found = False
            else:
                continue
    os.mkdir(new_system_dir)
    #param["RL_system_name"]=new_system_name
    new_param_fnwp=os.path.join(RL_systemn_dir, new_system_name,"config.json")
    json.dump(param,open(new_param_fnwp,"w"),indent=4)

    l_fn_tocopy=[fn for fn in os.listdir(source_system_dir) if fn.endswith(".csv")]
    for fn in l_fn_tocopy:
        sfnwp=os.path.join(source_system_dir, fn)
        nfnwp=os.path.join(new_system_dir, fn)
        shutil.copy(sfnwp, nfnwp)

def remove_system_sub_dirs(system_dir):

    l_sub_dir = ["analysis", "Explore_worker_0", "log", "model", "name_pipe",
                 "record_tb", "record_state", "record_send_buffer","record_sim","tensorboard"]
    l_sub_dir_eval=[dn for dn in os.listdir(system_dir) if "Eval_" in dn]
    l_sub_dir.extend(l_sub_dir_eval)
    for sub_dir in l_sub_dir:
        directory_to_remove = os.path.join(system_dir, sub_dir)
        if os.path.exists(directory_to_remove):
            shutil.rmtree(directory_to_remove)
            print("remove ", directory_to_remove)


def start_tensorboard(port, logdir):
    from tensorboard import program
    import tensorflow as tf
    import logger_comm as lcom
    lcom.setup_tf_logger("tensorboad")
    sys.stdout = open(os.path.join(logdir,str(os.getpid()) + ".out"), "a", buffering=0)
    sys.stderr = open(os.path.join(logdir,str(os.getpid()) + "_error.out"), "a", buffering=0)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--port',str(port) ,'--logdir', logdir])
    url = tb.launch()
    print("******************************",url)
    while True:
        time.sleep(6000)

    #"scp /home/rdchujf/n_workspace/T*.py 192.168.199.100:/home/rdchujf/n_workspace"
    #"scp -r user@your.server.example.com:/path/to/foo /home/user/Desktop/"

class copy_between_two_machine:
    def __init__(self,flag_debug=False):
        self.flag_debug=flag_debug

    def run_command(self, command_to_run):
        print(command_to_run)
        if self.flag_debug:
            return ""
        myCmd = os.popen(command_to_run).read()
        return myCmd


    def copy_between_two_machine(self,src_IP_address, des_IP_address):
        l_sub_dir_to_copy=["analysis", "Eval_0","tensorboard","record_state"]


        command_to_run ="ls {0}".format(sc.base_dir_RL_system)
        if src_IP_address!="":
            command_to_run="ssh {0} {1}".format(src_IP_address,command_to_run)
        str_sns=self.run_command(command_to_run)
        l_system_name=str_sns.split("\n")
        system_name=getselected_item_name(l_system_name)

        # check and remove system directory on des
        command_to_run ="ls {0}".format(sc.base_dir_RL_system)
        if des_IP_address!="":
            command_to_run="ssh {0} {1}".format(des_IP_address,command_to_run)
        str_sns=self.run_command(command_to_run)
        l_system_name=str_sns.split("\n")
        if system_name in l_system_name:
            command_to_run = "rm -rf {0}".format(os.path.join(sc.base_dir_RL_system, system_name))
            if des_IP_address!="":
                command_to_run="ssh {0} {1}".format(des_IP_address,command_to_run)
            self.run_command(command_to_run)

        #create system directory in des
        command_to_run = "mkdir {0}".format(os.path.join(sc.base_dir_RL_system, system_name))
        if des_IP_address!="":
            command_to_run="ssh {0} {1}".format(des_IP_address,command_to_run)
        self.run_command(command_to_run)


        base_dir = os.path.join(sc.base_dir_RL_system,system_name)
        for sub_dir in l_sub_dir_to_copy:
            copy_dir=os.path.join(base_dir, sub_dir)
            src_address="{0}:{1}".format(src_IP_address,copy_dir ) if src_IP_address !="" else copy_dir
            des_address="{0}:{1}".format(des_IP_address,copy_dir ) if des_IP_address !="" else copy_dir
            command_to_run="scp -pr {0} {1}".format(src_address,des_address )
            self.run_command(command_to_run)

        src_address="{0}:{1}".format(src_IP_address,base_dir ) if src_IP_address !="" else base_dir
        des_address="{0}:{1}".format(des_IP_address,base_dir ) if des_IP_address !="" else base_dir
        command_to_run="scp  {0}/*.csv {1}".format(src_address,des_address)
        self.run_command(command_to_run)

        command_to_run="scp  {0}/config.json {1}".format(src_address,des_address)
        self.run_command(command_to_run)

def create_eval_system(new_data_name):  #new_data_name="T5_V2_"
    print("select the system to base from")

    lfn=os.listdir(sc.base_dir_RL_system)
    source_system_name=getselected_item_name(lfn)
    source_system_dir=os.path.join(sc.base_dir_RL_system,source_system_name)
    source_fnwp=os.path.join(source_system_dir,"config.json")
    param = json.load(open(source_fnwp, "r"), object_pairs_hook=OrderedDict)
    flag_not_found=True
    new_system_dir=""
    new_system_name=""
    while flag_not_found:
        new_system_name=input("Enter the new system name: ")
        new_system_dir=os.path.join(sc.base_dir_RL_system,new_system_name)
        if not os.path.exists(new_system_dir):
            flag_not_found=False
        else:
            decision=input("{0} already exists, delete it? Yes/No".format(new_system_name))
            if decision=="Yes":
                shutil.rmtree(new_system_dir)
                flag_not_found = False
            else:
                continue
    os.mkdir(new_system_dir)
    param["RL_system_name"]=new_system_name
    param["data_name"]=new_data_name

    new_param_fnwp=os.path.join(sc.base_dir_RL_system, new_system_name,"config.json")
    json.dump(param,open(new_param_fnwp,"w"),indent=4)

    source_system_model_dir=os.path.join(source_system_dir,"model")
    new_system_model_dir=os.path.join(new_system_dir,"model")
    if not os.path.exists(new_system_model_dir): os.mkdir(new_system_model_dir)
    l_fn_tolink=os.listdir(source_system_model_dir)
    for fn in l_fn_tolink:
        sfnwp=os.path.join(source_system_model_dir,fn)
        dfnwp=os.path.join(new_system_model_dir,fn)
        os.symlink(sfnwp, dfnwp)

    return new_system_name


def load_config_from_system(system_name):
    param_fnwp = os.path.join(sc.base_dir_RL_system, system_name, "config.json")
    if not os.path.exists(param_fnwp):
        raise ValueError("{0} does not exisit".format(param_fnwp))
    lgc = sc.gconfig()
    lgc.read_from_json(param_fnwp)
    return lgc


from data_T5 import FH_RL_data_1stock
def check_correct(stock):
    i=FH_RL_data_1stock("T5",stock)
    l_np_date_s, l_np_large_view, l_np_small_view, l_np_support_view=i.load_main_data()

    for idx,np_support_view in enumerate(l_np_support_view):
        if any(np_support_view[:-2,0]=="True") or (np_support_view[-1, 0]!="True") :
            np_support_view[:-2,0]= False
            np_support_view[-1, 0]= True
            fnwp = i.get_dump_fnwp(stock)
            os.rename(fnwp, fnwp+"_old")
            i.save_main_data([l_np_date_s, l_np_large_view, l_np_small_view, l_np_support_view])
            print("{0} period_idx {1} error found and corrected".format(stock, idx))
            return True
    print("{0} no error found".format(stock))
    return False
