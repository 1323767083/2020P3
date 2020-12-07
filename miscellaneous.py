import os,json,shutil,time,sys,re
from collections import OrderedDict
import config as sc

def getselected_item_name(l_items, colum_per_row=3,flag_sort=True):
    if flag_sort:
        l_items.sort()
    num_column_last_row=len(l_items)%colum_per_row
    num_row = len(l_items) // colum_per_row
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

    #l_fn_tocopy=[fn for fn in os.listdir(source_system_dir) if fn.endswith(".csv")]
    #for fn in l_fn_tocopy:
    #    sfnwp=os.path.join(source_system_dir, fn)
    #    nfnwp=os.path.join(new_system_dir, fn)
    #    shutil.copy(sfnwp, nfnwp)

def remove_system_sub_dirs(system_dir,SubDirs):

    for sub_dir in SubDirs:
        directory_to_remove = os.path.join(system_dir, sub_dir)
        if os.path.exists(directory_to_remove):
            shutil.rmtree(directory_to_remove)
            print("remove ", directory_to_remove)

def start_tensorboard(port, logdir):
    from tensorboard import program
    import tensorflow as tf
    import logger_comm as lcom

    sys.stdout = open(os.path.join(logdir,str(os.getpid()) + ".out"), "a", buffering=1)
    sys.stderr = open(os.path.join(logdir,str(os.getpid()) + "_error.out"), "a", buffering=1)


    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--port',str(port) ,'--logdir', logdir, '--host','0.0.0.0'])   #--host 0.0.0.0 make can use 192.168.199.100 access
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

    def get_IP_input(self, Title):
        IP_get="Fake"
        while not (len(IP_get.split(""))==4 or len(IP_get)==0):
            IP_get = input("Enter {0} IP address: ".format(Title))
        return IP_get

    #def copy_between_two_machine(self,src_IP_address, des_IP_address):
    def copy_between_two_machine(self):
        src_IP_address=self.get_IP_input("Source")
        des_IP_address = self.get_IP_input("Destination")
        #src_IP_address = input("Enter source IP address: ")


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



def load_config_from_system(system_name):
    param_fnwp = os.path.join(sc.base_dir_RL_system, system_name, "config.json")
    if not os.path.exists(param_fnwp):
        raise ValueError("{0} does not exisit".format(param_fnwp))
    lgc = sc.gconfig()
    lgc.read_from_json(param_fnwp)
    return lgc



def get_GPU_index(GPU_name_str):
    return int(GPU_name_str[-1])

def get_VGPU_lists(lc):
    ll_VGPU_config=[[],[]]
    ll_VGPU_process=[[],[]]

    ll_VGPU_config[int(lc.Brian_core[-1])].append(lc.Brian_gpu_percent)
    ll_VGPU_process[int(lc.Brian_core[-1])].append("Brain")
    for idx,work_core,percent_gpu in enumerate(zip(lc.l_work_core,lc.l_percent_gpu_core_for_work)):
        ll_VGPU_config[int(work_core[-1])].append(percent_gpu)
        ll_VGPU_process[int(work_core[-1])].append("worker_{0}".format(idx))
    for idx,eval_core,percent_gpu in enumerate(zip(lc.l_eval_core,lc.l_percent_gpu_core_for_eva)):
        ll_VGPU_config[int(eval_core[-1])].append(percent_gpu)
        ll_VGPU_process[int(eval_core[-1])].append("eval_{0}".format(idx))

def find_model_surfix(model_dir, eval_loop_count):
    l_model_fn = [fn for fn in os.listdir(model_dir) if "_T{0}.".format(eval_loop_count) in fn]
    if len(l_model_fn) == 2:
        regex = r'\w*(_\d{4}_\d{4}_T\d*).h5'
        match = re.search(regex, l_model_fn[0])
        return match.group(1)
    else:
        return None