import setproctitle,shutil,os,re,time
import datetime as dt
from multiprocessing import Process
import logger_comm  as lcom
import pipe_comm as pcom
from Buffer_comm import buffer_series

def init_A3C_brain(lgc):
    global lc
    lc=lgc

class Train_Process(Process):
    def __init__(self, process_name, process_idx, LL_input,D_share, E_stop, LE_update_weight, LE_worker_work):
        Process.__init__(self)
        self.logger= lcom.setup_logger(process_name,flag_file_log=lc.flag_brain_log_file,
                                       flag_screen_show=lc.flag_brain_log_screen)
        self.process_name       = process_name
        self.process_idx        = process_idx
        self.LL_input           = LL_input
        self.E_stop             = E_stop
        self.D_share            = D_share
        self.LE_update_weight   = LE_update_weight
        self.LE_worker_work     = LE_worker_work
        self.inp=pcom.name_pipe_cmd(self.process_name)
        self.l_i_bs=[buffer_series() for _ in range (lc.num_workers)]
    def run(self):
        import tensorflow as tf
        from nets import Train_Brain, init_gc,init_virtual_GPU
        lcom.setup_tf_logger(self.process_name)
        tf.random.set_seed(2)
        init_gc(lc) #init_gc(lgc)
        setproctitle.setproctitle("{0}_{1}".format(lc.RL_system_name,self.process_name))
        #with tf.device("/cpu:0" if lc.Brian_gpu_percent==0.0 else "/GPU:0"):
        assert lc.Brian_gpu_percent != 0.0, "Only support GPU"
        virtual_GPU = init_virtual_GPU(11*1024 * lc.Brian_gpu_percent)
        with tf.device(virtual_GPU):
            self.logger.info("{0} {1} started".format(self.process_name, self.process_idx))
            if lc.load_AIO_fnwp != "" and lc.load_config_fnwp != "" and lc.load_weight_fnwp != "":
                l_load_fnwp = [lc.load_AIO_fnwp, lc.load_config_fnwp, lc.load_weight_fnwp]
                train_count_init = lc.start_train_count  # this is a bug should be 0
                self.logger.info("To load brain from {0}".format(l_load_fnwp))
            else:
                l_load_fnwp = []
                train_count_init = 0
                shutil.rmtree(lc.brain_model_dir)  ## otherwise the eval process might not start due to more than 2 _T0 found
                os.makedirs(lc.brain_model_dir)
                self.logger.info("clean model directory and create new brain")
            #i_brain = locals()[lc.CLN_brain_train](GPU_per_program=lc.Brian_gpu_percent, load_fnwps=l_load_fnwp)
            i_brain=locals()[lc.CLN_brain_train](GPU_per_program=lc.Brian_gpu_percent, load_fnwps=l_load_fnwp,train_count_init=train_count_init)

            #flag_weight_ready = False
            Ds={"train_count":                          train_count_init,
                "saved_trc":                            train_count_init,
                "print_trc":                            train_count_init,
                "received_count_sum":                   0,
                "l_recieved_count":                     [0 for _ in range(11)],
                "accumulate_item_get_this_train_count": 0,
                "flag_weight_ready":                    False}
            self.init_weight_update(i_brain, Ds)
            while not self.E_stop.is_set():
                if Ds["flag_weight_ready"]:
                    self.weight_update(i_brain, Ds)
                    Ds["flag_weight_ready"] = False
                else:
                    optimized_flag=self.train_brain(i_brain, Ds)
                    if optimized_flag:
                        Ds["train_count"]+=1
                    Ds["flag_weight_ready"]=self.check_save_print(i_brain, Ds)
                self.name_pipe_cmd(i_brain,Ds)

    def save_AIO_model_weight_config(self, i_brain, train_count):
        ct = dt.datetime.now()
        surfix = "_{0:02d}{1:02d}_{2:02d}{3:02d}_T{4}".format(ct.month, ct.day, ct.hour, ct.minute, train_count)
        AIO_fnwp    = os.path.join(lc.brain_model_dir,"{0}{1}.h5".format(lc.actor_model_AIO_fn_seed, surfix))
        weight_fnwp = os.path.join(lc.brain_model_dir, "{0}{1}.h5".format(lc.actor_weight_fn_seed, surfix))
        config_fnwp = os.path.join(lc.brain_model_dir, "{0}.json".format(lc.actor_config_fn_seed))
        i_brain.save_model([AIO_fnwp, config_fnwp,weight_fnwp ])
        return weight_fnwp

    def find_model_surfix(self, eval_loop_count):
        l_model_fn = [fn for fn in os.listdir(lc.brain_model_dir) if "_T{0}.".format(eval_loop_count) in fn]
        if len(l_model_fn) == 2:
            regex = r'\w*(_\d{4}_\d{4}_T\d*).h5'
            match = re.search(regex, l_model_fn[0])
            return match.group(1)
        else:
            return None

    def init_weight_update(self, i_brain, Ds):
        found_model_surfix=self.find_model_surfix(Ds["train_count"])
        if  found_model_surfix is None:
            last_saved_weights_fnwp=self.save_AIO_model_weight_config(i_brain,Ds["train_count"])
        else:
            actor_weight_fn = "{0}{1}.h5py".format(lc.actor_weight_fn_seed, found_model_surfix)
            last_saved_weights_fnwp = os.path.join(lc.brain_model_dir, actor_weight_fn)
        Ds["saved_trc"] = Ds["train_count"]
        self.logger.info("init_weights ready at {0} start worker weight update".format(last_saved_weights_fnwp))
        self.D_share["weight_fnwp"]=last_saved_weights_fnwp
        for E_update in self.LE_update_weight:
            E_update.set()
        while any([E_update.is_set() for E_update in self.LE_update_weight] ):
            time.sleep(1)
        for E_worker_work in self.LE_worker_work:
            E_worker_work.set()
        self.logger.info("finish worker init weight update and start worker work")

    def weight_update(self,i_brain, Ds):
        last_saved_weights_fnwp = self.save_AIO_model_weight_config(i_brain, Ds["train_count"])
        Ds["saved_trc"] = Ds["train_count"]

        self.logger.info("train_count {0} weights ready at {1} start worker weight update".format(Ds["train_count"],
                                                                                            last_saved_weights_fnwp))
        self.D_share["weight_fnwp"]=last_saved_weights_fnwp
        for E_update in self.LE_update_weight:
            E_update.set()
        '''
        flag_buffer_cleaned=False
        train_count_for_clean_buffer=0
        Ds_copy = dict()
        Ds_copy["train_count"]="Clean"+str(Ds["train_count"])
        Ds_copy["saved_trc"] = "Clean" + str(Ds["saved_trc"])
        Ds_copy["print_trc"] = "Clean" + str(Ds["print_trc"])
        Ds_copy["received_count_sum"]= 0
        Ds_copy["l_recieved_count"]=[0 for _ in range(11)]
        Ds_copy["accumulate_item_get_this_train_count"]= 0

        while any([E_update.is_set() for E_update in self.LE_update_weight] ) or not flag_buffer_cleaned:
            if not flag_buffer_cleaned:
                optimized_flag = self.train_brain(i_brain, Ds_copy)
                if optimized_flag:
                    train_count_for_clean_buffer+=1
                else:
                    flag_buffer_cleaned=True
                    print_str = "train_count {0} totally use {1} train round clean buffer ||".format(Ds["train_count"],
                                                                                    train_count_for_clean_buffer)
                    for idx, count in enumerate(Ds["l_recieved_count"]):
                        if Ds_copy["l_recieved_count"][idx] != 0:
                            print_str = "{0} > {1} times {2} ||".format(print_str, idx * 100,
                                                                        Ds_copy["l_recieved_count"][idx])
                    self.logger.info(print_str)


                    #train_count, saved_trc, print_trc, received_count_sum, l_recieved_count = l_status
                    #train_count += 1
                    #l_status = [train_count, saved_trc, print_trc, received_count_sum, l_recieved_count]
            else:
                time.sleep(1)
        '''
        while any([E_update.is_set() for E_update in self.LE_update_weight]):
            time.sleep(1)
        num_record_cleaned=self.get_train_records(self.delete_list)
        self.logger.info("train_count {0} clean {1} records".format(Ds["train_count"],num_record_cleaned))
        for E_worker_work in self.LE_worker_work:
            E_worker_work.set()
        self.logger.info("train_count {0} finish worker weight update and start worker work".format(Ds["train_count"]))
        return

    def delete_list(self, input_list):
        del input_list[:]

    def train_brain(self, i_brain, Ds):
        accumulate_item_get = self.get_train_records(i_brain.train_push_many)
        Ds["received_count_sum"] += accumulate_item_get
        Ds["l_recieved_count"][int(accumulate_item_get * 1.0 / 100) if accumulate_item_get < 1000 else 10] += 1
        Ds["accumulate_item_get_this_train_count"]=accumulate_item_get
        numb_record_trained, _ = i_brain.optimize()
        if lc.flag_record_state:
            i_brain.mc.rv.recorder_process([Ds["train_count"], Ds["saved_trc"], Ds["print_trc"]])
            i_brain.mc.rv.saver()
        optimized_Flag =True if numb_record_trained>0 else False
        return optimized_Flag

    def check_save_print(self,i_brain, Ds):
        flag_weight_ready = False
        if Ds["train_count"] % lc.num_train_to_save_model == 0 and Ds["train_count"] != lc.start_train_count:  # this need to adjust config setting not same as log
            if Ds["train_count"] != Ds["saved_trc"]:
                flag_weight_ready = True
        if Ds["train_count"] % 250 == 0 and Ds["train_count"] != Ds["print_trc"]:
            print_str = "train_count {0} receive total {1} current buffer size {2}  ||".format(Ds["train_count"],
                                                                                                  Ds["received_count_sum"],
                                                                                                  i_brain.get_buffer_size())
            for idx, _ in enumerate(Ds["l_recieved_count"]):
                if Ds["l_recieved_count"][idx] != 0:
                    print_str = "{0} > {1} times {2} ||".format(print_str, idx * 100, Ds["l_recieved_count"][idx])
            self.logger.info(print_str)
            print_str = "train_count {0}||".format(Ds["train_count"])
            for idx, i_bs in enumerate(self.l_i_bs):
                print_str = "{0} worker {1} send buffer series {2}||".format(print_str, idx, i_bs.get_current())
            self.logger.info(print_str)
            Ds["print_trc"] = Ds["train_count"]
            Ds["l_recieved_count"] = [0 for _ in range(11)]
            Ds["received_count_sum"] = 0
        return flag_weight_ready

    def get_train_records(self, fun_buffer_add):
        accumulate_count_buffer_item_get = 0
        for idx in range(lc.num_workers):
            while True:
                if len(self.LL_input[idx])==0:
                    break
                input_item=self.LL_input[idx].pop(0)
                worker_idx, bs, input_buffer=input_item
                assert worker_idx==idx
                accumulate_count_buffer_item_get += len(input_buffer)
                if self.l_i_bs[worker_idx].valify(bs):
                    self.l_i_bs[worker_idx].set(bs)
                else:
                    self.l_i_bs[worker_idx].set(bs)
                    self.logger.warn("from worker {0} received wrong order last series {1} this serise {2}"
                                     .format(worker_idx, self.l_i_bs[worker_idx].get_current(), bs))
                fun_buffer_add(input_buffer)
        return accumulate_count_buffer_item_get

    def name_pipe_cmd(self,i_brain, Ds):
        cmd_list = self.inp.check_input_immediate_return()
        if cmd_list is not None:
            if cmd_list[0][:-1] == "status":
                print("Brain train_count:{0} count_get_item:{1}" .format(Ds["train_count"],
                                                                         Ds["accumulate_item_get_this_train_count"]))
                print(Ds)

                print("Brain hold buffer size {0}".format(i_brain.tb.get_buffer_size()))
            else:
                print("Unknown command: {0} receive from name pipe: {1}".format(cmd_list, self.inp.np_fnwp))
