import setproctitle,shutil,os,time,re
import datetime as dt
from multiprocessing import Process
import logger_comm  as lcom
import pipe_comm as pcom
from Buffer_comm import buffer_series
from miscellaneous import find_model_surfix

class Train_Process(Process):
    def __init__(self, lc, LL_input,D_share, E_stop, E_update_weight):
        Process.__init__(self)
        self.lc,  self.LL_input,self.D_share, self.E_stop, self.E_update_weight=lc, LL_input, D_share, E_stop, E_update_weight
        self.process_name=self.lc.server_process_name_seed
        self.logger= lcom.setup_logger(self.lc,self.process_name,flag_file_log=lc.flag_brain_log_file,flag_screen_show=lc.flag_brain_log_screen)
        self.inp=pcom.name_pipe_cmd(self.lc,self.process_name)
        self.l_i_bs=[buffer_series() for _ in range (lc.num_workers)]
    def run(self):
        import tensorflow as tf
        from nets import Train_Brain, init_virtual_GPU
        tf.random.set_seed(2)
        setproctitle.setproctitle("{0}_{1}".format(self.lc.RL_system_name,self.process_name))
        assert self.lc.Brian_gpu_percent != 0.0, "Only support GPU"
        virtual_GPU = init_virtual_GPU(self.lc.Brian_gpu_percent)
        with tf.device(virtual_GPU):
            self.logger.info("{0} started".format(self.process_name))
            if self.lc.load_AIO_fnwp != "" and self.lc.load_config_fnwp != "" and self.lc.load_weight_fnwp != "":
                l_load_fnwp = [self.lc.load_AIO_fnwp, self.lc.load_config_fnwp, self.lc.load_weight_fnwp]
                #train_count_init = self.lc.start_train_count
                self.train_count_init=int(re.findall(r'T(\d+)', self.lc.load_AIO_fnwp)[0])
                self.logger.info("To load brain from {0}".format(l_load_fnwp))
            else:
                l_load_fnwp = []
                self.train_count_init = 0
                if os.path.exists(self.lc.brain_model_dir):
                    shutil.rmtree(self.lc.brain_model_dir)  ## otherwise the eval process might not start due to more than 2 _T0 found
                os.makedirs(self.lc.brain_model_dir)
                self.logger.info("clean model directory and create new brain")
            i_brain=locals()[self.lc.CLN_brain_train](self.lc,GPU_per_program=self.lc.Brian_gpu_percent,
                                                      load_fnwps=l_load_fnwp,train_count_init=self.train_count_init)

            #flag_weight_ready = False
            Ds={"train_count":                          self.train_count_init,
                "saved_trc":                            self.train_count_init,
                "print_trc":                            self.train_count_init,
                "received_count_sum":                   0,
                "l_recieved_count":                     [0 for _ in range(11)],
                "accumulate_item_get_this_train_count": 0,
                "flag_weight_ready":                    False}
            self.logger.info("{0} brain ready".format(self.process_name))
            self.init_weight_update(i_brain, Ds)
            self.logger.info("{0} worker weight ready".format(self.process_name))
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
        AIO_fnwp    = os.path.join(self.lc.brain_model_dir,"{0}{1}.h5".format(self.lc.actor_model_AIO_fn_seed, surfix))
        weight_fnwp = os.path.join(self.lc.brain_model_dir, "{0}{1}.h5".format(self.lc.actor_weight_fn_seed, surfix))
        config_fnwp = os.path.join(self.lc.brain_model_dir, "{0}.json".format(self.lc.actor_config_fn_seed))
        i_brain.save_model([AIO_fnwp, config_fnwp,weight_fnwp ])
        return weight_fnwp


    def init_weight_update(self, i_brain, Ds):
        found_model_surfix = find_model_surfix(self.lc.brain_model_dir,Ds["train_count"], self.lc.flag_train_store_AIO_model)
        if  found_model_surfix is None:
            last_saved_weights_fnwp=self.save_AIO_model_weight_config(i_brain,Ds["train_count"])
        else:
            actor_weight_fn = "{0}{1}.h5".format(self.lc.actor_weight_fn_seed, found_model_surfix)
            last_saved_weights_fnwp = os.path.join(self.lc.brain_model_dir, actor_weight_fn)
        Ds["saved_trc"] = Ds["train_count"]
        self.logger.info("init_weights ready at {0} start worker weight update".format(last_saved_weights_fnwp))
        self.D_share["weight_fnwp"]=last_saved_weights_fnwp
        self.E_update_weight.set()
        while self.E_update_weight.is_set():
            time.sleep(1)
        self.logger.info("finish worker init weight update and start worker work")

    def weight_update(self,i_brain, Ds):
        last_saved_weights_fnwp = self.save_AIO_model_weight_config(i_brain, Ds["train_count"])
        Ds["saved_trc"] = Ds["train_count"]

        self.logger.info("train_count {0} weights ready at {1} start worker weight update".format(Ds["train_count"],
                                                                                            last_saved_weights_fnwp))
        self.D_share["weight_fnwp"]=last_saved_weights_fnwp
        self.E_update_weight.set()
        while self.E_update_weight.is_set():
            time.sleep(1)
        if self.lc.Flag_Delete_Train_Brain_Buffer_After_Weight_Update:
            num_record_cleaned=self.get_train_records(self.delete_list)
            self.logger.info("train_count {0} clean {1} records".format(Ds["train_count"],num_record_cleaned))
            i_brain.tb.reset_tb()  # also need to reset in the brain buffer train queque(tq)
        self.logger.info("train_count {0} finish worker weight update and start worker work".format(Ds["train_count"]))
        return

    def delete_list(self, input_list):
        del input_list[:]

    def train_brain(self, i_brain, Ds):
        accumulate_item_get = self.get_train_records(i_brain.train_push_many)
        Ds["received_count_sum"] += accumulate_item_get
        Ds["l_recieved_count"][int(accumulate_item_get * 1.0 / 100) if accumulate_item_get < 1000 else 10] += 1
        Ds["accumulate_item_get_this_train_count"]=accumulate_item_get
        numb_record_trained = i_brain.optimize()
        if self.lc.flag_record_state:
            i_brain.mc.rv.recorder_process([Ds["train_count"], Ds["saved_trc"], Ds["print_trc"]])
            i_brain.mc.rv.saver()
        optimized_Flag =True if numb_record_trained>0 else False
        return optimized_Flag

    def check_save_print(self,i_brain, Ds):
        flag_weight_ready = False
        #if Ds["train_count"] % self.lc.num_train_to_save_model == 0 and Ds["train_count"] != self.lc.start_train_count:  # this need to adjust config setting not same as log
        if Ds["train_count"] % self.lc.num_train_to_save_model == 0 and Ds["train_count"] != self.train_count_init:  # this need to adjust config setting not same as log
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
        for idx in range(self.lc.num_workers):
            if len(self.LL_input[idx])==0:
                continue
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
