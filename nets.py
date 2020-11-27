#from tensorflow import keras
import os
import tensorflow as tf
from tensorflow import keras
from File_comm import check_model_save_finish_write
from Buffer_comm import brain_buffer, brain_buffer_reuse
from action_comm import actionOBOS
from nets_trainer_LHPP2V2 import *
from nets_trainer_LHPP2V3 import *

def init_virtual_GPU(memory_limit):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
    except RuntimeError as e:
        assert False, e
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    return logical_gpus[0]



class Train_Brain:
    def __init__(self, lc, GPU_per_program, load_fnwps,train_count_init):
        keras.backend.set_learning_phase(1)  # add by john for error solved by
        self.lc=lc
        self.mc=globals()[lc.CLN_trainer](lc)
        #self.tb = train_buffer(lc.Buffer_nb_Features)
        #self.tb = globals()[lc.CLN_brain_buffer](lc.Buffer_nb_Features)
        self.tb = globals()[lc.CLN_brain_buffer](lc)
        self.train_push_many = self.tb.train_push_many
        self.get_buffer_size = self.tb.get_buffer_size
        if len(load_fnwps) == 0:
            self.Tmodel, self.Pmodel = self.build_model()
        else:
            assert len(load_fnwps) == 3
            self.Tmodel, self.Pmodel = self.load_model(load_fnwps)
        self.i_wait = check_model_save_finish_write(time_out=600)

        self.tensorboard = keras.callbacks.TensorBoard(
                log_dir=lc.tensorboard_dir,
                histogram_freq=0,
                write_graph=True
        )
        self.tensorboard.set_model(self.Tmodel)
        self.tensorboard_batch_id = train_count_init

    def build_model(self):
        return self.mc.build_train_model(name="T")

    def save_model(self, fnwps):
        model_AIO_fnwp, config_fnwp, weight_fnwp = fnwps

        self.Tmodel.save(model_AIO_fnwp, overwrite=True, include_optimizer=True,save_format="h5")
        if not os.path.exists(config_fnwp):
            model_json = self.Pmodel.to_json()
            with open(config_fnwp, "w") as json_file:
                json_file.write(model_json)

        model_dn, fn = os.path.split(weight_fnwp)
        temp_fnwp = os.path.join(model_dn, "temp.h5")
        with open(temp_fnwp, 'a'):
            os.utime(temp_fnwp, None)
        self.i_wait.start_monitor(temp_fnwp)
        self.Pmodel.save_weights(temp_fnwp,save_format="h5")
        wait_result = self.i_wait.wait_till_finish()
        self.i_wait.stop_monitor(temp_fnwp)
        assert wait_result, "Fail to save {0} in 10 minuts(600 second)".format(weight_fnwp)
        os.rename(temp_fnwp, weight_fnwp)

    def load_model(self, fnwps):
        return self.mc.load_train_model(fnwps)

    def named_loss(self, model, loss):
        result = {}
        if len(model.metrics_names) == 1:
            result[model.metrics_names[0]] = loss
        else:
            assert len(model.metrics_names) > 1
            for l in zip(model.metrics_names, loss):
                result[l[0]] = l[1]
        return result

    def optimize(self):
        num_record_to_train, loss_this_round = self.mc.optimize_com(self.tb, self.Pmodel, self.Tmodel)
        if num_record_to_train != 0:
            self.tensorboard.on_epoch_end(self.tensorboard_batch_id,
                                          self.named_loss(self.Tmodel, loss_this_round))
            self.tensorboard_batch_id += 1
            if self.lc.flag_record_state:
                self.mc.rv.recorder_brainer([self.Tmodel.metrics_names, loss_this_round])
        return num_record_to_train, loss_this_round


class Explore_Brain:
    def __init__(self,lc):
        self.mc = globals()[lc.system_type+"_Agent"](lc)
        self.mc.build_predict_model("P")
        self.choose_action=self.mc.choose_action
        if hasattr(self.mc,"choose_action_CC"):  #TODO find sinmpleway
            self.choose_action_CC = self.mc.choose_action_CC
        self.load_weight=self.mc.load_weight


