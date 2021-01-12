import re
from nets_trainer import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
import tensorflow as tf
def comm_init_lgc(gc):
    global lgc
    lgc=gc

class fake_buffer:
    def __init__(self):
        self.flag_data_valid=False
    def add(self, buffer_to_add):
        self.flag_data_valid=True
        self.buffer=buffer_to_add
    def train_get(self, size_to_get):
        if self.flag_data_valid:
            self.flag_data_valid=False
            return [True] + self.buffer
        else:
            assert [False] + ["" for _ in range (10)]

class visual_trainer_brain:
    def __init__(self):
        with tf.device("/GPU:0"):
            #tf.reset_default_graph()
            #self.default_graph = tf.get_default_graph()
            #config = tf.ConfigProto(
            #    log_device_placement=False
            #)
            #config.gpu_options.allow_growth = True
            #config.gpu_options.per_process_gpu_memory_fraction = 0.3
            #self.session = tf.Session(config=config, graph=self.default_graph)
            #K.set_session(self.session)
            #K.set_learning_phase(1)  # add by john for error solved by
            self.tb=fake_buffer()
            self.mc = globals()[lgc.CLN_trainer]()

    '''
    def save_train_model(self, Tmodel, model_AIO_fnwp):
        with tf.device("/GPU:0"):
            with self.default_graph.as_default():
                with self.session.as_default():
                    Tmodel.save(model_AIO_fnwp, overwrite=True, include_optimizer=True)
    '''
    def load_model(self, model_AIO_fnwp, config_fnwp):
        with tf.device("/GPU:0"):
            fnwps=[model_AIO_fnwp, config_fnwp, ""]
            return self.mc.load_train_model(fnwps)

    def optimize(self, Tmodel, Pmodel,inputs):
        with tf.device("/GPU:0"):
            self.tb.add(inputs)
            num_record_to_train = self.mc.optimize_com(self.tb, Pmodel, Tmodel)
            return num_record_to_train

    def get_layer_wb(self, model, layer_name):
        return model.get_layer(name=layer_name).get_weights()


    def get_trainable_layer_list(self, model):
        with tf.device("/GPU:0"):
            l_layer_name = []
            l_layer_output_shape = []
            for layer in model.layers:
                if self._check_layer_type(layer.name)!="Non_param_layer":
                    l_layer_name.append(layer.name)
                    l_layer_output_shape.append(layer.output_shape)
            return l_layer_name, l_layer_output_shape

    def _check_layer_type(self,layer_name):
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

    def get_trainable_layer_list_from_config_file(self, system_name):
        model_config_fnwp = os.path.join(lgc.brain_model_dir, "config.json")
        load_jason_custom_objects = {"softmax": keras.backend.softmax, "tf": tf, "concatenate": keras.backend.concatenate,"lc":lgc}
        with open(model_config_fnwp, 'r') as json_file:
            loaded_model_json = json_file.read()
        Pmodel = keras.models.model_from_json(loaded_model_json, custom_objects=load_jason_custom_objects)
        l_layer_name = []
        l_layer_output_shape = []
        for layer in Pmodel.layers:
            if self._check_layer_type(layer.name) != "Non_param_layer":
                l_layer_name.append(layer.name)
                l_layer_output_shape.append(layer.output_shape)
        return l_layer_name, l_layer_output_shape


class show_op:
    def _wb_stats(self, wb):
        lpercent=[0,25,50,75,100]
        lpercentiles=[]
        for idx in range(2):
            lpercentile=[]
            for percent in [0,25,50,75,100]:
                value=np.percentile(wb[idx], percent)
                lpercentile.append(value)
            lpercentiles.append(lpercentile)
        return lpercentiles
    def _wb_hist_op(self, wb,axes,lpercentiles):
        num_step=[40, 20]
        for idx in range(2):
            ax=axes[idx]
            vmax = lpercentiles[idx][-1]
            vmin = lpercentiles[idx][0]
            vstep = (vmax- vmin)/num_step[idx]
            if vstep!=0:
                bins = np.arange(vmin, vmax + vstep, vstep)
                ax.hist(wb[idx].reshape((-1,)), bins=bins)
            else:
                ax.hist(wb[idx].reshape((-1,)))

    def _wb_imshow(self, fig,ax, wb):
        divider3 = make_axes_locatable(ax)
        cax = divider3.append_axes("right", size="5%", pad=0.05)
        im = ax.imshow(wb, aspect='auto')
        cax.tick_params(labelsize=8)
        cbar = fig.colorbar(im, cax=cax, format='%.0e')
        ax.tick_params(direction='out', labelsize=9)

    def _wbs_imshow(self, fig, wb):
        allaxes = fig.get_axes()
        for axe in allaxes:
            axe.remove()

        fig.subplots_adjust(bottom=0.08, top=0.92, left=0.08, right=0.92, wspace=0.14, hspace=0.14)
        if len(wb[0].shape) == 3:  # means 1cov
            num_plot = wb[0].shape[0]
            num_row = num_plot / 2 + 1
            base_chart_num = num_row * 100 + 2 * 10 + 1
            for idx in range(num_plot):
                ax = fig.add_subplot(base_chart_num + idx)
                ax.set_title("weight {0}".format(idx))
                self._wb_imshow(fig, ax, wb[0][idx])

            ax = fig.add_subplot(base_chart_num + num_plot)
            ax.plot(wb[1])
            ax.set_title("bias")
            ax.tick_params(direction='out', labelsize=9)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))

        else:
            assert len(wb[0].shape) == 2
            ax = fig.add_subplot(121)
            ax.set_title("weight")
            self._wb_imshow(fig, ax, wb[0])
            ax = fig.add_subplot(122)
            ax.plot(wb[0])
            ax.set_title("bias")
            ax.tick_params(direction='out', labelsize=9)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))

class plot3D:
    def plot3d_surface(self, fig, wb):
        x, y = np.meshgrid(list(range(wb[0][0].shape[1])), list(range(wb[0][0].shape[0])))
        z = wb[0][0] * 100
        from mpl_toolkits.mplot3d import Axes3D
        plt3d = plt.figure().gca(projection='3d')
        plt3d.plot_surface(x, y, z, alpha=0.2, label='parametric curve')
        plt3d.legend()
        plt.show()

    def plot3d_surface_sub(self, fig, z):
        #fig = plt.figure()
        #z = wb[0][0]
        x, y = np.meshgrid(list(range(z.shape[1])), list(range(z.shape[0])))

        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, alpha=0.2, label='parametric curve')
        #ax.plot_surface(x, y, z + 1, alpha=0.2, label='parametric curve')
        #plt.show()

    def plot3d_line_sub(self, fig, wb):
        # fig = plt.figure()
        xx, yy = np.meshgrid(list(range(wb[0][0].shape[0])), list(range(wb[0][0].shape[1])))
        x = xx.reshape((-1,))
        y = yy.reshape((-1,))
        z = wb[0][0].reshape((-1,))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, label='parametric curve')
        # ax.scatter(x, y, z, label='parametric curve')
        ax.legend()
        #plt.show()
