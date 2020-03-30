import tensorflow as tf

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


def test_virtual_GPU(memory_limit):
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

    #assert len(gpus)==1, "current A3C setup each process only get one GPU {0}".format(gpus)
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit),
             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])  #memory_limit=1024
        tf.config.experimental.set_virtual_device_configuration(
            gpus[1],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit),
             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])  #memory_limit=1024

    except RuntimeError as e:
        assert False, e
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    for logical_gpu in logical_gpus:
        with tf.device(logical_gpu):
            # Create some tensors
            print ("On {0}".format(logical_gpu))
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)


'''
2020-03-27 11:10:46.426752: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1241] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1024 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:02:00.0, compute capability: 7.5)
2020-03-27 11:10:46.427876: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1241] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 1024 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:02:00.0, compute capability: 7.5)
2020-03-27 11:10:46.429460: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1241] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:2 with 1024 MB memory) -> physical GPU (device: 1, name: TITAN Xp, pci bus id: 0000:82:00.0, compute capability: 6.1)
2020-03-27 11:10:46.430547: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1241] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:3 with 1024 MB memory) -> physical GPU (device: 1, name: TITAN Xp, pci bus id: 0000:82:00.0, compute capability: 6.1)
2020-03-27 11:10:46.432549: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ff6b3537f0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-03-27 11:10:46.432570: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): GeForce RTX 2080 Ti, Compute Capability 7.5
2020-03-27 11:10:46.432589: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (1): TITAN Xp, Compute Capability 6.1
On LogicalDevice(name='/device:GPU:0', device_type='GPU')
2020-03-27 11:10:46.468963: I tensorflow/core/common_runtime/eager/execute.cc:573] Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0
2020-03-27 11:10:46.469123: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
On LogicalDevice(name='/device:GPU:1', device_type='GPU')
2020-03-27 11:10:46.767183: I tensorflow/core/common_runtime/eager/execute.cc:573] Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:1
On LogicalDevice(name='/device:GPU:2', device_type='GPU')
2020-03-27 11:10:46.786433: I tensorflow/core/common_runtime/eager/execute.cc:573] Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:2
On LogicalDevice(name='/device:GPU:3', device_type='GPU')
2020-03-27 11:10:46.982446: I tensorflow/core/common_runtime/eager/execute.cc:573] Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:3
'''

'''
conclusion:
1. virtual device GPU:0,GPU:1 not one to one map to physical GPU:0
2. virtual device series 0, 1 following the order set_virtual_device_configuration put in.
3. the above print out shows 2 virual device on physical device 0 and another 2 virtual device on physical device 1.


'''
#test purpose
if __name__ == '__main__':
    print ("test purpose")
    init_virtual_GPU(1024)

