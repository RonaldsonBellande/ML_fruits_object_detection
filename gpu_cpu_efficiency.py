from header_imports import *


def gpu_enable():
    '''
    Enable you to use multiple gpu for your model training 
    and being to train different models on a single gpu and spu at the same time 
    '''

    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    if info.free < 964157696:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    device_name = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
    if device_name != []:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])
        tensorflow_strategy = tf.distribute.MirroredStrategy(devices= ["/cpu:0", "/gpu:0"],cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        print("GPU GROWTH")
    else:
        device_name = "/device:CPU:0"
        tensorflow_strategy = tf.distribute.MirroredStrategy(["CPU:0"])
        print("CPU")


def gpu_disable():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def xla():
    '''
    Linear Algebra Compiler to improve the training speed

        Benifits: Does not have to write a lot of operations in memory, reduce memory bandwith
        --> Fast Optimizing Compiler
        --> Works with gpu and cpu
    '''

    tf.config.optimizer.set_jit(True)


def mixed_precision():
    '''
    Allows for 16-bit and 32-bit floating-point(fp32/fp16) to be changed, using the 16-bit floating-point 
    allows for faster the utilization of having for ram or vram avaliable to use and 
    reducing training time while giving the same output in preformence

        Benifits: Trains faster and use less memory
        --> preformence improve more than 3x
        --> Works with gpu and cpu

        Disadvantages: Accuracy from Underfitting and Overfitting
        --> With proper care there will be little to no accuracy lost, 
        when you make and train your model properly
    '''

    tf.keras.mixed_precision.set_global_policy('mixed_float16')




def ram_reset():
    '''
    Allows for running multiple instants of models since it will clear the global states in RAM

        Benifits: Multiple model run one after the other

        Disadvantages: None
    '''

    tf.keras.backend.clear_session()





