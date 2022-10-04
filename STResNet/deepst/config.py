from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVIDES"] = "/gpu:0"
import platform
import tensorflow as tf

config = tf.ConfigProto()

config.gpu_options.allow_growth=True

sess = tf.Session(config=config)
class Config(object):
    """docstring for Config"""

    def __init__(self):
        super(Config, self).__init__()

        DATAPATH = os.environ.get('DATAPATH')
        if DATAPATH is None:
            if platform.system() == "Windows" or platform.system() == "Linux":
                DATAPATH = "D:/Hierarchical-FCM-ResNets/STResNet/data"
            # elif platform.system() == "Linux":
            #     DATAPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
            else:
                print("Unsupported/Unknown OS: ", platform.system, "please set DATAPATH")
        self.DATAPATH = DATAPATH
