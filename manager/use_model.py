import numpy as np
import pylib as py
import tensorflow as tf
from tf_lib import Checkpoint
import imlib

from .data import *
from model import module

class Generate_Model():

    def __init__(self, img_size, experiment_dir) -> None:
        self.img_size = img_size
        # model
        self.G_A2B = module.ResnetGenerator(input_shape=(img_size[0], img_size[1], img_size[2])) # 256,256,3
        self.G_B2A = module.ResnetGenerator(input_shape=(img_size[0], img_size[1], img_size[2]))

        # resotre
        Checkpoint(dict(G_A2B=self.G_A2B, G_B2A=self.G_B2A), py.join(experiment_dir, 'checkpoints')).restore()



    def Model_Run(self, batch_size, input_img_dirs):



        # 데이터 request를 배치성으로 받아 여러개를 한번에 처리하도록 수정
        A_dataset_test = make_dataset(input_img_dirs, batch_size, self.img_size[0], self.img_size[1],
                                        training=False, drop_remainder=False, shuffle=False, repeat=1)
        


        @tf.function
        def sample_A2B(A):
            A2B = self.G_A2B(A, training=False)
            A2B2A = self.G_B2A(A2B, training=False)
            return A2B, A2B2A
    

        # run
        for A in A_dataset_test:
            A2B, A2B2A = sample_A2B(A)
            for A_i, A2B_i, A2B2A_i in zip(A, A2B, A2B2A):
                img = imlib.im2uint(np.concatenate([A2B_i.numpy()], axis=1))
                # im.imwrite(img, py.join(save_dir, py.name_ext(A_img_paths_test[i])))
            

        return img