import sys
sys.path.append(r"C:\\Users\\COMPUTER\\Desktop\\aihackathon\\hairgan_pipeline")
import numpy as np
import pylib as py
import tensorflow as tf
from tf_lib import Checkpoint

import data
from model import module


def Model_Run(experiment_dir, input_img):

    args = py.args_from_yaml(py.join(experiment_dir, 'settings.yml'))

    # data
    # A_img_paths_test = py.glob(f"../image/{usernum}/rawhair", '*.jpg')
    # B_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testB'), '*.jpg')
    #make_dataset을 손봐야 함 (배치성 데이터로 구성되어있음)
    A_dataset_test = data.make_dataset(input_img, args.batch_size, args.load_size, args.crop_size,
                                    training=False, drop_remainder=False, shuffle=False, repeat=1)

    # model
    G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))
    G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))

    # resotre
    Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(args.experiment_dir, 'checkpoints')).restore()


    @tf.function
    def sample_A2B(A):
        A2B = G_A2B(A, training=False)
        A2B2A = G_B2A(A2B, training=False)
        return A2B, A2B2A


    # run
    i = 0
    for A in A_dataset_test:
        A2B, A2B2A = sample_A2B(A)
        for A_i, A2B_i, A2B2A_i in zip(A, A2B, A2B2A):
            img = np.concatenate([A2B_i.numpy()], axis=1)
            # im.imwrite(img, py.join(save_dir, py.name_ext(A_img_paths_test[i])))
            i += 1

    return img