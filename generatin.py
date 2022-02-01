import json
import os
import random
from argparse import ArgumentParser
from datetime import datetime

import cv2
import numpy as np
import pygame
import tensorflow as tf
from PIL import Image
from pygame import freetype
from tqdm import tqdm

from Passport import PassportContent
import SRNet
from utils.path_utils import Paths
from SRNetDataGenPassportRF.Synthtext import data_cfg
from SRNetDataGenPassportRF.Synthtext import render_text_mask
from SRNetDataGenPassportRF.Synthtext import render_standard_text

image_formats = ['jpg',
                 'jpeg',
                 'png',
                 'tiff',
                 'bmp',
                 ]


def gen_i_t(text: str, shape: tuple) -> np:
    """
    Generate i_t.

    """
    pygame.init()
    name = 'arial.ttf'
    font = freetype.Font(name)
    font.antialiased = True
    font.origin = True
    font.size = random.choice(range(14, 22))

    i_t = render_standard_text.make_standard_text(name, text, shape)

    return i_t


def gen_srnet_img(i_s: np, i_t: np) -> Image:
    """
    Predict image.

    """
    model = SRNet.model.SRNet(shape=[64, None], name='predict')

    with model.graph.as_default():
        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, args.model)
            o_sk, o_t, o_b, o_f = model.predict(sess, i_t, i_s)

    return Image.fromarray(o_f)


def generate_path(input: str, output: str) -> None:
    """
    Generate passport images.

    """

    for file in tqdm(os.listdir(input)):
        type_file = file.split('.')[-1]
        if type_file in image_formats:
            json_file = input + ".".join(file.split('.')[:-1]) + '.json'
            with open(json_file, "r") as write_file:
                markup = json.load(write_file)

            img = Image.open(input + file)
            img = img.convert('RGB')

            passport_content = PassportContent()
            passport_content.random_init()

            for lable_elem in markup['shapes']:
                if not lable_elem['label'] in ['officer_signature', 'signature', 'photo', 'passport']:
                    text = passport_content.get(lable_elem['label'])

                    i_s = img.crop(tuple(
                        [int(lable_elem['points'][0][0]),
                         int(lable_elem['points'][0][1]),
                         int(lable_elem['points'][2][0]),
                         int(lable_elem['points'][2][1])]))
                    if lable_elem['label'] in ['number_group1', 'number_group2']:
                        i_s = i_s.rotate(90, expand=True)
                        shape = i_s.size
                    else:
                        shape = (i_s.size[1], i_s.size[0])
                    i_t = gen_i_t(text, shape)
                    i_s = np.array(i_s)
                    to_scale = i_s.shape
                    i_t = np.resize(i_t, to_scale)
                    i_s = np.resize(i_s, to_scale)
                    i_t = i_t.astype(np.float32) / 127.5 - 1.
                    i_s = i_s.astype(np.float32) / 127.5 - 1.

                    o_f = gen_srnet_img(i_s, i_t)
                    if lable_elem['label'] in ['number_group1', 'number_group2']:
                        o_f = o_f.rotate(270, expand=True)
                    img.paste(o_f, (int(lable_elem['points'][0][0]), int(lable_elem['points'][0][1])))

            img_filepath = Paths.outputs(output) / f'{datetime.now().strftime("%Y-%m-%d-%H.%M.%S.%f")}.png'
            img.save(str(img_filepath))
            del img


def init_argparse():
    """
    Initializes argparse
    Returns parser.
    """
    parser = ArgumentParser(description='Aug')

    parser.add_argument(
        '--output_path',
        nargs='?',
        help='Path to save files',
        default='./passports_gen/',
        type=str)
    parser.add_argument(
        '--input_path',
        nargs='?',
        help='Path with passports and json.',
        default='./passports/',
        type=str)
    parser.add_argument(
        '--model',
        nargs='?',
        help='Path with passports and json.',
        default='./model/iter-50000',
        type=str)

    parser.add_argument(
        '--gpu',
        nargs='?',
        help='Gpu for model.',
        default=0,
        type=int)

    return parser


if __name__ == "__main__":
    args = init_argparse().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    generate_path(args.input_path, args.output_path)
