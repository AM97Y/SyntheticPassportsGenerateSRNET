import json
import os
import random
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pygame
import tensorflow as tf
from pathlib import Path
from PIL import Image
from pygame import freetype
from tqdm import tqdm

from Passport import PassportContent
from SRNet.model import SRNet
from utils.path_utils import Paths
from SRNetDataGenPassportRF.Synthtext import render_standard_text


IMAGE_FORMATS = [
    '.jpg',
    '.jpeg',
    '.png',
    '.tiff',
    '.bmp',
]
FONT_NAME = 'arial.ttf'


def gen_i_t(text: str, shape: tuple) -> np:
    """
    Generate i_t

    """
    pygame.init()
    font = freetype.Font(FONT_NAME)
    font.antialiased = True
    font.origin = True
    font.size = random.choice(range(14, 22))

    return render_standard_text.make_standard_text(FONT_NAME, text, shape)


def gen_srnet_img(i_s: np, i_t: np) -> Image:
    """
    Predict image

    """
    model = SRNet(shape=[64, None], name='predict')

    with model.graph.as_default():
        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, args.model)
            o_f = model.predict(sess, i_t, i_s)[-1]

    return Image.fromarray(o_f)


def generate_path(input: str, output: str) -> None:
    """
    Generate passport images

    """

    for file in tqdm(os.listdir(input)):
        if Path(file).suffix in IMAGE_FORMATS:
            json_file = Path(input) / Path(file).with_suffix('.json')
            with open(json_file, "r") as write_file:
                markup = json.load(write_file)

            img = Image.open(input + file).convert('RGB')

            passport_content = PassportContent()
            passport_content.random_init()

            for label_elem in markup['shapes']:
                if not label_elem['label'] in ['officer_signature', 'signature', 'photo', 'passport']:
                    text = passport_content.get(label_elem['label'])

                    i_s = img.crop(tuple(
                        [int(label_elem['points'][0][0]),
                         int(label_elem['points'][0][1]),
                         int(label_elem['points'][2][0]),
                         int(label_elem['points'][2][1])]))
                    if label_elem['label'] in ['number_group1', 'number_group2']:
                        i_s = i_s.rotate(90, expand=True)
                        shape = i_s.size
                    else:
                        shape = (i_s.size[1], i_s.size[0])
                    i_t = gen_i_t(text, shape)
                    i_s = np.array(i_s)
                    i_t = np.resize(i_t, i_s.shape).astype(np.float32) / 127.5 - 1.0
                    i_s = np.resize(i_s, i_s.shape).astype(np.float32) / 127.5 - 1.0

                    o_f = gen_srnet_img(i_s, i_t)
                    if label_elem['label'] in ['number_group1', 'number_group2']:
                        o_f = o_f.rotate(270, expand=True)
                    img.paste(o_f, (int(label_elem['points'][0][0]), int(label_elem['points'][0][1])))

            img_filepath = Paths.outputs(output) / f'{datetime.now().strftime("%Y-%m-%d-%H.%M.%S.%f")}.png'
            img.save(str(img_filepath))
            del img


def init_argparse():
    """
    Initializes argparse
    Returns parser.
    """
    parser = ArgumentParser(description='Transform text in original images of RF passports by means of SRNet model')
    parser.add_argument(
        '--input_path',
        '-i',
        nargs='?',
        help='Path with original passports to transform',
        default='./passports/',
        type=str)
    parser.add_argument(
        '--output_path',
        '-o',
        nargs='?',
        help='Path to save synthetic images',
        default='./synthetic_passports/',
        type=str)
    parser.add_argument(
        '--model',
        '-m',
        nargs='?',
        help='Path to checkpoint to SRNet model',
        default='./model/iter-50000',
        type=str)
    parser.add_argument(
        '--gpu',
        nargs='?',
        help='GPU for model.',
        default=-1,
        type=int)
    return parser


if __name__ == "__main__":
    args = init_argparse().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    generate_path(args.input_path, args.output_path)
