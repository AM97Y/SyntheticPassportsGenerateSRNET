"""
Based on projects SRNet  (https://github.com/youdao-ai/SRNet)
and  SRNet-Datagen(https://github.com/youdao-ai/SRNet-Datagen).
"""

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


def create_styled_text_on_gray(text: str, shape: tuple) -> np.ndarray:
    """
    Generate skeleton text on gray background (i_t from SRNet).
    text: new text for generate skeleton on gray background.
    shape: shape of new image.

    return: skeleton image (i_t from SRNet).
    """

    pygame.init()
    return render_standard_text.make_standard_text(FONT_NAME, text, shape)


def create_srnet_styled_text(changeable_image: np.ndarray, standard_text: np.ndarray) -> Image:
    """
    This function styles the text as changeable_image (styled text rendering on background image).

    changeable_image: original image.
    standard_text: new text skeleton on gray background.

    return: new styled image (o_f from SRNet).
    """
    model = SRNet(shape=[64, None], name='predict')

    with model.graph.as_default():
        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, args.model_path)
            styled_text = model.predict(sess, standard_text, changeable_image)[-1]

    return Image.fromarray(styled_text)


def create_styled_img(text: str, changeable_image: Image, shape: tuple) -> Image:
    """
    This function changes the image text(o_f from SRNet).
    text: new text of entity.
    changeable_image: cut out image.
    shape: shape of new img

    return: cutout image of the entity.
    """

    standard_text = create_styled_text_on_gray(text, shape)

    changeable_image = np.array(changeable_image)
    # normalize the image in the range from -1 to 1.
    standard_text = np.resize(standard_text, changeable_image.shape).astype(np.float32) / 127.5 - 1.0
    changeable_image = np.resize(changeable_image, changeable_image.shape).astype(np.float32) / 127.5 - 1.0

    # predict o_f
    styled_text = create_srnet_styled_text(changeable_image, standard_text)

    return styled_text


def generate_path(input_path: str, output_path: str) -> None:
    """
    Generate passport images.

    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    for file in tqdm(os.listdir(input_path)):
        if Path(file).suffix in IMAGE_FORMATS:
            json_file = Path(input_path) / Path(file).with_suffix('.json')
            with open(json_file, "r") as write_file:
                markup = json.load(write_file)

            img = Image.open(input_path + file).convert('RGB')

            passport_content = PassportContent()
            passport_content.random_init()

            for label_elem in markup['shapes']:
                if not label_elem['label'] in ['officer_signature', 'signature', 'photo', 'passport']:
                    x_0, y_0, x_1, y_1 = int(label_elem['points'][0][0]), int(label_elem['points'][0][1]), int(
                        label_elem['points'][2][0]), int(label_elem['points'][2][1])
                    # i_s from SRNet
                    changeable_image = img.crop(tuple([x_0, y_0, x_1, y_1]))
                    if label_elem['label'] in ['number_group1', 'number_group2']:
                        changeable_image = changeable_image.rotate(90, expand=True)
                        shape = changeable_image.size
                    else:
                        shape = (changeable_image.size[1], changeable_image.size[0])

                    text = passport_content.get(label_elem['label'])

                    # o_f from SRNet
                    styled_text = create_styled_img(text, changeable_image, shape)
                    if label_elem['label'] in ['number_group1', 'number_group2']:
                        styled_text = styled_text.rotate(270, expand=True)

                    img.paste(styled_text, (x_0, y_0))

            img.save(str(Paths.outputs(output_path) / f'{datetime.now().strftime("%Y-%m-%d-%H.%M.%S.%f")}.png'))
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
        help='Path with original passports to transform.',
        default='./passports/',
        type=str)
    parser.add_argument(
        '--output_path',
        '-o',
        nargs='?',
        help='Path to save synthetic images.',
        default='./synthetic_passports/',
        type=str)
    parser.add_argument(
        '--model_path',
        '-m',
        nargs='?',
        help='Path to checkpoint to SRNet model.',
        default='./model/iter-50000',
        type=str)
    parser.add_argument(
        '--gpu',
        nargs='?',
        help='GPU for model. If use cpu gpu is -1.',
        default=-1,
        type=int)
    return parser


if __name__ == "__main__":
    args = init_argparse().parse_args()
    if args.gpu != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    generate_path(args.input_path, args.output_path)
