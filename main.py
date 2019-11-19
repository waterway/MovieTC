"""
author: Mario Grabovaj
description: Text classification with preprocessed text
date: 19.11.2019
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow import keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import numpy as np


if __name__ == '__main__':
    print('Text classification with preprocessed text')
    print('Tensorflow: ', tf.__version__)