import glob

import numpy
import tensorflow
from tqdm import tqdm

DEFAULT_IMAGE_WIDTH = 128
DEFAULT_IMAGE_HEIGHT = 128
DEFAULT_NUMBER_COLOR_CHANNELS = 3
DEFAULT_IMAGE_NORMALIZATION = 10
DEFAULT_DATASET_IMAGE_PATH = "Dataset"


class LoadImage:

    def __init__(self):

        self.image_width = DEFAULT_IMAGE_WIDTH
        self.image_height = DEFAULT_IMAGE_HEIGHT
        self.number_color_channels = DEFAULT_NUMBER_COLOR_CHANNELS
        self.dataset_path = DEFAULT_DATASET_IMAGE_PATH
        self.size_batch = 32
        self.image_list = []
        self.image_loaded = None
        pass

    def parse_image(self, filename):

        image = tensorflow.io.read_file(filename)
        image = tensorflow.image.decode_png(image, channels=self.number_color_channels)
        image = tensorflow.image.convert_image_dtype(image, tensorflow.float32)
        image = tensorflow.image.resize(image, [self.image_width, self.image_height])
        return numpy.array(numpy.asarray(image/127.5 - 1.0, dtype="float32"))


    def load_images(self):

        directory_images = glob.glob("{}/*".format(self.dataset_path))

        for i in tqdm(directory_images):
            self.image_list.append(self.parse_image(i))

        residual = len(self.image_list)%self.size_batch

        self.image_loaded = numpy.array(self.image_list[0:len(self.image_list) - residual])
        self.image_loaded = self.image_loaded + self.image_loaded

    def get_dataset_image(self):

        self.load_images()
        return numpy.array(self.image_loaded)


