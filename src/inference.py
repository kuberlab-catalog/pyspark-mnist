import argparse

from bigdl.nn import layer
from bigdl.util import common
import numpy as np
from PIL import Image
import pyspark

from six.moves import StringIO


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--master',
        default='local[*]',
        help='URI to spark master.'
    )
    parser.add_argument(
        '--input',
        help='input image path.'
    )
    parser.add_argument(
        '--executor-cores',
        '-c',
        default=4,
        type=int,
        help='Number of executor cores to use.'
    )
    parser.add_argument(
        '--model-dir',
        help='Trained model dir.'
    )
    return parser


def load_input(data):
    image = Image.open(StringIO(data))
    im_width, im_height = image.size
    return np.array(image.getdata()).reshape(
        (1, im_height, im_width)).astype(np.uint8)


def main():
    parser = get_parser()
    args = parser.parse_args()

    cores = args.executor_cores
    conf = (
        common.create_spark_conf()
        .setAppName('pyspark-mnist')
        .setMaster(args.master)
    )
    conf = conf.set('spark.executor.cores', cores)
    conf = conf.set('spark.cores.max', cores)

    print('initialize with spark conf:')
    print(conf.getAll())
    sc = pyspark.SparkContext(conf=conf)
    common.init_engine()

    model = layer.Model.loadModel(
        args.model_dir + "/model.pb",
        args.model_dir + "/model.bin"
    )
    images = sc.binaryFiles(args.input)
    # Load raw data into numpy arrays
    images = images.mapValues(load_input)

    print('image count: %s' % images.count())

    # TODO: how to do something like
    # result = model.predict(images.values())
    # print(result.collect())
    # ????

    for filename, image_data in images.collect():
        predict_result = model.predict(image_data)
        print('%s: %s' % (filename, predict_result[0].argmax()))

    sc.stop()


if __name__ == '__main__':
    main()