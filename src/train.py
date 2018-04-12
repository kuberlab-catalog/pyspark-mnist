import argparse
import logging
import os

from bigdl.dataset import mnist
from bigdl.dataset import transformer
from bigdl.nn import criterion
from bigdl.nn import layer
from bigdl.optim import optimizer
from bigdl.util import common
import pyspark


logging.basicConfig(
    format='%(asctime)s %(levelname)-10s %(name)-25s [-] %(message)s',
    level='INFO'
)
logging.root.setLevel(logging.INFO)
LOG = logging.getLogger('train')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--master',
        default='local[*]',
        help='URI to spark master.'
    )
    parser.add_argument(
        '--batch-size',
        default=20,
        type=int,
        help='Batch size (number of simultaneously processed images).'
    )
    parser.add_argument(
        '--executor-cores',
        '-c',
        default=4,
        type=int,
        help='Number of executor cores to use.'
    )
    parser.add_argument(
        '--output-dir',
        help='Trained model output dir.'
    )
    parser.add_argument(
        '--data-dir',
        default=os.environ.get('DATA_DIR'),
        help='Trained model output dir.'
    )
    return parser


def build_model(class_num):
    model = layer.Sequential()
    model.add(layer.Reshape([1, 28, 28]))
    model.add(layer.SpatialConvolution(1, 6, 5, 5))
    model.add(layer.Tanh())
    model.add(layer.SpatialMaxPooling(2, 2, 2, 2))
    model.add(layer.Tanh())
    model.add(layer.SpatialConvolution(6, 12, 5, 5))
    model.add(layer.SpatialMaxPooling(2, 2, 2, 2))
    model.add(layer.Reshape([12 * 4 * 4]))
    model.add(layer.Linear(12 * 4 * 4, 100))
    model.add(layer.Tanh())
    model.add(layer.Linear(100, class_num))
    model.add(layer.LogSoftMax())
    return model


def get_mnist(sc, data_type="train", location=os.environ.get('DATA_DIR')):
    """Get and normalize the mnist data.

    We would download it automatically
    if the data doesn't present at the specific location.
    :param sc: SparkContext
    :param data_type: training data or testing data
    :param location: Location storing the mnist
    :return: A RDD of (features: Ndarray, label: Ndarray)
    """
    (images, labels) = mnist.read_data_sets(location, data_type)
    images = sc.parallelize(images)
    labels = sc.parallelize(labels + 1)  # Target start from 1 in BigDL
    record = images.zip(labels)
    return record


def get_end_trigger():
    return optimizer.MaxEpoch(10)


def main():
    parser = get_parser()
    args = parser.parse_args()

    # BATCH_SIZE must be multiple of <executor.cores>:
    # in this case multiple of 3: 3,6,9,12 etc.
    if args.batch_size % args.executor_cores != 0:
        raise RuntimeError(
            'batch size must be multiple of <executor-cores> parameter!'
        )

    cores = args.executor_cores
    batch_size = args.batch_size
    conf = (
        common.create_spark_conf()
        .setAppName('pyspark-mnist')
        .setMaster(args.master)
    )
    conf = conf.set('spark.executor.cores', cores)
    conf = conf.set('spark.cores.max', cores)

    LOG.info('initialize with spark conf:')
    LOG.info(conf.getAll())
    sc = pyspark.SparkContext(conf=conf)
    common.init_engine()

    train_data = (
        get_mnist(sc, "train", location=args.data_dir)
        .map(lambda rec_tuple: (
            transformer.normalizer(
                rec_tuple[0], mnist.TRAIN_MEAN, mnist.TRAIN_STD
            ),
            rec_tuple[1])
        )
        .map(lambda t: common.Sample.from_ndarray(t[0], t[1]))
    )
    test_data = (
        get_mnist(sc, "test", location=args.data_dir)
        .map(lambda rec_tuple: (
            transformer.normalizer(
                rec_tuple[0], mnist.TEST_MEAN, mnist.TEST_STD
            ),
            rec_tuple[1])
        )
        .map(lambda t: common.Sample.from_ndarray(t[0], t[1]))
    )

    LOG.info(train_data.count())
    LOG.info(test_data.count())

    opt = optimizer.Optimizer(
        model=build_model(10),
        training_rdd=train_data,
        criterion=criterion.ClassNLLCriterion(),
        optim_method=optimizer.SGD(
            learningrate=0.01, learningrate_decay=0.0002
        ),
        end_trigger=get_end_trigger(),
        batch_size=batch_size
    )
    opt.set_validation(
        batch_size=batch_size,
        val_rdd=test_data,
        trigger=optimizer.EveryEpoch(),
        val_method=[optimizer.Top1Accuracy()]
    )
    trained_model = opt.optimize()
    LOG.info("training finished")

    results = trained_model.evaluate(
        test_data, batch_size * 2, [optimizer.Top1Accuracy()]
    )
    for result in results:
        LOG.info(result)

    LOG.info('saving model...')
    path = args.output_dir
    trained_model.saveModel(
        path + '/model.pb',
        path + '/model.bin',
        over_write=True
    )
    LOG.info('successfully saved!')

    sc.stop()


if __name__ == '__main__':
    main()
