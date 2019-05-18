import sys
import train
from argparse import ArgumentParser
from functools import partial

SUB = 'subparser'
TRAIN = 'train'
TEST = 'test'


def get_argparser():
    parser = ArgumentParser('Train or evaluate colorization CNN')

    _add_subparser(parser)

    _add_mutually_exclusive_bins_group(parser)

    parser.add_argument(
        '-l',
        '--log-dir',
        default='log',
        type=str,
        help='Path to the log files used for tensorboard visualization')

    parser.add_argument(
        '-p',
        '--pretrained-model-path',
        default='',
        help='Path to the pretrained model',
        type=str)

    parser.add_argument(
        '-b',
        '--batch-size',
        default=10,
        help='Batch size used for training',
        type=int)

    parser.add_argument(
        '-nw',
        '--num-workers',
        default=2,
        type=int,
        help=('Number of workers used during data loading'))

    return parser


def _add_mutually_exclusive_bins_group(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-bin',
        '--bin-path',
        help=('Path to the .bin data file, used to compute the prior log '
              'probability distribution of the colors in the training data'),
        type=str)
    group.add_argument(
        '-npz',
        '--npz-path',
        help=(
            'Path to the .npz file containing a dictionary of numpy arrays.'
            'Keys in the dictionary: a_bins, b_bins, ab_bins and w_bins.'
            'These numpy arrays are calculated using the function '
            'get_and_store_ab_bins_and_rarity_weights from lab_distribution.py'
        ),
        type=str)


def _add_subparser(parser):
    subparsers = parser.add_subparsers(dest=SUB)
    subparsers.required = True

    _add_train_subparser(subparsers)
    _add_test_subparser(subparsers)


def _add_train_subparser(subparsers):
    train = subparsers.add_parser(TRAIN, help='Train network')

    train.add_argument(
        '-t',
        '--train-dir',
        required=True,
        help='Path to directory with training data',
        type=str)

    train.add_argument(
        '-v',
        '--eval-dir',
        required=True,
        help='Path to directory with validation data',
        type=str)

    train.add_argument(
        '-en',
        '--eval-every-n',
        default=500,
        help='Evaluate network performance on validation data every n epochs',
        type=int)

    train.add_argument(
        '-e',
        '--num-epochs',
        default=3000,
        help='Number of epochs to train',
        type=int)

    train.add_argument(
        '-log-every-n',
        '--log-every-n',
        default=80,
        help='Save log info every n epochs',
        type=int)

    train.add_argument(
        '-cdir',
        '--checkpoint-dir',
        required=True,
        help='Path used to save model checkpoints',
        type=str)

    train.add_argument(
        '-c',
        '--checkpoint-every-n',
        default=400,
        help='Save model checkpoint every n epochs',
        type=int)

    train.add_argument(
        '-lr',
        '--learning-rate',
        default=.001,
        help='Learning rate used by Adam optimizer',
        type=float)

    train.add_argument(
        '-betas',
        '--betas',
        nargs=2,
        default=[.9, .999],
        help='Betas used by Adam optimizer',
        type=float)

    train.add_argument(
        '-eps',
        '--epsilon',
        default=1e-8,
        help='Epsilon used by Adam optimizer',
        type=float)

    train.add_argument(
        '-wd',
        '--weight-decay',
        default=.001,
        help='Weight decay used by Adam optimizer',
        type=float)


def _add_test_subparser(subparsers):
    test = subparsers.add_parser(TEST, help='Test a trained network')

    test.add_argument(
        '-t',
        '--test-dir',
        required=True,
        help='Path to directory with test data')


def _handle_train_parser(args):
    train_partial = partial(
        train.train,
        pretrained_model_path=args.pretrained_model_path,
        train_dir=args.train_dir,
        eval_dir=args.eval_dir,
        eval_every_n=args.eval_every_n,
        log_dir=args.log_dir,
        log_every_n=args.log_every_n,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every_n=args.checkpoint_every_n,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        betas=args.betas,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay)

    # npz bin path given
    if args.npz_path:
        train_partial(npz_path=args.npz_path)
    # bin path given, prior distributions can be computed
    else:
        train_partial(bin_path=args.bin_path)


def _handle_test_parser(args):
    if args.pretrained_model_path == '':
        raise ValueError('Please provide a pretrained model for evaluation!')

    test_partial = partial(
        train.test,
        pretrained_model_path=args.pretrained_model_path,
        test_dir=args.test_dir,
        log_dir=args.log_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    # npz bin path given
    if args.npz_path:
        test_partial(npz_path=args.npz_path)
    # bin path given, prior distributions can be computed
    else:
        test_partial(bin_path=args.bin_path)


def parse_args(sys_args):
    parser = get_argparser()
    args = parser.parse_args(sys_args)
    return args


def handle_parsed_args(args):
    if getattr(args, SUB) == TRAIN:
        _handle_train_parser(args)
    elif getattr(args, SUB) == TEST:
        _handle_test_parser(args)
    else:  # Impossible
        assert False
