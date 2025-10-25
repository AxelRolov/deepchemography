import argparse


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # Model
    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--q_cell',
                           type=str, default='lstm', choices=['gru', 'lstm'],
                           help='Encoder rnn cell type')
    model_arg.add_argument('--q_bidir',
                           default=False, action='store_true',
                           help='If to add second direction to encoder')
    model_arg.add_argument('--q_d_h',
                           type=int, default=128,
                           help='Encoder h dimensionality (per direction)')
    model_arg.add_argument('--q_n_layers',
                           type=int, default=2,
                           help='Encoder number of layers')
    model_arg.add_argument('--q_dropout',
                           type=float, default=0.0,
                           help='Encoder layers dropout')
    model_arg.add_argument('--d_cell',
                           type=str, default='lstm', choices=['gru', 'lstm'],
                           help='Decoder rnn cell type')
    model_arg.add_argument('--d_n_layers',
                           type=int, default=2,
                           help='Decoder number of layers')
    model_arg.add_argument('--d_dropout',
                           type=float, default=0.0,
                           help='Decoder layers dropout')
    model_arg.add_argument('--d_z',
                           type=int, default=256,
                           help='Latent vector dimensionality (bottleneck)')
    model_arg.add_argument('--d_d_h',
                           type=int, default=256,
                           help='Decoder hidden dimensionality')
    model_arg.add_argument('--freeze_embeddings',
                           default=False, action='store_true',
                           help='If to freeze embeddings while training')
    model_arg.add_argument('--use_batch_norm',
                           default=False, action='store_true',
                           help='If to use batch normalization (crucial for high accuracy)')

    # Train
    train_arg = parser.add_argument_group('Train')
    train_arg.add_argument('--n_batch',
                           type=int, default=256,
                           help='Batch size (paper: 256)')
    train_arg.add_argument('--clip_grad',
                           type=float, default=5.0,
                           help='Clip gradients to this value (lower = more stable)')
    train_arg.add_argument('--lr_start',
                           type=float, default=0.001,
                           help='Initial learning rate (0.001 for stability, paper used 0.005)')
    train_arg.add_argument('--lr_patience',
                           type=int, default=3,
                           help='Epochs with no improvement before reducing LR (3 for stability, paper: 2)')
    train_arg.add_argument('--lr_factor',
                           type=float, default=0.5,
                           help='Factor by which to reduce learning rate')
    train_arg.add_argument('--lr_min',
                           type=float, default=1e-6,
                           help='Minimum learning rate')
    train_arg.add_argument('--early_stop_patience',
                           type=int, default=10,
                           help='Epochs with no improvement before stopping (10 for stability, paper: 5)')
    train_arg.add_argument('--early_stop_metric',
                           type=str, default='loss', choices=['loss', 'accuracy'],
                           help='Metric to use for early stopping (loss is faster)')
    train_arg.add_argument('--n_epochs',
                           type=int, default=100,
                           help='Maximum number of epochs')
    train_arg.add_argument('--val_accuracy_samples',
                           type=int, default=1000,
                           help='Number of samples to use for validation accuracy (0 = all, but slow!)')
    train_arg.add_argument('--n_last',
                           type=int, default=1000,
                           help='Number of iters to smooth loss calc')
    train_arg.add_argument('--n_jobs',
                           type=int, default=1,
                           help='Number of threads')
    train_arg.add_argument('--n_workers',
                           type=int, default=1,
                           help='Number of workers for DataLoaders')

    return parser


