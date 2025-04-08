import copy
import json
from types import SimpleNamespace as Namespace


def add_model_args(parser):
    parser.add_argument('--decoder', type=str, default='trans_enc', help='Decoder type.')
    parser.add_argument('--latent_dim', type=int, default=256, help='Transformer/GRU latent dimension.')
    parser.add_argument('--ff_size', type=int, default=1024, help='Feed-forward size.')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of model layers.')

def add_diffusion_args(parser):
    parser.add_argument('--noise_schedule', type=str, default='cosine', help='Noise schedule: "cosine", "linear", etc.')
    parser.add_argument('--diffusion_steps', type=int, default=4, help='Number of diffusion steps.')
    parser.add_argument('--sigma_small', action='store_true', help='Use small sigma values.')

def add_train_args(parser):
    parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--lr_anneal_steps', type=int, default=0, help='Annealing steps.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size.')
    parser.add_argument('--cond_mask_prob', type=float, default=0.15, help='Conditioning mask probability.')
    parser.add_argument('--workers', type=int, default=4, help='Data loader workers.')
    parser.add_argument('--ema', default=False, type=bool, help='Use Exponential Moving Average (EMA) for model parameters.')


def config_parse(args):
    config = copy.deepcopy(json.load(open(args.config), object_hook=lambda d: Namespace(**d)))

    config.data = args.data
    config.name = args.name

    config.arch.decoder = args.decoder
    config.arch.latent_dim = args.latent_dim
    config.arch.ff_size = args.ff_size
    config.arch.num_heads = args.num_heads
    config.arch.num_layers = args.num_layers

    config.diff.noise_schedule = args.noise_schedule
    config.diff.diffusion_steps = args.diffusion_steps
    config.diff.sigma_small = args.sigma_small

    config.trainer.epoch = args.epoch
    config.trainer.lr = args.lr
    config.trainer.lr_anneal_steps = args.lr_anneal_steps
    config.trainer.weight_decay = args.weight_decay
    config.trainer.batch_size = args.batch_size
    config.trainer.ema = True #if args.ema else config.trainer.ema
    config.trainer.cond_mask_prob = args.cond_mask_prob
    config.trainer.workers = args.workers
    config.trainer.save_freq = int(config.trainer.epoch // 5)


    # Save directory
    data_prefix = args.data.split('/')[-1].split('.')[0]
    config.save = f'{args.save}/{args.name}_{data_prefix}' if 'debug' not in config.name else f'{args.save}/{args.name}'

    return config
