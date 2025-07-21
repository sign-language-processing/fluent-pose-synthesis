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
    parser.add_argument('--diffusion_steps', type=int, default=8, help='Number of diffusion steps.')
    parser.add_argument('--sigma_small', action='store_true', help='Use small sigma values.')


def add_train_args(parser):
    parser.add_argument('--epoch', type=int, default=500, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate.')
    parser.add_argument('--lr_anneal_steps', type=int, default=0, help='Annealing steps.')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--cond_mask_prob', type=float, default=0.15, help='Conditioning mask probability.')
    parser.add_argument('--workers', type=int, default=4, help='Data loader workers.')
    parser.add_argument('--ema', default=False, type=bool,
                        help='Use Exponential Moving Average (EMA) for model parameters.')
    parser.add_argument('--lambda_vel', type=float, default=0.0, help='Weight factor for the velocity loss term.')
    parser.add_argument('--use_loss_vel', action='store_true', default=True, help='Enable velocity loss term.')
    parser.add_argument('--use_loss_accel', action='store_true', default=False, help='Enable acceleration loss term.')
    parser.add_argument('--lambda_accel', type=float, default=1.0, help='Weight factor for the acceleration loss term.')
    parser.add_argument('--guidance_scale', type=float, default=2.0,
                        help='Classifier-free guidance scale for inference.')
    parser.add_argument('--load_num', type=int, default=-1, help='Number of models to load.')
    parser.add_argument('--use_amp', action='store_true', default=False, help='Use mixed precision training (AMP).')
    parser.add_argument('--eval_freq', type=int, default=1, help='Frequency of evaluation during training.')


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
    config.trainer.ema = True  #if args.ema else config.trainer.ema
    config.trainer.cond_mask_prob = args.cond_mask_prob
    config.trainer.workers = args.workers
    config.trainer.save_freq = int(config.trainer.epoch // 100)
    config.trainer.lambda_vel = args.lambda_vel
    config.trainer.use_loss_vel = args.use_loss_vel
    config.trainer.use_loss_accel = args.use_loss_accel
    config.trainer.lambda_accel = args.lambda_accel
    config.trainer.guidance_scale = args.guidance_scale
    config.trainer.load_num = args.load_num

    # Save directory
    data_prefix = args.data.split('/')[-1].split('.')[0]
    config.save = f'{args.save}/{args.name}_{data_prefix}' if 'debug' not in config.name else f'{args.save}/{args.name}'

    return config
