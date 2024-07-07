from wisp.app_utils import default_log_setup, args_to_log_format
import wisp.config_parser as config_parser
from wisp.trainers import BaseTrainer, MultiviewTrainer
from wisp.datasets import MultiviewDataset, SampleRays
import argparse
import logging
import os
import numpy as np
import wisp

def is_interactive() -> bool:
    """ Returns True if interactive mode with gui is on, False is HEADLESS mode is forced """
    return os.environ.get('WISP_HEADLESS') != '1'

def parse_args():
    """Wisp mains define args per app.
    Args are collected by priority: cli args > config yaml > argparse defaults
    For convenience, args are divided into groups.
    """
    parser = argparse.ArgumentParser(description='A script for training simple NeRF variants.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str,
                        help='Path to config file to replace defaults.')
    parser.add_argument('--profile', action='store_true',
                        help='Enable NVTX profiling')

    log_group = parser.add_argument_group('logging')
    log_group.add_argument('--exp-name', type=str,
                           help='Experiment name, unique id for trainers, logs.')
    log_group.add_argument('--log-level', action='store', type=int, default=logging.INFO,
                           help='Logging level to use globally, DEBUG: 10, INFO: 20, WARN: 30, ERROR: 40.')
    log_group.add_argument('--perf', action='store_true', default=False,
                           help='Use high-level profiling for the trainer.')

    data_group = parser.add_argument_group('dataset')
    data_group.add_argument('--dataset-path', type=str,
                            help='Path to the dataset')
    data_group.add_argument('--dataset-num-workers', type=int, default=-1,
                            help='Number of workers for dataset preprocessing, if it supports multiprocessing. '
                                 '-1 indicates no multiprocessing.')
    data_group.add_argument('--dataloader-num-workers', type=int, default=0,
                            help='Number of workers for dataloader.')
    data_group.add_argument('--bg-color', default='black' if is_interactive() else 'white',
                            choices=['white', 'black'], help='Background color')
    data_group.add_argument('--multiview-dataset-format', default='standard', choices=['standard', 'rtmv'],
                            help='Data format for the transforms')
    data_group.add_argument('--num-rays-sampled-per-img', type=int, default='4096',
                            help='Number of rays to sample per image')
    data_group.add_argument('--mip', type=int, default=None,
                            help='MIP level of ground truth image')

    grid_group = parser.add_argument_group('grid')
    grid_group.add_argument('--grid-type', type=str, default='OctreeGrid',
                            choices=config_parser.list_modules('grid'),
                            help='Type of to use, i.e.:'
                                 '"OctreeGrid", "CodebookOctreeGrid", "TriplanarGrid", "HashGrid".'
                                 'Grids are located in `wisp.models.grids`')
    grid_group.add_argument('--interpolation-type', type=str, default='linear', choices=['linear', 'closest'],
                            help='Interpolation type to use for samples within grids.'
                                 'For a 3D grid structure, linear uses trilinear interpolation of 8 cell nodes,'
                                 'closest uses the nearest neighbor.')
    grid_group.add_argument('--blas-type', type=str, default='octree',  # TODO(operel)
                            choices=['octree',],
                            help='Type of acceleration structure to use for fast raymarch occupancy queries.')
    grid_group.add_argument('--multiscale-type', type=str, default='sum', choices=['sum', 'cat'],
                            help='Aggregation of choice for multi-level grids, for features from different LODs.')
    grid_group.add_argument('--feature-dim', type=int, default=32,
                            help='Dimensionality for features stored within the grid nodes.')
    grid_group.add_argument('--feature-std', type=float, default=0.0,
                            help='Grid initialization: standard deviation used for randomly sampling initial features.')
    grid_group.add_argument('--feature-bias', type=float, default=0.0,
                            help='Grid initialization: bias used for randomly sampling initial features.')
    grid_group.add_argument('--base-lod', type=int, default=2,
                            help='Number of levels in grid, which book-keep occupancy but not features.'
                                 'The total number of levels in a grid is `base_lod + num_lod - 1`')
    grid_group.add_argument('--num-lods', type=int, default=1,
                            help='Number of levels in grid, which store concrete features.')
    grid_group.add_argument('--codebook-bitwidth', type=int, default=8,
                            help='For Codebook and HashGrids only: determines the table size as 2**(bitwidth).')
    grid_group.add_argument('--tree-type', type=str, default='geometric', choices=['geometric', 'quad'],
                            help='For HashGrids only: how the resolution of the grid is determined. '
                                 '"geometric" uses the geometric sequence initialization from InstantNGP,'
                                 'where "quad" uses an octree sampling pattern.')
    grid_group.add_argument('--min-grid-res', type=int, default=16,
                            help='For HashGrids only: min grid resolution, used only in geometric initialization mode')
    grid_group.add_argument('--max-grid-res', type=int, default=2048,
                            help='For HashGrids only: max grid resolution, used only in geometric initialization mode')
    grid_group.add_argument('--prune-min-density', type=float, default=(0.01 * 512) / np.sqrt(3),
                            help='For HashGrids only: Minimum density value for pruning')
    grid_group.add_argument('--prune-density-decay', type=float, default=0.6,
                            help='For HashGrids only: The decay applied on the density every pruning')
    grid_group.add_argument('--blas-level', type=float, default=7,
                            help='For HashGrids only: Determines the number of levels in the acceleration structure '
                                 'used to track the occupancy status (bottom level acceleration structure).')

    nef_group = parser.add_argument_group('nef')
    nef_group.add_argument('--pos-embedder', type=str, choices=['none', 'identity', 'positional'],
                           default='positional',
                           help='MLP Decoder of neural field: Positional embedder used to encode input coordinates'
                                'or view directions.')
    nef_group.add_argument('--view-embedder', type=str, choices=['none', 'identity', 'positional'],
                           default='positional',
                           help='MLP Decoder of neural field: Positional embedder used to encode view direction')
    nef_group.add_argument('--position-input', type=bool, default=False,
                           help='If True, position coords will be concatenated to the '
                                'features / positional embeddings when fed into the decoder.')
    nef_group.add_argument('--pos-multires', type=int, default=10,
                           help='MLP Decoder of neural field: Number of frequencies to use for positional encoding'
                                'of input coordinates')
    nef_group.add_argument('--view-multires', type=int, default=4,
                           help='MLP Decoder of neural field: Number of frequencies to use for positional encoding'
                                'of view direction')
    nef_group.add_argument('--layer-type', type=str, default='none',
                           choices=['none', 'spectral_norm', 'frobenius_norm', 'l_1_norm', 'l_inf_norm'])
    nef_group.add_argument('--activation-type', type=str, default='relu',
                           choices=['relu', 'sin'])
    nef_group.add_argument('--hidden-dim', type=int, help='MLP Decoder of neural field: width of all hidden layers.')
    nef_group.add_argument('--num-layers', type=int, help='MLP Decoder of neural field: number of hidden layers.')

    tracer_group = parser.add_argument_group('tracer')
    tracer_group.add_argument('--raymarch-type', type=str, choices=['ray', 'voxel'], default='ray',
                              help='Marching algorithm to use when generating samples along rays in tracers.'
                                   '`ray` samples fixed amount of randomized `num_steps` along the ray.'
                                   '`voxel` samples `num_steps` samples in each cell the ray intersects.')
    tracer_group.add_argument('--num-steps', type=int, default=1024,
                              help='Number of samples to generate along traced rays. See --raymarch-type for '
                                   'algorithm used to generate the samples.')

    trainer_group = parser.add_argument_group('trainer')
    trainer_group.add_argument('--epochs', type=int, default=250,
                               help='Number of epochs to run the training.')
    trainer_group.add_argument('--batch-size', type=int, default=512,
                               help='Batch size for the training.')
    trainer_group.add_argument('--resample', action='store_true',
                               help='Resample the dataset after every epoch.')
    trainer_group.add_argument('--only-last', action='store_true',
                               help='Train only last LOD.')
    trainer_group.add_argument('--resample-every', type=int, default=1,
                               help='Resample every N epochs')
    trainer_group.add_argument('--model-format', type=str, default='full', choices=['full', 'state_dict'],
                               help='Format in which to save models.')
    trainer_group.add_argument('--pretrained', type=str,
                               help='Path to pretrained model weights.')
    trainer_group.add_argument('--save-as-new', action='store_true',
                               help='Save the model at every epoch (no overwrite).')
    trainer_group.add_argument('--save-every', type=int, default=(-1 if is_interactive() else 5),
                               help='Save the model at every N epoch.')
    trainer_group.add_argument('--render-tb-every', type=int, default=(-1 if is_interactive() else 5),
                               help='Render every N epochs')
    trainer_group.add_argument('--log-tb-every', type=int, default=5,  # TODO (operel): move to logging
                               help='Render to tensorboard every N epochs')
    trainer_group.add_argument('--log-dir', type=str, default='_results/logs/runs/',
                               help='Log file directory for checkpoints.')
    trainer_group.add_argument('--prune-every', type=int, default=-1,
                               help='Prune every N epochs')
    trainer_group.add_argument('--grow-every', type=int, default=-1,
                               help='Grow network every X epochs')
    trainer_group.add_argument('--growth-strategy', type=str, default='increase',
                               choices=['onebyone',      # One by one trains one level at a time.
                                        'increase',      # Increase starts from [0] and ends up at [0,...,N]
                                        'shrink',        # Shrink strats from [0,...,N] and ends up at [N]
                                        'finetocoarse',  # Fine to coarse starts from [N] and ends up at [0,...,N]
                                        'onlylast'],     # Only last starts and ends at [N]
                               help='Strategy for coarse-to-fine training')
    trainer_group.add_argument('--valid-only', action='store_true',
                               help='Run validation only (and do not run training).')
    trainer_group.add_argument('--valid-every', type=int, default=-1,
                               help='Frequency of running validation.')
    trainer_group.add_argument('--random-lod', action='store_true',
                               help='Use random lods to train.')
    trainer_group.add_argument('--wandb-project', type=str, default=None,
                               help='Weights & Biases Project')
    trainer_group.add_argument('--wandb-run-name', type=str, default=None,
                               help='Weights & Biases Run Name')
    trainer_group.add_argument('--wandb-entity', type=str, default=None,
                               help='Weights & Biases Entity')
    trainer_group.add_argument('--wandb-viz-nerf-angles', type=int, default=20,
                               help='Number of Angles to visualize a scene on Weights & Biases. '
                                    'Set this to 0 to disable 360 degree visualizations.')
    trainer_group.add_argument('--wandb-viz-nerf-distance', type=int, default=3,
                               help='Distance to visualize Scene from on Weights & Biases')

    optimizer_group = parser.add_argument_group('optimizer')
    optimizer_group.add_argument('--optimizer-type', type=str, default='adam',
                                 choices=config_parser.list_modules('optim'),
                                 help='Optimizer to be used, includes optimizer modules available within `torch.optim` '
                                      'and fused optimizers from `apex`, if apex is installed.')
    optimizer_group.add_argument('--lr', type=float, default=0.001,
                                 help='Base optimizer learning rate.')
    optimizer_group.add_argument('--eps', type=float, default=1e-8,
                                 help='Eps value for numerical stability.')
    optimizer_group.add_argument('--weight-decay', type=float, default=0,
                                 help='Weight decay, applied only to decoder weights.')
    optimizer_group.add_argument('--grid-lr-weight', type=float, default=100.0,
                                 help='Relative learning rate weighting applied only for the grid parameters'
                                      '(e.g. parameters which contain "grid" in their name)')
    optimizer_group.add_argument('--rgb-loss', type=float, default=1.0,
                                 help='Weight of rgb loss')

    # Evaluation renderer (definitions do not affect interactive renderer)
    offline_renderer_group = parser.add_argument_group('renderer')
    offline_renderer_group.add_argument('--render-res', type=int, nargs=2, default=[512, 512],
                                        help='Width/height to render at.')
    offline_renderer_group.add_argument('--render-batch', type=int, default=0,
                                        help='Batch size (in number of rays) for batched rendering.')
    offline_renderer_group.add_argument('--camera-origin', type=float, nargs=3, default=[-2.8, 2.8, -2.8],
                                        help='Camera origin.')
    offline_renderer_group.add_argument('--camera-lookat', type=float, nargs=3, default=[0, 0, 0],
                                        help='Camera look-at/target point.')
    offline_renderer_group.add_argument('--camera-fov', type=float, default=30,
                                        help='Camera field of view (FOV).')
    offline_renderer_group.add_argument('--camera-proj', type=str, choices=['ortho', 'persp'], default='persp',
                                        help='Camera projection.')
    offline_renderer_group.add_argument('--camera-clamp', nargs=2, type=float, default=[0, 10],
                                        help='Camera clipping bounds.')

    # Parse CLI args & config files
    args = config_parser.parse_args(parser)

    # Also obtain args as grouped hierarchy, useful for, i.e., logging
    args_dict = config_parser.get_grouped_args(parser, args)
    return args, args_dict

def load_dataset(args) -> MultiviewDataset:
    """ Loads a multiview dataset comprising of pairs of images and calibrated cameras.
    The types of supported datasets are defined by multiview_dataset_format:
    'standard' - refers to the standard NeRF format popularized by Mildenhall et al. 2020,
                 including additions to the metadata format added by Muller et al. 2022.
    'rtmv' - refers to the dataset published by Tremblay et. al 2022,
            "RTMV: A Ray-Traced Multi-View Synthetic Dataset for Novel View Synthesis".
            This dataset includes depth information which allows for performance improving optimizations in some cases.
    """
    transform = SampleRays(num_samples=args.num_rays_sampled_per_img)
    train_dataset = wisp.datasets.load_multiview_dataset(dataset_path=args.dataset_path,
                                                         split='train',
                                                         mip=args.mip,
                                                         bg_color=args.bg_color,
                                                         dataset_num_workers=args.dataset_num_workers,
                                                         transform=transform)
    validation_dataset = None
    if args.valid_every > -1 or args.valid_only:
        validation_dataset = train_dataset.create_split(split='val', transform=None)
    return train_dataset, validation_dataset

def load_trainer(pipeline, train_dataset, validation_dataset, device, scene_state, args, args_dict) -> BaseTrainer:
    """ Loads the NeRF trainer.
    The trainer is responsible for managing the optimization life-cycles and can be operated in 2 modes:
    - Headless, which will run the train() function until all training steps are exhausted.
    - Interactive mode, which uses the gui. In this case, an OptimizationApp uses events to prompt the trainer to
      take training steps, while also taking care to render output to users (see: iterate()).
      In interactive mode, trainers can also share information with the app through the scene_state (WispState object).
    """
    # args.optimizer_type is the name of some optimizer class (from torch.optim or apex),
    # Wisp's config_parser is able to pick this app's args with corresponding names to the optimizer constructor args.
    # The actual construction of the optimizer instance happens within the trainer.
    optimizer_cls = config_parser.get_module(name=args.optimizer_type)
    optimizer_params = config_parser.get_args_for_function(args, optimizer_cls)

    trainer = MultiviewTrainer(pipeline=pipeline,
                               train_dataset=train_dataset,
                               validation_dataset=validation_dataset,
                               num_epochs=args.epochs,
                               batch_size=args.batch_size,
                               optim_cls=optimizer_cls,
                               lr=args.lr,
                               weight_decay=args.weight_decay,
                               grid_lr_weight=args.grid_lr_weight,
                               optim_params=optimizer_params,
                               log_dir=args.log_dir,
                               device=device,
                               exp_name=args.exp_name,
                               info=args_to_log_format(args_dict),
                               extra_args=vars(args),
                               render_tb_every=args.render_tb_every,
                               save_every=args.save_every,
                               scene_state=scene_state,
                               trainer_mode='validate' if args.valid_only else 'train',
                               using_wandb=args.wandb_project is not None)
    return trainer