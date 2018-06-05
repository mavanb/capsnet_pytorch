import configargparse
import torch.cuda
import os

def parse_bool(v):
    """ Bool type to set in add in parser.add_argument to fix not parsing of False. See:
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    import configargparse
    if v.lower().strip() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower().strip() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


def get_conf(custom_args=lambda x: x):
    import configargparse
    import configurations.general_confs

    parser = configargparse.get_argument_parser()

    # set module config
    parser = custom_args(parser)

    try:
        conf = parser.parse_args()
    except:
        print(parser.format_help())
        raise ValueError("Could not parse config.")

    # combined configs
    conf.model_checkpoint_path = "{}/{}{}".format(conf.trained_model_path, conf.model_name,
                                                  "_debug" if conf.debug else "")
    conf.model_load_path = "{}/{}".format(conf.trained_model_path, conf.load_name)

    # if use visdom, save config to visdom path to easily find config with environment
    if conf.use_visdom:

        if not os.path.exists(conf.visdom_path):
            os.makedirs(conf.visdom_path)

        with open(f"{conf.visdom_path}/{conf.model_name}.txt", "w") as f:
            f.write(parser.format_values())

    return conf, parser


p = configargparse.get_argument_parser()


# add configurations file
if torch.cuda.is_available():
    p.add('--general_config', is_config_file=True, default="./configurations/general_cuda.conf",
          help='configurations file path')
else:
    # config file for local / non-cuda run
    p.add('--general_config', is_config_file=True, default="./configurations/general_local.conf",
          help='configurations file path')

# required arguments: specified in configurations file or in
p.add_argument('--trained_model_path', required=True, type=str, help='Path of checkpoints.')
p.add_argument('--batch_size', type=int, required=True, help='Batch size.')
p.add_argument('--epochs', type=int, required=True, help='Number of epochs')
p.add_argument('--seed', type=int, required=True, help="Torch and numpy random seed. To ensure repeatability.")
p.add_argument('--save_trained', type=parse_bool, required=True, help='Save fully trained model for inference.')
p.add_argument('--debug', type=parse_bool, required=True, help="debug mode: break early")
p.add_argument('--print_time', type=parse_bool, required=True, help="print train time per sample")
p.add_argument('--load_name', type=str, required=True, help="Name of the model to load")
p.add_argument('--load_model', type=parse_bool, required=True, help="Load model yes/no")
p.add_argument("--log_file_name", type=str, required=True, help="log file to log output to")
p.add_argument("--log_file", type=parse_bool, required=True, help="log file to log output to")
p.add_argument("--drop_last", type=parse_bool, required=True, help="drop last incomplete batch")
p.add_argument('--shuffle', type=parse_bool, required=True, help='Shuffle dataset')
p.add_argument('--n_saved', type=int, required=True, help='Models are save every epoch. N_saved is length of this '
                                                          'history')
p.add_argument('--early_stop', type=parse_bool, required=True, help='Early stopping on validation loss')
p.add_argument('--cudnn_benchmark', type=parse_bool, required=True, help='Bool for cudnn benchmarking. Faster for large')
p.add_argument('--use_visdom', type=parse_bool, required=True, help='Makes plot in visdom yes/no. Slows startup time.')
p.add_argument('--start_visdom', type=parse_bool, required=True, help='Automatically start visdom if not running yes/no.')
p.add_argument('--visdom_path', type=str, required=True, help='Path where visdom envs are saved if automatic start')



