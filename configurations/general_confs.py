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


class ArchLayer:
    def __init__(self, layer_str):
        l = [int(e) for e in layer_str.split(",")]
        assert len(l)==2, "Each layer should have two ints seperatated by a komma."
        self.caps = l[0]
        self.len = l[1]


class Architecture:
    def __init__(self, arch_str):

        arch = arch_str.split(";")
        assert 2 <= len(arch), "Architecture should have at least a primary and final layer."

        self.prim = ArchLayer(arch[0])
        self.final = ArchLayer(arch[-1])

        self.other_layers = []
        for i in arch[1:]:
            self.other_layers.append(ArchLayer(i))


def capsule_arguments(config_file_name, path_root="."):
    """ Adds all arguments used by a capsule network.
    """

    def custom_args(parser):
        parser.add(f'--config_file_{config_file_name}', is_config_file=True,
                   default=f"{path_root}/configurations/{config_file_name}.conf", help='configurations file path')
        parser.add_argument('--model_name', type=str, required=True, help='Name of the model.')
        parser.add_argument('--alpha', type=float, required=True, help="Alpha of CapsuleLoss")
        parser.add_argument('--m_plus', type=float, required=True, help="m_plus of margin loss")
        parser.add_argument('--m_min', type=float, required=True, help="m_min of margin loss")
        parser.add_argument('--routing_iters', type=int, required=True,
                            help="Number of iterations in the routing algo.")
        parser.add_argument('--dataset', type=str, required=True, help="Either mnist or cifar10")
        parser.add_argument('--stdev_W', type=float, required=True, help="stddev of W of capsule layer")
        parser.add_argument('--bias_routing', type=parse_bool, required=True, help="whether to use bias in routing")
        parser.add_argument('--excessive_testing', type=parse_bool, required=True,
                            help="Do excessive tests on tests set")
        parser.add_argument('--sparse_threshold', type=float, required=True, help="Threshold of routing to sparsify.")
        parser.add_argument('--sparsify', type=str, required=True, help="The method used to sparsify the parse tree.")
        parser.add_argument('--sparse_topk', type=str, required=True, help="Percentage of non top k elements to exclude.")
        parser.add_argument('--architecture', type=Architecture, required=True,
                            help="Architecture of the capsule network. Notation: Example: 32,8;10,16")

        return parser
    return custom_args


def get_conf(custom_args_list=[], path_root="."):
    """

    Args:
        custom_args: (list of callables) List of functions that take the parsers and first add the config file,
        next add all arguments in this config file.
        path_root: path root to main project

    Returns:
        conf: (Configuration) Object with all configurations
        parser: (Parser) Object with the used parser

    """
    import configargparse
    general_arguments(path_root)

    parser = configargparse.get_argument_parser()

    # loop over all custom arguments
    if not type(custom_args_list) == list:
        custom_args_list = [custom_args_list]

    for custom_args in custom_args_list:
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


def general_arguments(path_root):

    p = configargparse.get_argument_parser()

    # add configurations file
    if torch.cuda.is_available():
        p.add('--general_config', is_config_file=True, default=f"{path_root}/configurations/general.conf",
              help='configurations file path')
    else:
        # config file for local / non-cuda run
        p.add('--general_config', is_config_file=True, default=f"{path_root}/configurations/general.conf",
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
    p.add_argument('--learning_rate', type=float, required=True, help='Learning rate of optimizer')
    p.add_argument('--early_stop', type=parse_bool, required=True, help='Early stopping on validation loss')
    p.add_argument('--cudnn_benchmark', type=parse_bool, required=True, help='Bool for cudnn benchmarking. Faster for large')
    p.add_argument('--use_visdom', type=parse_bool, required=True, help='Makes plot in visdom yes/no. Slows startup time.')
    p.add_argument('--start_visdom', type=parse_bool, required=True, help='Automatically start visdom if not running yes/no.')
    p.add_argument('--visdom_path', type=str, required=True, help='Path where visdom envs are saved if automatic start')
    p.add_argument('--valid_size', type=float, required=True, help='Size of the validation set (between 0.0 and 1.0)')
    p.add_argument('--score_file_name', type=str, required=True, help='File name of the best scores over all epochs. Save must be True. ')
    p.add_argument('--save_best', type=str, required=True, help='Save best score yes/no.')

