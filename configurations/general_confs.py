import configargparse
import torch.cuda
from configurations.config_utils import parse_bool


p = configargparse.get_argument_parser()

# add configurations file
if torch.cuda.is_available():
    p.add('--general_config', is_config_file=True, default="configurations/general_cuda.conf",
          help='configurations file path')
else:
    # config file for local / non-cuda run
    p.add('--general_config', is_config_file=True, default="configurations/general_local.conf",
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
p.add_argument('--log_interval', type=int, required=True,
                    help='how many batches to wait before logging training status')
p.add_argument("--log_file_name", type=str, required=True, help="log file to log output to")
p.add_argument("--log_file", type=parse_bool, required=True, help="log file to log output to")
p.add_argument("--drop_last", type=parse_bool, required=True, help="drop last incomplete batch")
p.add_argument('--shuffle', type=parse_bool, required=True, help='Shuffle dataset')
p.add_argument('--n_saved', type=int, required=True, help='Models are save every epoch. N_saved is length of this '
                                                          'history')
p.add_argument('--plot_train_progress', type=parse_bool, required=True, help='Plot train progress in visdom')
p.add_argument('--plot_eval_acc', type=parse_bool, required=True, help='Plot acc of validation in visdom')
p.add_argument('--early_stop', type=parse_bool, required=True, help='Early stopping on validation loss')





