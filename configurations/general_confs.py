import configargparse
import torch.cuda


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


# optional arguments
p.add_argument('--seed', type=int, default=None, help="Torch and numpy random seed. To ensure repeatability.")
p.add_argument('--save_trained', type=bool, default=False, help='Save fully trained model for inference.')
p.add_argument('--debug', type=bool, default=False, help="debug mode: break early")
p.add_argument('--print_time', type=bool, default=False, help="print train time per sample")
p.add_argument('--load', type=str, default=None, help="Name of the model to load")
p.add_argument('--log_interval', type=int, default=10,
                    help='how many batches to wait before logging training status')
p.add_argument("--log_file", type=str, default=None, help="log file to log output to")
p.add_argument("--drop_last", type=bool, default=False, help="drop last incomplete batch")
p.add_argument('--shuffle', type=bool, default=False, help='Shuffle dataset')


