import configargparse

p = configargparse.get_argument_parser()

# add configurations file
p.add('--general_config', is_config_file=True, default="configurations/general.conf", help='configurations file path')

# required arguments: specified in configurations file or in
p.add_argument('--trained_model_path', required=True, type=str, help='Path of checkpoints.')
p.add_argument('--batch_size', type=int, required=True, help='Batch size.')
p.add_argument('--epochs', type=int, required=True, help='Number of epochs')

# optional arguments
p.add_argument('--seed', type=int, default=None, help="Torch and numpy random seed. To ensure repeatability.")
p.add_argument('--save_trained', type=bool, default=True, help='Save fully trained model for inference.')
p.add_argument('--debug', type=bool, default=False, help="debug mode: break early")
p.add_argument('--print_time', type=bool, default=False, help="print train time per sample")
p.add_argument('--load', type=bool, default=False, help="load previously trained model")
p.add_argument('--log_interval', type=int, default=10,
                    help='how many batches to wait before logging training status')
p.add_argument("--log_file", type=str, default=None, help="log file to log output to")



