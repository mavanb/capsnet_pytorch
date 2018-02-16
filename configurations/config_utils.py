import logging


def parse(parser):
    try:
        config = parser.parse_args()
        return config
    except:
        print(parser.format_help())
        raise ValueError("Could not parse config.")


def get_logger(log_file):
    if log_file is not None:
        logging.basicConfig(filename=log_file, level=logging.INFO)
        logger = logging.getLogger()
        logger.addHandler(logging.StreamHandler())
        logger = logger.info
    else:
        logger = print
    return logger