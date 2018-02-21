import logging


def get_conf_logger(custom_args=lambda x: x):
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

    if conf.log_file is not None:
        logging.basicConfig(filename=conf.log_file, level=conf.logging.INFO)
        logger = conf.logging.getLogger()
        logger.addHandler(conf.logging.StreamHandler())
        logger = logger.info
    else:
        logger = print

    # combined configs
    conf.model_checkpoint_path = "{}/{}".format(conf.trained_model_path, conf.model_name)
    conf.model_load_path = "{}/{}".format(conf.trained_model_path, conf.load)

    # log configurations summary
    logger(parser.format_values())
    return conf, logger
