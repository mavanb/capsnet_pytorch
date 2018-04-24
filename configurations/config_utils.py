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

    if conf.log_file:
        logging.basicConfig(filename=conf.log_file_name, level=conf.logging.INFO)
        logger = conf.logging.getLogger()
        logger.addHandler(conf.logging.StreamHandler())
        logger = logger.info
    else:
        logger = print

    # combined configs
    conf.model_checkpoint_path = "{}/{}{}".format(conf.trained_model_path, conf.model_name,
                                                  "_debug" if conf.debug else "")
    conf.model_load_path = "{}/{}".format(conf.trained_model_path, conf.load_name)

    # log configurations summary
    logger(parser.format_values())
    return conf, logger


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