import os, logging, time,shutil
from logging.handlers import RotatingFileHandler
from multiprocessing import log_to_stderr,get_logger


def setup_logger(lc,label,  maxBytes=1000000, backupCount=10, flag_file_log=False, flag_screen_show=False):
    logger = logging.getLogger(label)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if flag_file_log:
        logger_fnwp=os.path.join(lc.log_dir,"{0}.log".format(label))
        fh = RotatingFileHandler(logger_fnwp, maxBytes=maxBytes, backupCount=backupCount)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if flag_screen_show:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    #logger.info("start")
    return logger
