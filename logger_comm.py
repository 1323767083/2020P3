import os, logging, time,shutil
from logging.handlers import RotatingFileHandler
from multiprocessing import log_to_stderr,get_logger
class lconfig:
    def __init__(self):
        self.log_dir=""

def init_gc(lgc):
    global lc
    lc=lconfig()
    for key in lc.__dict__.keys():
        lc.__dict__[key] = lgc.__dict__[key]


def setup_logger(label,  maxBytes=1000000, backupCount=10, flag_file_log=False, flag_screen_show=False):
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

def setup_tf_logger(label):
    # get TF logger
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.WARN)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler which logs even debug messages
    logger_fnwp = os.path.join(lc.log_dir, 'tf_{0}.log'.format(label))
    fh = logging.FileHandler(logger_fnwp)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)