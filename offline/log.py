import logging
import datetime

def setup_custom_logger(name, level):    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # to stdout
    s_handler = logging.StreamHandler()
    logger.addHandler(s_handler)
    
    # to file
    f_handler = logging.FileHandler("./logs/%s.log" % datetime.datetime.now().isoformat())
    logger.addHandler(f_handler)
    
    return logger