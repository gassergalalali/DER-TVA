"""
logger.py
logs to a log.log file
"""
import logging


loggers = {}

def get_logger(name: str = "log"):
    if name in loggers.keys():
        return loggers[name]
    else:
        filename = str(name) + ".log"
        logger = logging.Logger(name)
        logger.setLevel(logging.DEBUG)
        # File Handler
        fh = logging.FileHandler(filename, mode="w")
        fh.setLevel(logging.DEBUG)
        # Console Handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        # print(f"[logger.py]: Logger {filename} Created")
        loggers[name] = logger
        return logger
