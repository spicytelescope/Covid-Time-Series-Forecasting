import logging

class CustomFormatter(logging.Formatter):

    """
    CustomFormatter for the logging module !
    """

    grey = "\x1b[38;20m"
    green = '\033[92m'
    yellow = "\x1b[33;20m"
    blue = "\x1b[34;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    FORMAT: str = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: blue + FORMAT + reset,
        logging.INFO: green + FORMAT + reset,
        logging.WARNING: yellow + FORMAT + reset,
        logging.ERROR: red + FORMAT + reset,
        logging.CRITICAL: bold_red + FORMAT + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger: logging.Logger = logging.getLogger("My_app")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())

logger.addHandler(ch)