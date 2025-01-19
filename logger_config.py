import logging

def setup_logger():
    # Create a custom logger
    logger = logging.getLogger(__name__)

    # Set the default logging level
    logger.setLevel(logging.DEBUG)

    # Create a file handler that logs messages to a file
    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.DEBUG)

    # Create a stream handler to log to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Create a log format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)

    # Add the formatter to the handlers
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
