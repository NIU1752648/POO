import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('random_forest.log'),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)

logger = logging.getLogger('RandomForestLogger')

def log_callable(func):
    def wrapper(*args, **kwargs):
        logger.info(f"Calling function: {func.__name__} with args: {args}, kwargs: {kwargs}")

        result = func(*args, **kwargs)

        logger.info(f"Function {func.__name__} returned: {result}")
        return result
    return wrapper


def log_instance(cls):
    class Wrapper(cls):
        def __init__(self, *args, **kwargs):
            logger.info(f"Instantiating class: {cls.__name__} with args: {args}, kwargs: {kwargs}")

            super().__init__(*args, **kwargs)

    return Wrapper