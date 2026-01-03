import logging
import os
from datetime import datetime


LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOGS_PATH = os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(LOGS_PATH, exist_ok=True)

LOGS_FILE_PATH = os.path.join(LOGS_PATH,LOG_FILE)

logging.basicConfig(
    filename=LOGS_FILE_PATH,
    format="[%(asctime)s-%(lineno)d-%(name)s-%(levelname)s-%(message)s]",
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)

