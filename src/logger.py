# Any execution that happens we log it using this logger
# This logger is used to log the execution of the code
import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d')}.log" #LOG_FILE is the name of the log file, which is created with the current date
logs_path=os.path.join(os.getcwd(), 'logs',LOG_FILE)  #logs_path is the path where the log file is stored
os.makedirs(logs_path, exist_ok=True) #os.makedirs() method creates the directory if it does not exist

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE) #LOG_FILE_PATH is the path of the log file

logging.basicConfig(
    filename=LOG_FILE_PATH, #filename is the name of the log file
    format= '[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s', #format is the format of the log message
    level= logging.INFO #level is the logging level
)

'''
if __name__ == "__main__":
    logging.info("Logging is working") #logging.info() method logs the message with the INFO level
'''
