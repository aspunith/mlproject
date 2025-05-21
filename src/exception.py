from src.logger import logging
import sys #sys library is used to get the exception information in the except block 

def error_message_detail(error, error_detail):
    _, _, exc_tb = error_detail.exc_info()  # exc_info() method returns the exception information, like on which line the exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename  # co_filename gives the file name where the exception occurred
    error_message = "Error occurred in python script name [{0}] at line number [{1}] with error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    
    
    return error_message


class CustomException(Exception): #CustomException class inherits the Exception class, which is a base class for all exceptions in Python, and it is defined in the sys module.
    def __init__(self, error_message, error_detail):
        super().__init__(error_message) #super() function is used to call the parent class constructor
        self.error_message = error_message_detail(error_message, error_detail) #error_message_detail() function is used to get the error message details
        
    def __str__(self):
        return self.error_message #return the error message details
    
    '''
if __name__ == "__main__":
    
    try:
        a = 10/0
    except Exception as e:
        logging.info("Division by zero error")
     #logging.log() method logs the message with the ERROR level
        raise CustomException(e, sys)
    '''
