import sys
import traceback

def error_message_detail(error, error_detail):
    """
    Extracts details from an exception.
    
    :param error: The exception object.
    :param error_detail: The sys module object which contains the traceback details.
    :return: Formatted error message string.
    """
    _, _, exc_tb = error_detail.exc_info()  # Extract exception info
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = (
        f"Error occurred in script: [{file_name}] "
        f"line number: [{exc_tb.tb_lineno}] "
        f"error message: [{str(error)}]"
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        """
        Initialize the CustomException with a detailed error message.
        
        :param error_message: The error message to be included in the exception.
        :param error_detail: The sys module object which contains the traceback details.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
        
    def __str__(self):
        return self.error_message