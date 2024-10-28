import sys
import logging
from src.logger import logging

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message='Error Occured in python script name [{0}], line number [{1}], error massage [{2}]'.format(
        file_name,exc_tb.tb_lineno,str(error))
    return error_message


class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        '''The super().__init__(error_message) is used in object-oriented programming in Python, specifically when you are working with inheritance. 
            It allows you to call a method from the parent (superclass) within the child class (subclass). 
            In this case, it is typically used in exception classes when you want to inherit and extend the behavior of a built-in exception or custom base class.
            In simple Language: When you create a custom class that is a child of another class (called the parent class), sometimes you want to reuse the work that the parent class already does. 
            In Python, we use super() to call methods from the parent class in the child class.'''
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message

