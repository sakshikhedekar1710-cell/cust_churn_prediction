import sys


def error_message_detail(error, error_detail: sys):
    """
    Create a detailed error message including file name and line number.
    """
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    return (
        f"Error occurred in python script [{file_name}] "
        f"at line number [{exc_tb.tb_lineno}] "
        f"with error message [{str(error)}]"
    )


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message