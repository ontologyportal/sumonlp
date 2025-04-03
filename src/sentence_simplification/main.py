from complexity import * 
from simplification import *
from util import *
import os
import time
import datetime
import sys
import psutil





if __name__ == '__main__':

    start_time = time.time()
    date_time = datetime.datetime.now().strftime('%d%b%Y_%H%M').upper()

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    model_type = sys.argv[3]
    ram_in_bytes = psutil.virtual_memory().total  # total physical memory in Bytes
    ram_in_gb = ram_in_bytes / (1024 ** 3)  # convert to GB
    print(f"Available RAM: {ram_in_gb:.2f} GB")

    if ram_in_gb < 10:
        print("Insufficient RAM for llama3.1:8b q8. Using q4.")
    
    model_type =  'llama3.1:8b-instruct-q4_0'
    simplify_file(input_file, output_file, model_type)

    end_program(start_time)


