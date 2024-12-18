from complexity import * 
from simplification import *
from util import *
import os
import time
import datetime
import sys



if __name__ == '__main__':

    start_time = time.time()
    date_time = datetime.datetime.now().strftime('%d%b%Y_%H%M').upper()

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    model_type = sys.argv[3]
    
    input_file = cleanup_input(input_file)
    simplify_file(input_file, output_file, model_type)

    end_program(start_time)


