from complexity import * 
from simplification import *
import os
import time
import datetime


if __name__ == '__main__':

    start_time = time.time()
    date_time = datetime.datetime.now().strftime('%d%b%Y_%H%M').upper()

    input = 'input_ss.txt'
    output = 'output_ss.txt'
    file = os.path.join(input)

    output_lines = simplify_file(file, date_time)

    with open(output, 'w', encoding='utf-8') as outfile:
        for line in output_lines:
            outfile.write(line + '\n')

    end_program(start_time)