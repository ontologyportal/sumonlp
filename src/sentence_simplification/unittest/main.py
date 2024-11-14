import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_script():
    '''Prepares input and runs entry_point.sh as a subprocess'''
    print('Preparing input...')
    prepare_input()
    print('Done...')
    print('Running entry_point.sh')
    os.system('bash ../entry_point.sh')


def prepare_input():
    '''Prepares input_ss.txt file from parent dir for testing by copying the content of test_input.txt'''
    with open('test_input.txt', 'r') as f:
        lines = f.readlines()

    with open('../input_ss.txt', 'w') as f:
        f.writelines(lines)



if __name__ == '__main__':
    test_script()