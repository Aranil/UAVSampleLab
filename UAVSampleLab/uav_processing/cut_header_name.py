"""
rename file name

"""

import os

#root_dir = input("Insert Folder Path: ")
root_dir = r'M:\_test'

number_of_elements = 12

for f in os.listdir(root_dir):
    header, tail = f.split('.')
    if len(header.split('_')) == number_of_elements:
        if header.rsplit('_', 1)[1] == '32bit':
            header_subset, tail_subset = header.rsplit('_', 1)
            old_fname = os.path.join(root_dir, f)
            new_fname = os.path.join(root_dir, '.'.join([header_subset, tail]))
            os.rename(old_fname, new_fname)
            print('file is renamed to {}'.format(new_fname))
        else:
            print('no file with pattern 32_bit has been found !')
            print('skiping file !')
    else:
        print('header has  {} elements !'.format(len(header.split('_'))))
        print('skiping file !')
