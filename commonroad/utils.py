import numpy as np
import os


def format_line(name, value, unit=''):
    """
    Formats a line e.g.
    {Name:}           {value}{unit}
    """
    name += ':'
    if isinstance(value, (float, np.ndarray)):
        value = f'{value:{0}.{4}}'

    return f'{name.ljust(40)}{value}{unit}'


def save_arrays(path, a_dict):
    """
    :param path: Output path
    :param a_dict: A dict containing the name of the array as key.
    """
    path = path.rstrip('/')

    if not os.path.isdir(path):
        os.mkdir(path)

    if len(os.listdir(path)) == 0:
        folder_number = '000'
    else:
        folder_number = str(int(max(os.listdir(path))) + 1).zfill(3)

    os.mkdir(f'{path}/{folder_number}')

    for key in a_dict:
        np.save(f'{path}/{folder_number}/{key}.npy', a_dict[key])
