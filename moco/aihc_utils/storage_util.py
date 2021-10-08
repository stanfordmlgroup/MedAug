import os
import datetime

from pathlib import Path


def get_storage_folder(exp_folder, exp_name, exp_type):

    datestr = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    path_name = Path(os.path.join(
        exp_folder, f'{exp_name}_{exp_type}_{datestr}'))

    os.makedirs(path_name)

    print(f'Experiment storage is at {path_name}')
    return path_name
