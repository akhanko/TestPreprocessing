import os
import re


def get_features_group(columns):
    regex = 'feature_[0-9]*'
    groups = list()
    for col in columns:
        match = re.findall(regex, col)
        if match and match[0] not in groups:
            groups.append(match[0])
    return groups


def set_zcore_column_names(cols):
    splitted = [str(col).rsplit('_', 1) for col in cols]
    new_cols = ['_'.join([split[0], 'stand', split[1]]) for split in splitted]
    return new_cols


def is_dir_exists(dir):
    if not os.path.isdir(dir) or [f for f in os.listdir(dir) if not f.endswith('.tsv')] == []:
        return False
    return True


