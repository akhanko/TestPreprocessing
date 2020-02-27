import uuid

from src.configs import OUTPUT_CHUNKS_DIR
from src.utils import set_zcore_column_names, get_features_group

import numpy as np
import pandas as pd
from operator import add
import os


def mean_mapper(file):
    df = pd.read_csv(file, delimiter=',', header=0)

    feature_sum = list()
    features_df = df.iloc[:, 1:]

    for col in features_df.columns:
        col_sum = features_df[col].sum()
        feature_sum.append(col_sum)

    count = df.shape[0]
    return {'sum': feature_sum, 'count': count}


def squared_mapper(means, file):
    df = pd.read_csv(file, delimiter=',', header=0)
    features_df = df.iloc[:, 1:]

    squared = np.zeros(features_df.shape[1])
    for i in range(0, features_df.shape[0]):
        distance = np.array(features_df.iloc[i, :]) - np.array(means)
        col_sum = (distance ** 2)
        squared = squared + col_sum
    return {'squared': squared}


def feature_mapper(means, stds, file):
    df = pd.read_csv(file, delimiter=',', header=0)
    features_df = df.iloc[:, 1:].copy()
    os.makedirs(OUTPUT_CHUNKS_DIR, exist_ok=True)

    feature_groups = get_features_group(features_df.columns)
    columns = {'id_job': df.iloc[:, 0]}
    for group in feature_groups:
        current_features = df.loc[:, df.columns.str.startswith(group)]
        max_index = current_features.values.argmax(axis=1)
        max_el = [current_features.iloc[i, max_index[i]] for i in range(0, len(max_index))]
        mean_el = [means[i] for i in max_index]
        abs_diff = np.absolute(np.subtract(max_el, mean_el))
        columns['max_{}_index'.format(group)] = max_index.astype(int)
        columns['max_{}_abs_mean_diff'.format(group)] = abs_diff

    features = pd.DataFrame(data=columns)

    z_score = calculate_z_score(features_df, means, stds)

    data = pd.concat((features, z_score), axis=1, ignore_index=False)
    unique_filename = '{}.tsv'.format(str(uuid.uuid4()))
    unique_filepath = os.path.join(OUTPUT_CHUNKS_DIR, unique_filename)
    data.to_csv(unique_filepath, header=True, index=False)


def calculate_z_score(features_df, means, stds):
    for i in range(0, features_df.shape[1]):
        distance = (features_df.iloc[:, i] - means[i]) / stds[i]
        features_df.iloc[:, i] = distance
    z_columns = set_zcore_column_names(features_df.columns)
    features_df.columns = z_columns
    return features_df


def mean_reducer(agg1, agg2):
    total_sum = list(map(add, agg1['sum'], agg2['sum']))
    total_count = agg1['count'] + agg2['count']
    return {'sum': total_sum, 'count': total_count}


def squared_reducer(agg1, agg2):
    total_squared = list(map(add, agg1['squared'], agg2['squared']))
    return {'squared': total_squared}
