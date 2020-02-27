import os
import uuid
import pandas as pd


def split_file(input_file, output_dir, chunk_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chunk_pattern = '{}.tsv'
    chunk_path = os.path.join(output_dir, chunk_pattern)

    with open(input_file, 'r') as f:
        chunk_file = None
        header = f.readline()
        for i, line in enumerate(f):
            if i % chunk_size == 0:
                if chunk_file:
                    chunk_file.close()
                chunk_file = open(chunk_path.format(str(uuid.uuid4())), 'w')
                if line != header:
                    chunk_file.write(header)
            chunk_file.write(line)
        if chunk_file:
            chunk_file.close()


def preprocess_chunk(processed_dir, file):
    os.makedirs(processed_dir, exist_ok=True)
    df = pd.read_csv(file, delimiter='\t', header=0)

    feature_cols = df.columns.difference(['id_job'])
    features_df = [df[col].str.split(",", expand=True) for col in feature_cols]

    cleaned = list()

    for i, feature in enumerate(features_df, 1):
        id = feature[0][0]
        cols = ['feature_{}_{}'.format(id, j) for j in range(0, feature.shape[1] - 1)]
        cleaned_feature = feature.loc[:, 1:]
        cleaned_feature.columns = cols
        cleaned.append(cleaned_feature)

    df_concat = pd.concat(cleaned, axis=1)
    df_concat.insert(0, 'id_job', df['id_job'].astype(int))

    filename = '{}.tsv'.format(str(uuid.uuid4()))
    df_concat.to_csv(os.path.join(processed_dir, filename), index=False)
    os.remove(file)

