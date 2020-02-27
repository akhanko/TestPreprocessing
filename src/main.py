import shutil
from functools import reduce, partial

from src.feature_extraction import *
from src.file_processing import *
from src.utils import *
from src.configs import *
from multiprocessing import Pool
import math


def file_paths_generator(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))


def split_to_chunks(file, chunks_dir, processed_dir, chunk_size, pool):
    split_file(file, chunks_dir, chunk_size)
    if os.path.isdir(processed_dir):
        shutil.rmtree(processed_dir)
    pool.map(partial(preprocess_chunk, processed_dir), file_paths_generator(chunks_dir))


def merge_result_chunks(path):
    if os.path.isfile(path):
        os.remove(path)

    result_file = open(path, 'w+')
    for file in os.listdir(OUTPUT_CHUNKS_DIR):
        filename = os.path.join(OUTPUT_CHUNKS_DIR, file)
        curr_file = open(filename, 'r')
        if os.stat(path).st_size != 0:
            next(curr_file)
        result_file.write(curr_file.read())
        os.remove(filename)


def fit(file, chunk_size, pool, preprocess=True):
    if preprocess:
        split_to_chunks(file, TRAIN_CHUNKS_DIR, TRAIN_PREPROC_DIR, chunk_size, pool)
    else:
        if not is_dir_exists(TRAIN_PREPROC_DIR):
            raise Exception('Directory with preprocessed chunks is empty.')

    mean_mapped = pool.map(mean_mapper, file_paths_generator(TRAIN_PREPROC_DIR))
    mean_reduced = reduce(mean_reducer, mean_mapped)
    count = mean_reduced['count']
    means = [x / count for x in mean_reduced['sum']]

    squared_mapped = pool.map(partial(squared_mapper, means), file_paths_generator(TRAIN_PREPROC_DIR))
    squared_reduced = reduce(squared_reducer, squared_mapped)
    means_of_squared = [x / count for x in squared_reduced['squared']]
    stds = [math.sqrt(mean) for mean in means_of_squared]

    metrics = {'means': means,
              'stds': stds}
    return metrics


def transform(input_file, chunk_size, output_file, pool, fitted, preprocess=True):
    if preprocess:
        split_to_chunks(input_file, TEST_CHUNKS_DIR, TEST_PREPROC_DIR, chunk_size, pool)
    else:
        if not is_dir_exists(TEST_PREPROC_DIR):
            raise Exception('Directory with preprocessed chunks is empty.')

    # calculate features
    pool.map(partial(feature_mapper, fitted['means'], fitted['stds']), file_paths_generator(TEST_PREPROC_DIR))

    merge_result_chunks(output_file)


if __name__ == "__main__":
    num_threads = 4
    pool = Pool(num_threads)

    train = os.path.join(DATA_DIR, 'train.tsv')
    test = os.path.join(DATA_DIR, 'test.tsv')
    output = 'test_proc.tsv'

    chunk_len = 200
    fitted = fit(train, chunk_len, pool)
    transform(test, chunk_len, output, pool, fitted)
