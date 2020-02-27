import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

TRAIN_CHUNKS_DIR = os.path.join(TRAIN_DIR, 'chunked')
TEST_CHUNKS_DIR = os.path.join(TEST_DIR, 'chunked')
TRAIN_PREPROC_DIR = os.path.join(TRAIN_DIR, 'preproccesed')
TEST_PREPROC_DIR = os.path.join(TEST_DIR, 'preproccesed')
OUTPUT_CHUNKS_DIR = os.path.join(DATA_DIR, 'output')
