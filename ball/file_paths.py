from hyperparams import *

SYMBOL_GLOB = "../symbols/ubuntu/*.json"
SYM_DATA_PATH = f"../symbols/ball-sym-data-{BALL_RADIUS}"
MEM_GRAPH_PATH = f"./ball-mem-graph-{BALL_RADIUS}.pkl"

model_name = f"ball-model-{BALL_RADIUS}-{BALL_CONV_LAYERS}-{EPOCHS}-{BATCH_SIZE}-{UNKNOWN}-{SELF_LOOPS}-{LEARNING_RATE}"
MODEL_PATH = f"./{model_name}.pkl"
RESULTS_PATH = f"./{model_name}-results.pkl"

MATCHING_SYMBOLS_PATH = "../data_dump/vmlinux-5.4.0-58-generic.json"
TASKS_CSV_PATH = "../data_dump/nokaslr_tasks.csv"
POINTER_CSV_PATH = "../data_dump/nokaslr_pointers.csv"
RAW_DUMP_PATH = "../data_dump/nokaslr.raw"
