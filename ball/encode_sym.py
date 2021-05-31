import json
import pickle
from glob import glob
from multiprocessing import cpu_count
from concurrent.futures import TimeoutError
from pebble import ProcessPool
from encoding import VolatilitySymbolsEncoder
from encoding.ball import BallGraphBuilder
from file_paths import SYM_DATA_PATH, SYMBOL_GLOB

import warnings

warnings.filterwarnings("ignore", message="DGLGraph\.__len__")
warnings.filterwarnings("ignore", message="Undefined\ type\ encountered")

ENCODE_TIMEOUT = 120
TARGET_SYMBOL = "task_struct"
POINTER_SIZE = 8


def encode_sym_file(sym_path):
    with open(sym_path) as f:
        symbol_json = json.load(f)

    if symbol_json["base_types"]["pointer"]["size"] != POINTER_SIZE:
        raise ValueError(f"Wont encode {sym_path}, because its not a 64 bit architecture.")

    if TARGET_SYMBOL not in symbol_json["user_types"]:
        raise ValueError(f"Wont encode {sym_path}, because it has no {TARGET_SYMBOL} user type.")

    print(f"Encoding: {sym_path}")
    try:
        with open(sym_path) as f:
            syms = json.load(f)

        sym_encoder = VolatilitySymbolsEncoder(syms)
        ball_encoder = BallGraphBuilder()
        graph, node_ids = ball_encoder.create_type_graph(sym_encoder, TARGET_SYMBOL)

        print(f"Done encoding: {sym_path}")
        return (sym_path, graph, node_ids)
    except Exception as e:
        raise Exception(f"Failed to encode {TARGET_SYMBOL} from {sym_path}.") from e


pool = ProcessPool(max_workers=cpu_count())
all_paths = list(glob(SYMBOL_GLOB))
result = pool.map(encode_sym_file, all_paths, timeout=ENCODE_TIMEOUT).result()

all_data = []
while True:
    try:
        all_data.append(next(result))
    except TimeoutError:
        warnings.warn("A symbols file timed out.")
    except StopIteration:
        break
    except Exception as e:
        print(e)

processed_paths = [p for p, _, _ in all_data]
if set(all_paths) != set(processed_paths):
    failed = set(all_paths).difference(processed_paths)
    print(f"Failed to process {len(failed)} / {len(all_paths)} sym files: {failed}")

print("Saving symbol data as pickle.")
with open(SYM_DATA_PATH, "wb+") as f:
    pickle.dump(all_data, f)

print("Done")
