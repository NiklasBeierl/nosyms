import json
import pickle
from glob import glob
from concurrent.futures import TimeoutError
from pebble import ProcessPool
from encoding import VolatilitySymbolsEncoder
from encoding.ball import BallGraphBuilder
from file_paths import SYM_DATA_PATH, SYMBOL_GLOB
import develop.filter_warnings

TARGET_SYMBOL = "task_struct"
POINTER_SIZE = 8

# Parallel processing causes issues with rbi-tree sometimes, see: https://github.com/mikpom/rbi_tree/issues/4
DO_PARALLEL = False
# Only used if DO_PARALLEL == True
ENCODE_TIMEOUT = 120


def encode_sym_file(sym_path):
    with open(sym_path) as f:
        symbol_json = json.load(f)

    if symbol_json["base_types"]["pointer"]["size"] != POINTER_SIZE:
        raise ValueError(f"Wont encode {sym_path}, because its not a 64 bit architecture.")

    if TARGET_SYMBOL not in symbol_json["user_types"]:
        raise ValueError(f"Wont encode {sym_path}, because it has no {TARGET_SYMBOL} user type.")

    print(f"Encoding: {sym_path}")
    with open(sym_path) as f:
        syms = json.load(f)

    # This type is in a hand full of debugging symbols and its size is 4 GiB, causing the symbol encoder to produce
    # an OOM. It contains a 'mmuext_op' struct and then a zero length array of "long_unsigned_int".
    # Any other type in my dataset is below one MiB in size, so I guess its fair to consider this an outlier.
    del syms["user_types"]["unnamed_a315a22bd125afd5"]

    sym_encoder = VolatilitySymbolsEncoder(syms)
    ball_encoder = BallGraphBuilder()
    graph, node_ids = ball_encoder.create_type_graph(sym_encoder, TARGET_SYMBOL)

    print(f"Done encoding: {sym_path}")
    return sym_path, graph, node_ids


all_paths = list(glob(SYMBOL_GLOB))


if DO_PARALLEL:
    pool = ProcessPool()
    result = pool.map(encode_sym_file, all_paths, timeout=ENCODE_TIMEOUT).result()
    all_data = []
    while True:
        try:
            all_data.append(next(result))
        except TimeoutError:
            print("A symbols file timed out.")
        except StopIteration:
            break
        except Exception as e:
            print(e)
else:
    all_data = [encode_sym_file(p) for p in all_paths]

processed_paths = [p for p, _, _ in all_data]
if set(all_paths) != set(processed_paths):
    failed = set(all_paths).difference(processed_paths)
    print(f"Failed to process {len(failed)} / {len(all_paths)} sym files: {failed}")

print("Saving symbol data as pickle.")
with open(SYM_DATA_PATH, "wb+") as f:
    pickle.dump(all_data, f)

print("Done")
