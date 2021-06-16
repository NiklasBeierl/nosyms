import json
import pickle
from pathlib import Path
from glob import glob
from concurrent.futures import TimeoutError
from pebble import ProcessPool
from nosyms.encoding import VolatilitySymbolsEncoder, WordCompressor
from nosyms.encoding.ball import BallGraphBuilder
from file_paths import SYM_DATA_PATH, SYMBOL_GLOB
from hyperparams import BALL_RADIUS
import develop.filter_warnings

TARGET_SYMBOL = "task_struct"
POINTER_SIZE = 8


# Note that this will dictate memory consumption.
# With RADIUS == 200 you should calculate 10 GiB/core + some to spare.
# Some symbols (i.e. Ubuntu Kernel 5.x.x) also take 20 GiB/core \_(..)_/
CORES = 1

ENCODE_TIMEOUT = 240

# Ensure output dir exists
Path(SYM_DATA_PATH).mkdir(parents=True, exist_ok=True)

compressor = WordCompressor()


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
    if "unnamed_a315a22bd125afd5" in syms["user_types"]:
        del syms["user_types"]["unnamed_a315a22bd125afd5"]

    sym_encoder = VolatilitySymbolsEncoder(syms)
    bgb = BallGraphBuilder(radius=BALL_RADIUS)
    graph, node_ids = bgb.create_type_graph(sym_encoder)

    graph.ndata["blocks"] = compressor.compress_batch(graph.ndata["blocks"])
    out_path = Path(SYM_DATA_PATH, Path(sym_path).name).with_suffix(".pkl")
    with open(out_path, "wb+") as f:
        pickle.dump((graph, node_ids), f)

    print(f"Done encoding: {sym_path}")
    return sym_path


all_paths = list(glob(SYMBOL_GLOB))


pool = ProcessPool(CORES)
result = pool.map(encode_sym_file, all_paths, timeout=ENCODE_TIMEOUT).result()
encoded = []
while True:
    try:
        encoded.append(next(result))
    except TimeoutError:
        print("A symbols file timed out.")
    except StopIteration:
        break
    except Exception as e:
        print(f"Encoding failed: {e}")


if set(all_paths) != set(encoded):
    failed = set(all_paths).difference(encoded)
    print(f"Failed to process {len(failed)} / {len(all_paths)} sym files: {failed}")

print("Done")
