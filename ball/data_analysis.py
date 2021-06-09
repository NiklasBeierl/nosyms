from glob import glob
from symbol_analysis import get_symbols_batch_stats, get_symbols_batch_user_type_stats, json_load
from file_paths import SYMBOL_GLOB

sym_df = get_symbols_batch_stats(glob(SYMBOL_GLOB))

# Some symbols apparently have 4GiB large structs, lets check whats going on there.
user_types_df = get_symbols_batch_user_type_stats(glob(SYMBOL_GLOB))

# Seems like its only one specific type, lets take a look at an example...
syms_with_weird_type = json_load("../symbols/all_syms/4.14.111-200.el7.x86_64.json")
weird_type = syms_with_weird_type["user_types"]["unnamed_a315a22bd125afd5"]

# It contains only a 'mmuext_op' struct and then a zero length array of "long_unsigned_int". Probably safe to ignore.
# See encode_sym.py to see how to "ignore" a type by deleting it from the symbols before encoding them. :)
