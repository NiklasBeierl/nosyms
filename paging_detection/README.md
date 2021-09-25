# About
This is some experimental code to try and see if x86 64bit long mode paging structures can be detected in raw memory without knowing pml4 (PGD) positions. 

# General Notes:
- Paging structures as well as paging entries are handled with dataclasses. See `__init__.py`.
  - One should avoid storing "not present" entries, since that will likely overwhelm your memory / cause huge files.
- These dataclasses can conveniently be (de)serialized to/from json.
- Networkx is used to analyze the topology of the paging structures, graphs are stored as `.graphml` for use with other graph tools like gephi.
- Currently focused on linux.

# Data flow:
All of the python scripts use argparse. You can invoke them with `--help`.

### Get PML4 (PGD) addresses from your snapshot
Use the `pslist_with_pgds.PsListWithPGDs` Volatility3 plugin.
```bash
cd path/to/nosyms
vol -p volatility_plugins/ -f data/dump -r csv pslist_with_pgds.PsListWithPGDs > data/dump_pgds.csv
```
Note: You need a profile matching the linux kernel running in the snapshot.

### Extract known paging structures (Get the ground truth)
The last argument is the physical address of the PML4 used to translate `active_mm->pgd` of all the tasks in the snapshot.
```bash
cd path/to/nosyms/paging_detection
python3 extract_known_paging_structures.py --kpti ../data/dump ../data/dump_pgds.csv 39886848
```
Produces:
```
../data/dump_known_pages.json
../data/dump_known_pages.graphml
```
Pass `--kpti` or `--no-kpti` according to whether the snapshot comes from a kernel with page table isolation.

### Extract paging information for all pages 
```bash
cd path/to/nosyms/paging_detection
python3 extract_all_pages.py ../data/dump
```
Produces:
```
../data/dump_all_pages.json
../data/dump_all_pages.graphml
```

### Determine possible types for all pages (Prediction)
Point the script to the "all_pages" `.json` or `.graphml`, it will figure out the path of the other one automatically.
```bash
cd path/to/nosyms/paging_detection
python3 determine_types.py ../data/dump_all_pages.json
```
Produces:
```
../data/dump_all_pages_with_types.json
../data/dump_all_pages_with_types.graphml
```
### (Optinally) apply additional filters
Point the script to the "all_pages_with_types" `.json` or `.graphml`, it will figure out the path of the other one automatically.
```bash
cd path/to/nosyms/paging_detection
python3 filters.py ../data/dump_all_pages_with_types.json
```
Produces:
```
../data/dump_all_pages_with_types_filtered.json
```

### Compare results
Point it to the "prediction" and "ground truth" `.json`.
It prints a table with some stats.
```bash
cd path/to/nosyms/paging_detection
python3 analze_type_prediction.py ../data/dump_all_pages_with_types.json ../data/dump_known_pages.json
```