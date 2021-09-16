# About
This is some experimental code to try and see if x86 64bit long mode paging structures can be detected in raw memory without knowing pml4 (PGD) positions. 

# General Notes:
- Paging structures as well as paging entries are handled with dataclasses. See `__init__.py`.
  - One should avoid storing "not present" entries, since that will likely overwhelm your memory / cause huge files.
- These dataclasses can conveniently be (de)serialized to/from json.
- Networkx is used to analyze the topology of the paging structures, graphs are stored as `.graphml` for use with other graph tools like gephi.
- Currently only supports linux

# Data low:

## Get PML4 (PGD) addresses from your snapshot
Use the `pslist_with_pgds.PsListWithPGDs` Volatility3 plugin.
```bash
cd path/to/nosyms
vol -p volatility_plugins/ -f path/to/snapshot -r csv pslist_with_pgds.PsListWithPGDs  >> path/to/pslist.csv
```
Note: You need a profile matching the linux kernel running in the snapshot.

## Extract known paging structures
```bash
cd path/to/nosyms/paging_detection
python3 extract_known_paging_structures.py
```
Note: Assumes that your snapshot comes from a linux OS with page table isolation.

## Extract paging information for all pages
```bash
cd path/to/nosyms/paging_detection
python3 extract_all_pages.py
```

## Determine possible types for all pages

```bash
cd path/to/nosyms/paging_detection
python3 determine_types.py 
```

## Compare results

```bash
cd path/to/nosyms/paging_detection
python3 analze_type_prediction.py 
```
