# About

// TODO

# Repo / File structure:
- `encoding`: Stuff needed to encode Memory Snapshots and Volatility Symbols into graphs
- `networks`: NN related code (pytorch models)
- `ball`: Code for experimenting with ball encoding
  - I mostly run my code in a debugger and inspect the variables directly. I would encourage you to do the same.
- `data_dump` / `symbols` Where I keep my memory snapshot data and volatility3 symbols. They are not checked into git. 
  You can get the files used in my experiments from 
  [my dropbox](https://www.dropbox.com/sh/iouddhc3zzut0xy/AACcREb-8JiESOntFIv59XjHa?dl=0).
  
# Setup
> Note: You do **not** need to set up the dependencies of this project to use the volatility3 plugins! A working
installation of [volatility3](https://github.com/volatilityfoundation/volatility3) is sufficient.

This project uses [poetry](https://python-poetry.org/docs/) to manage its python dependencies. The most convenient way 
to get started probably is using `pip3 install --user poetry` to install poetry "globally" and then running
`poetry install` after `cd`ing into this projects folder. You can then use `poetry shell` to `activate` the `venv`
poetry created and manages for this project. Wanna get out? Just run `deactivate` like in any old `venv`. :) 

## Using CUDA (recommended)
If you want to run computations on a GPU, you need to install the cuda version of dgl. See how to choose ad install 
the cuda version correctly [here](https://www.dgl.ai/pages/start.html).
The easyiest way is to `peotry shell` into your venv after it was created with
`poetry install` and then `pip install dgl-cudaXX.X  the cuda version of dgl.

# Volatility3 Plugins
## `pointer_scan.HighmemPointerScan` 
... scans a memory snapshots "physical layer" for x86-64 "canonical" high mem pointers (8 byte aligned words with
bits 63-48 set to `1`). Then tries to translate them to a physical address. See 
`volatility.framework.layers.intel.Intel32e.translate`. If translation is unsuccessful the _physical_ column for the 
corresponding row is left blank. The output csv file with columns _offset_, _virtual_ and _physical_ is written to the 
path specified via `--outfile`. The results are not put out via Volatilities TreeGrid because there usually are quite a 
lot of results and volatilities "renderer" is extremely slow.  
Usage:
```shell
vol \
-p <this-project>/volatility_plugins/ \
-f /path/to/memdump \
pointer_scan.HighmemPointerScan \
--outfile pointers.csv 
```

## `pslist_with_pointers.PsListWithPointers`
An extension of Volatilities `linux.PsList` with additional columns for the virtual and physical addresses of the task 
structs as well as their `tasks.next` pointers.  
Usage:
```shell
vol \
-p <this-project>/volatility_plugins/ V\
-f /path/to/memdump \
pslist_with_pointers.PsListWithPointers
```
# Warnings

I see ... on my console, what does it mean?

### `Undefined type encountered while encoding symbols` 

Sometimes volatility symbols define structs with pointers to other structs it has no definition for. This should not
be a showstopper, but hampers the accuracy of the model. NoSysms treats these pointers the same way it treats void
pointers: It knows that the corresponding bytes are supposed to be a pointer, but it doesn't try to follow the pointer
while encoding.
