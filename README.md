# About
The code in this repo is two things:  
Experiments conducted for my bachelor thesis, I will add a link to the PDF once I finalized it.  

And secondly it's a small library with tools that allow processing debugging symbols and memory snapshots with neural 
networks. More specifically, it allows encoding `struct` definitions from 
[Volatilty3](https://github.com/volatilityfoundation/volatility3) symbol json files, as well as actual memory snapshots 
as [DGL](dgl.ai) graphs. 
If that sounds like something you want to dive into, the best place to get started is probably the thesis mentioned 
above. Or you can of course just dive into the code itself:


# File / package structure:
- `nosyms.encoding`: Encode Memory Snapshots and Volatility Symbols into dgl graphs
- `nosyms.encoding.ball`: Implementation of the encoding mechanism used in my experiments
- `nosyms.nn`: NN stuff: Models / Some utils
- `nosyms.symbol_analysis`: Some utils to analyze batches of Volatility3 symbol json files.
  (Check `ball/data_analysis.py` for usage examples.)
- `./ball`: Scripts for my experiments with ball encoding (Take a look there to see how `nosyms` is used.) 
  - Note: I mostly run this code "interactively". I would encourage you to do the same.
- `./data_dump` / `./symbols` Where I keep my memory snapshot data and volatility3 symbols. 
  They are not checked into git, since they are huge files. You can get the files used in my experiments from 
  [my dropbox](https://www.dropbox.com/sh/iouddhc3zzut0xy/AACcREb-8JiESOntFIv59XjHa?dl=0).
  
# Setup
> Note: You do **not** need to set up the dependencies of this project to use the volatility3 plugins! A working
installation of [volatility3](https://github.com/volatilityfoundation/volatility3) is sufficient.

This project uses [poetry](https://python-poetry.org/docs/) to manage its python dependencies. The most convenient way 
to get started probably is using `pip3 install --user poetry` to install poetry "globally" and then running
`poetry install` after `cd`ing into this projects folder. You can then use `poetry shell` to `activate` the `venv`
poetry created and manages for this project. Wanna get out? Just run `deactivate` like in any old `venv`. :)

You can have volatility3 and matplotlib added to the venv by respectively appending `-E vol` or `-E plotting` to 
`poetry install`.

## Using CUDA (recommended)
If you want to run computations on a GPU, you need to install the cuda version of dgl. See how to choose ad install 
the cuda version correctly [here](https://www.dgl.ai/pages/start.html).
The easyiest way is to `peotry shell` into your venv after it was created with `poetry install` and then 
`pip install dgl-cudaXX.X`  the cuda version of dgl.

# Volatility3 Plugins
## `pointer_scan.KernelMemPointerScan` 
... scans a memory snapshots "physical layer" for x86-64 "canonical" "kernel space" pointers (8 byte aligned words with
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
pointer_scan.KernelMemPointerScan \
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

### `Misaligned pointer detected!`
Usually pointers are aligned, meaning they are an offset which is a multiple of their length.
There are of course exceptions to this rule, like for example structs that serialize pointers:  
The warning prints the args of the encoding function, which should somehow reveal which struct contained this
misaligned pointer. Figuring out whether that struct containing a misaligned pointer is a problem will likely require
doing a bit of research. For the Linux Kernel I recommend starting 
[here](https://elixir.bootlin.com/linux/latest/source).

Some cases I encountered : 
- `saved_context` from [suspend64](https://github.com/torvalds/linux/blob/614124bea77e452aa6df7a8714e8bc820b489922/arch/x86/include/asm/suspend_64.h#L21)
  - since it is related to suspension I guess it's fine. :)
- A bunch of of structs related to acpi: `acpi_resource_uart_serialbus`, `acpi_resource_source`, 
  `acpi_resource_pin_function`
  - Not really sure about that so for, but it only happened with one of my symbol files.
