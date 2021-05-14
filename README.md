# About

// TODO

# Volatility Plugins
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
