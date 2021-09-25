#!/usr/bin/env python3
import sys
import numpy as np
import subprocess

VOLPY = "/home/dl9rdz/src/volatility3/vol.py"

# Set to true if all top-level PGD shall be printed
PRINTPGD = False

# Set to true if all top-level differences scan <-> volatility tasklist shall be printed
PRINTDIFF = True

# Filter FLAGS in all levels?
#  not really much tested, so far everything worked just fine without this.
#  but it helps reducing false negatives at l2/l3 level.
FILTERALLFLAGS = 0

# Filter top level for only those with [0x1ff] present and valid
# 0=off, 1=needs to be present, 2=strict: all entries in referenced element needs to be good
FILTERx1ff = 0

# Filter top level based on FLAG variability
# Probably only few different combinations are used on the top level
# So many different combintations might be an indicatino of random garbage
# valus is threshold for maximum different values, 0=filter is off
FILTERFLAGS = 0

# Filter top level based on "sparsedness"
# only accept top level if less then FILTERMAXENTRY elements are present (0=filter is off)
FILTERMAXENTRY = 0

# Filter top level based on pairs (on Linux with KPTI only):
# only accept top level if top level +/- 1 is also top level
FILTERPAIRS = 0

# Filter top level based on frequency of kernal mapping (KMSIM: kernel mapping similarity)
# (kernel mapping should usually be identical across processes. not sure if there are exceptions?
# so a L4 page that differs in the kernel part is likely not a L4 page)
# valus: how many times a specific mapping has to exist
FILTERKMSIM = 5

### some statistics with vmneu2.dump (FILTERx1ff, FILTERFLAGS)
### 0,0 Correct: 66 PTE/96044 nonPTE, missed: 0, false positives: 145
### 1,0 Correct: 66 PTE/96088 nonPTE, missed: 0, false positives: 101
### 2,0 Correct: 66 PTE/96177 nonPTE, missed: 0, false positives: 12
### 0,2 Correct: 33 PTE/96168 nonPTE, missed: 33, false positives: 21
### 0,4 Correct: 66 PTE/96161 nonPTE, missed: 0, false positives: 28
### 0,0,300  Correct: 66 PTE/96067 nonPTE, missed: 0, false positives: 122
### 0,0,100  Correct: 66 PTE/96077 nonPTE, missed: 0, false positives: 112
### 1,0,100  Correct: 66 PTE/96118 nonPTE, missed: 0, false positives: 71
###
### false positives: in case on 2,0: all look like valid L4 structures, maybe kernel-intern pgd not refernces by process list,
###                                  or something left in memory (unused) from a process that already terminated
###                   additionally in the other cases: many look like valid paging data structures, but are not from L4 (nor L3,
###                                  in fact, I guess mostly L1?)

# Generate "ground truth" using volatility
def getpgd(dumpfile):
    print("\nStartup: Obtaining real PGDs")
    result = subprocess.run(
        ["python3", VOLPY, "-f", dumpfile, "-p", "volatility_plugins", "pgd"], stdout=subprocess.PIPE
    )
    reslist = result.stdout.splitlines()
    del reslist[0:4]  # remove header (first 4 line)
    reslist = list(map(lambda x: (int(x.split(b"\t")[1]) & 0x000FFFFFFFFFF000) >> 12, reslist))
    print("pgd results from volatility: ")
    print(list(map(lambda x: hex(x), reslist)))
    print("appending pgd+1 for KPTI")
    reslist.extend(list(map(lambda x: x + 1, reslist)))
    print(list(map(lambda x: hex(x), reslist)))
    return reslist


# optional add knowledge about mem mapped io addresses
# in the vm.dump test data, there is one pgd that (at top level) points to pfn 17fc9 at index 0xff
good = []

if False:
    good.extend(range(0x17E00, 0x17FFF))
    good.extend(range(0xF2000, 0xF2008))
    good.append(0xF3050)
    good.extend(range(0xFC000, 0xFC009))
    good.append(0xFED00)
    good.append(0xFEFFC)
    good.append(0xFEFFF)

good = dict.fromkeys(good)

# Goal: find likely candidates for top level pgd

# Possibly a paging data structure (pds) if
# at level 1:
#  * at least on present bit is set to 1
#  * for each entry with present bit set, pfn points to existing physical memory with pfn>0
#  * for each entry with present bit set, the size bit is 0
# note: this will miss some pds: completely non-present pds, pds pointing to mem mapped io, pds pointing to pfn 0
#
# at level > 1:
#  * for each entry with present bit set, pfn points to existing physical memory
#  * there is at least one present entry with
#        either size bit 0 and pointing to a pds considered valid at level-1
#        or size bit 1 (and pointing to existing physical memory)
# note: this will miss some pds: completely non-present pds, pds pointing to large mem mapped io pages
#       pds pointing *only* to pds missed at level-1 (unlikely)
#
# (We exclude pfn 0 here as this (almost) always is a false positive)
#
# returns the number of entries that look like good pds entries (0 if something looks bad)
def maybepte(page, curlevel, verbose=False):
    somepresent = 0
    seenflags = {}
    for i in range(512):
        pte = dumpfile[page * 512 + i]
        present = np.bitwise_and(pte, np.uint64(1))
        if not present:
            continue
        pfn = int(np.bitwise_and(pte, np.uint64(0x000FFFFFFFFFF000)).astype(int)) >> 12
        if (pfn <= 0 or pfn >= maxpfn) and not pfn in good:
            if verbose:
                print("%x:%x pfn %x out of range (%x)" % (page, i, pfn, pte))
            return 0
        flags = int(np.bitwise_and(pte, np.uint64(0xFFF)).astype(int))
        seenflags[flags] = 1
        size = np.bitwise_and(pte, np.uint64(128))
        if curlevel == None:
            if size > 0:
                # PT entries (lowest level) have mandatory size bit 0
                if verbose:
                    print("%x:%x pfn %x size bit is 1" % (page, i, pfn))
                return 0
            else:
                somepresent += 1
        else:
            if size == 0:
                if (curlevel >> pfn) & 1 == 1:
                    somepresent += 1
            else:
                somepresent += 1
    if FILTERALLFLAGS > 0 and len(seenflags) > FILTERALLFLAGS:
        if verbose:
            print("%x: too many different flag types: %d" % (page, len(seenflags)))
        return 0
    if verbose and (somepresent == 0):
        print("%x: no present pages found" % page)
    return somepresent


# just for testing: show all present entries of a page
def printpresent(page):
    s = ""
    for i in range(512):
        pte = dumpfile[page * 512 + i]
        present = np.bitwise_and(pte, np.uint64(1))
        if present:
            s += " [" + hex(i) + ":" + hex(pte) + "]"
    print("Page %x: PTE present at %s" % (page, s))


# Helper functions for analyzing the "ground truth":
# For all pgd elements (found with volatility, see above), determine all paging data structures in memory

#
# return list of next level paging data structures (all present entries with size bit not size; size set ==> direct mapped page at this level)
def gtnextlevel(page):
    set = []
    for i in range(512):
        pte = dumpfile[page * 512 + i]
        present = np.bitwise_and(pte, np.uint64(1))
        if not present:
            continue
        size = np.bitwise_and(pte, np.uint64(128))
        if size:
            continue
        pfn = int(np.bitwise_and(pte, np.uint64(0x000FFFFFFFFFF000)).astype(int)) >> 12
        if pfn >= maxpfn:
            # Just for seeing which pages are mapped outside memory image
            # print("Skipping %x" % pfn)
            continue
        set.append(pfn)
    return set


# "good" according to our standard (i.e. not 0, not outside memory dump)
# real pgd entries are not good if, e.g., they point to io mem
# returns [#present, #good]
def getgood(page):
    cntgood = 0
    cntpresent = 0
    for i in range(512):
        pte = dumpfile[page * 512 + i]
        present = np.bitwise_and(pte, np.uint64(1))
        pfn = int(np.bitwise_and(pte, np.uint64(0x000FFFFFFFFFF000)).astype(int)) >> 12
        if not present:
            continue
        cntpresent += 1
        if (pfn > 0 and pfn < maxpfn) or pfn in good:
            cntgood += 1
    return [cntpresent, cntgood]


# count the numbers of elements that are "good" in our sense and thus should be found if everything works perfectly
# real pds entries that are not good are not expected to be found
# most non-good L1 pds are those that have 0 present entries (there are many of them)
def countgood(pdsset):
    good = 0
    for i in pdsset:
        res = getgood(i)
        if res[0] == res[1] and res[0] > 0:
            good += 1
        # else:
        #    print("not good: ",res[0],"/",res[1])
    return good


def groundtruth(pgds):
    # bit arrays (i.e. arbitrary size integers)
    # note: in the |= 1<<page operations, we must have type(page)==int.
    #       if type(page) is numpy.uint64_t, it will not work (resulst will also be a 64bit integer, resulting in overflow/truncation)
    l4 = 0
    l3 = 0
    l2 = 0
    l1 = 0
    l3set = []
    l2set = []
    l1set = []
    print("maxpfn is %x" % maxpfn)
    print("l4set size: ", len(pgds), " nice entires: ", countgood(pgds))

    for page in pgds:
        l4 |= 1 << page
        l3set.extend(gtnextlevel(page))
    l3set = list(dict.fromkeys(l3set))
    print("l3set size: ", len(l3set), " nice entires: ", countgood(l3set))

    for page in l3set:
        l3 |= 1 << page
        l2set.extend(gtnextlevel(page))
    l2set = list(dict.fromkeys(l2set))
    print("l2set size: ", len(l2set), " nice entires: ", countgood(l2set))

    for page in l2set:
        l2 |= 1 << page
        l1set.extend(gtnextlevel(page))
    l1set = list(dict.fromkeys(l1set))
    print("l1set size: ", len(l1set), " nice entires: ", countgood(l1set))

    for page in l1set:
        l1 |= 1 << page

    print("Returning ground truth for l4,l3,l2,l1\n\n")
    return [l4, l3, l2, l1]


def realitycheck(found, real, lower=None, verbose=False):
    ok = 0
    onlyreal = 0
    onlyfound = 0
    nonok = 0
    for i in range(maxpfn):
        if ((found >> i) & 1) == 1:
            if ((real >> i) & 1) == 1:
                ok += 1
            else:
                onlyfound += 1
                if verbose:
                    print("\nPage found, but not in ground truth")
                    if lower and ((lower >> i) & 1) == 1:
                        print("In reality, page is at the lower level")
                    printpresent(i)

        else:
            if ((real >> i) & 1) == 1:
                onlyreal += 1
                if verbose:
                    print("\nPage in ground truth, but not found")
                    printpresent(i)
                    n = maybepte(i, l3, True)
                    print("n is ", n)
            else:
                nonok += 1
    print(
        "Reality check: Correct: %d PTE/%d nonPTE, missed: %d, false positives: %d\n" % (ok, nonok, onlyreal, onlyfound)
    )


def flagstat(page):
    d = {}
    for i in range(512):
        pte = dumpfile[page * 512 + i]
        flags = int(np.bitwise_and(pte, np.uint64(0xFFF)).astype(int))
        d[flags] = 1
    return len(d)


def entrystat(page):
    p = 0
    for i in range(256):  # ! nur user-part
        pte = dumpfile[page * 512 + i]
        present = int(np.bitwise_and(pte, np.uint64(0x1)).astype(int))
        p += present
    return p


def main():
    global dumpfile
    global maxpfn
    global reality
    global l3
    if len(sys.argv) > 1:
        dumpname = sys.argv[1]
    else:
        print("Usage: " + sys.argv[0] + " <dumpfile>")
        print("dumpfile needs to be a raw direct memory image (if using lime: format=padded)")
        exit(1)
    dumpfile = np.memmap(dumpname, mode="r", dtype="uint64")

    # len counts uint64 units, i.e. 512 units = 4kb
    maxpfn = len(dumpfile) // 512

    # Lets obtain reality with volatility
    truepgds = getpgd(dumpname)
    reality = groundtruth(truepgds)

    # Now scan for pages that look lake paging data structures
    good = 0
    l1 = 0
    for page in range(maxpfn):
        if maybepte(page, None) > 0:
            good += 1
            l1 |= 1 << page
    print("L1 Good: ", good, " out of ", maxpfn)
    realitycheck(l1, reality[3])

    good = 0
    l2 = 0
    for page in range(maxpfn):
        if maybepte(page, l1) > 0:
            good += 1
            l2 |= 1 << page
    print("L2 Good: ", good, " out of ", maxpfn)
    realitycheck(l2, reality[2])

    good = 0
    l3 = 0
    for page in range(maxpfn):
        if maybepte(page, l2) > 0:
            good += 1
            l3 |= 1 << page
    print("L3 Good: ", good, " out of ", maxpfn)
    realitycheck(l3, reality[1])

    good = 0
    l4 = 0
    for page in range(maxpfn):
        n = maybepte(page, l3)
        # some magic threshold. In my test image: Usually 6 entries in "top half" (but as low as 4 seen)
        if n < 4:
            continue

        # Most likely, at least on Linux systems, the top-most entry is always used
        if FILTERx1ff > 0:
            pte = dumpfile[page * 512 + 511]
            pfn = int(np.bitwise_and(pte, np.uint64(0x000FFFFFFFFFF000)).astype(int)) >> 12
            if pfn == 0 or pfn > maxpfn:
                continue
            if FILTERx1ff > 1 and not maybepte(pfn, l2):
                continue

        if FILTERFLAGS > 0 and flagstat(page) > FILTERFLAGS:
            continue
        if FILTERMAXENTRY > 0 and entrystat(page) > FILTERMAXENTRY:
            continue

        if PRINTPGD:
            printpresent(page)
        l4 |= 1 << page
        good += 1

    if FILTERPAIRS:
        good = 0
        l4u = l4
        l4 = 0
        for dpage in range(maxpfn // 2):
            if (l4u >> (dpage * 2)) & 3 == 3:
                l4 |= 3 << (2 * dpage)
                good += 2

    if FILTERKMSIM > 0:
        kmapmap = {}
        for page in range(maxpfn):
            if (l4 >> page) & 1 == 0:
                continue
            kmap = dumpfile[page * 512 + 256 : page * 512 + 512].tobytes()
            if kmap in kmapmap:
                kmapmap[kmap] += 1
            else:
                kmapmap[kmap] = 1
        for page in range(maxpfn):
            if (l4 >> page) & 1 == 0:
                continue
            kmap = dumpfile[page * 512 + 256 : page * 512 + 512].tobytes()
            if kmapmap[kmap] < FILTERKMSIM:
                l4 ^= 1 << page
                good -= 1

    print("L4 Good: ", good, " out of ", maxpfn)
    realitycheck(l4, reality[0], reality[1] | reality[2] | reality[3], PRINTDIFF)


if __name__ == "__main__":
    main()
