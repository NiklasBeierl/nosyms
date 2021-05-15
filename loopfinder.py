#!/usr/bin/python

import csv

### copy/past from https://en.wikipedia.org/wiki/Cycle_detection
def floyd(f, x0):
    # Main phase of algorithm: finding a repetition x_i = x_2i.
    # The hare moves twice as quickly as the tortoise and
    # the distance between them increases by 1 at each step.
    # Eventually they will both be inside the cycle and then,
    # at some point, the distance between them will be
    # divisible by the period λ.
    tortoise = f(x0) # f(x0) is the element/node next to x0.
    hare = f(f(x0))
    while tortoise != hare:
        tortoise = f(tortoise)
        hare = f(f(hare))

    # At this point the tortoise position, ν, which is also equal
    # to the distance between hare and tortoise, is divisible by
    # the period λ. So hare moving in circle one step at a time,
    # and tortoise (reset to x0) moving towards the circle, will
    # intersect at the beginning of the circle. Because the
    # distance between them is constant at 2ν, a multiple of λ,
    # they will agree as soon as the tortoise reaches index μ.

    # Find the position μ of first repetition.
    mu = 0
    tortoise = x0
    while tortoise != hare:
        tortoise = f(tortoise)
        hare = f(hare)   # Hare and tortoise move at same speed
        mu += 1

    # Find the length of the shortest cycle starting from x_μ
    # The hare moves one step at a time while tortoise is still.
    # lam is incremented until λ is found.
    lam = 1
    hare = f(tortoise)
    while tortoise != hare:
        hare = f(hare)
        lam += 1

    return lam, mu

###
ptr = dict()
with open("ptr.csv", newline="") as csvfile:
    r = csv.reader(csvfile, delimiter=",")
    for row in r:
        if not row[2]:
            next
        ptr[row[0]] = row[2]

used = dict()

def fwd(a):
    if a in used:
        raise Exception("alreay processed")
    return ptr[a]

def getcycle(a, len):
    cycle = [a]
    used[a] = True
    while len:
        print("appending ",a)
        a = ptr[a]
        cycle.append(a)
        used[a] = True
        len -= 1
    return cycle

# print("test...")
# print(getcycle('0xbee8358',120))


print("Starting...")
cycles = []
for a in ptr.keys():
    try:
        lam, mu = floyd(fwd, a)
        # size 1 loop is not really a loop.... 2 is also boring
        if lam <= 2:
            next
        # mu>0: just a pointer to the cycle, not the cycle itself
        elif mu > 0:
            next
        else:
            cycles.append( (lam, getcycle(a,lam)) )
    except Exception as e:
        # print(e)
        next

cycles.sort(key=lambda x:x[0], reverse=True)

for c in cycles:
    print("Cycle of length {} at {}".format(c[0],c[1][0]))
    print("Full cycle: ",c[1])

