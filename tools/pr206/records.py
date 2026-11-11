"""Retain individual events from a live PID, then verify its final circular histories."""
import json
import mmap
import struct
import time
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HEADER = struct.Struct('<8Q')
MAGIC = 0x5052323036455654
CAPACITIES = (1 << 18, 1 << 18, 1 << 16)
NAMES = ('process', 'pricing', 'address_sample')
OPS = ('', 'malloc', 'calloc', 'posix_memalign', 'free_call', 'free_return',
       'realloc_call', 'realloc_return')
# Each row: local ring sequence, process sequence, operation, thread token,
# address, operation-specific argument or initiating call sequence, size, caller PC.
def utc():
    return datetime.now(timezone.utc).isoformat()

def unpack_range(buf, ring, start, stop):
    cap = CAPACITIES[ring]
    start = max(start, stop - cap + 1, 1)
    if start > stop:
        return np.empty((0, 8), dtype='<u8')
    base = HEADER.size + sum(CAPACITIES[:ring]) * 64
    count = stop - start + 1
    slot = (start - 1) % cap
    first = min(count, cap - slot)
    raw = buf[base + slot*64:base + (slot+first)*64]
    if count > first:
        raw += buf[base:base + (count-first)*64]
    rows = np.frombuffer(raw, dtype='<u8').reshape(-1,8)
    valid = rows[:,0] == np.arange(start,stop+1,dtype='<u8')
    current = np.ndarray((cap,),dtype='<u8',buffer=buf,offset=base,strides=(64,))
    valid[:first] &= rows[:first,0] == current[slot:slot+first]
    if count>first:
        valid[first:] &= rows[first:,0] == current[:count-first]
    valid &= (rows[:,1]>0) & (rows[:,2]>=1) & (rows[:,2]<=7)
    return rows if valid.all() else rows[valid]

class Reader:
    def __init__(self, path):
        self.path = path
        self.fd = path.open('rb')
        self.buf = mmap.mmap(self.fd.fileno(), 0, access=mmap.ACCESS_READ)
        self.last = [0,0,0]
        self.next_window = [0.,0.,0.]
        self.cap = [0, 256<<20, 192<<20]
        self.encoded = [0,0,0]
        self.saved = [0,0,0]
        self.chunks = [0,0,0]
        self.streams = [path.with_suffix('.'+n+'.evt').open('xb') for n in NAMES]

    def read(self, final=False):
        magic,version,pid,*values = HEADER.unpack_from(self.buf)
        if magic != MAGIC or version != 1:
            return
        now = time.monotonic()
        for k,end in enumerate(values[:3]):
            if end <= self.last[k]:
                continue
            # Once a continuous archive reaches its budget, preserve periodic
            # individual events and the entire final ring. The counters expose gaps.
            limited = self.encoded[k] >= self.cap[k]
            if limited and now < self.next_window[k] and not final:
                continue
            end_stable = end if final else max(0, end - 64)
            start = self.last[k]+1
            if limited:
                start=max(start, end_stable - (4096 if k==0 else 8192)+1)
            rows = unpack_range(self.buf, k, start, end_stable)
            if not len(rows):
                continue
            # Columns are delta encoded modulo 2^64. First value is absolute.
            columns = rows.T.copy()
            columns[:,1:] = columns[:,1:] - columns[:,:-1]
            packed = zlib.compress(columns.tobytes(), 1)
            self.streams[k].write(struct.pack('<IIQ',len(rows),len(packed),time.time_ns())+packed)
            self.encoded[k] += len(packed)+16
            self.saved[k] += len(rows)
            self.chunks[k] += 1
            self.last[k] = int(rows[-1,0])
            self.next_window[k] = now + 30.

    def close(self):
        self.read(final=True)
        for s in self.streams:
            s.close()
        header=HEADER.unpack_from(self.buf)
        out={'file':self.path.name,'pid':header[2],'wheel_mapped':header[7]>header[6]>0,
             'streams':{},'wheel_operations':{},'wheel_allocation_free_pair':False}
        for i,n in enumerate(NAMES):
            end=header[3+i]
            rows=unpack_range(self.buf,i,1,end)
            out['streams'][n]={'events':end,'final_rows':len(rows),'archived_rows':self.saved[i],
                               'compressed_bytes':self.encoded[i],'chunks':self.chunks[i]}
            if i==1:
                out['wheel_operations']=dict(Counter(OPS[int(op)] for op in rows[:,2]))
                live={}
                example=None
                pending={}
                for row in rows:
                    _,seq,op,thread,ptr,aux,size,pc=map(int,row)
                    if op in (1,2,3,7) and ptr:
                        live[ptr]=(seq,pc-header[6],size)
                    elif op==4 and ptr in live:
                        a=live.pop(ptr)
                        pending[seq]=(ptr,{'allocation_seq':a[0],'allocation_image_offset':hex(a[1]),
                                 'bytes':a[2],'free_call_seq':seq,'free_image_offset':hex(pc-header[6])})
                    elif op==5 and aux in pending and pending[aux][0]==ptr:
                        example=pending[aux][1] | {'free_return_seq':seq}
                        break
                out['wheel_allocation_free_pair']=example or False
        self.buf.close()
        self.fd.close()
        return out

def monitor(directory, done):
    readers={}
    while True:
        for path in directory.glob('events-*.bin'):
            if path not in readers and path.stat().st_size == HEADER.size+sum(CAPACITIES)*64:
                readers[path]=Reader(path)
        for r in readers.values():
            r.read()
        if done.is_set(): break
        done.wait(0.2)
    return [r.close() for r in readers.values()]
