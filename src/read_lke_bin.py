import numpy as np
def read_vector(fname,offset):
    # print(offset)

    HEADER = np.fromfile(fname,dtype=np.int32,count=3,offset=offset)
    # print(HEADER)
    offset = offset + 3*4
    DATA = np.fromfile(fname,dtype=np.double,offset=offset,count=HEADER[2])
    offset = offset + HEADER[2]*8
    return DATA, offset

def read_int(fname,offset):
    VALUE = np.fromfile(fname,dtype=np.int32,count=1,offset=offset)
    offset = offset + 4

    return VALUE, offset

fname = '../../cases/MDO_250K/aero_nodes.xyz.meshdef'

offset = 0
i = 0

FILE_DATA={}

while i < 6:
    FILE_DATA[i], offset = read_vector(fname,offset)
    print(i)
    print(FILE_DATA[i])
    i = i+1

FILE_DATA[i], offset = read_int(fname,offset)
print(FILE_DATA[i])
i = i+1

while i < 14:
    FILE_DATA[i], offset = read_vector(fname,offset)
    print(i)
    print(FILE_DATA[i])
    i = i+1

