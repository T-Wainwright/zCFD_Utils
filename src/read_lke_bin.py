import numpy as np
def read_vector(fname,offset):
    # Read Header 
    HEADER = np.fromfile(fname,dtype=np.int32,count=3,offset=offset)
    offset = offset + 3*4

    # Process Header 
    # print(HEADER)
    if HEADER[0] != 1:
        print('Wrong data type')
        exit()
    if HEADER[1] != 1:
        print('Wrong dimensions')
        exit()

    # Read Data
    DATA = np.fromfile(fname,dtype=np.double,offset=offset,count=HEADER[2])
    offset = offset + HEADER[2]*8
    return DATA, offset

def read_intvector(fname,offset):
    # Read Header 
    HEADER = np.fromfile(fname,dtype=np.int32,count=3,offset=offset)
    offset = offset + 3*4

    # Process Header 
    print(HEADER)
    if HEADER[0] != 0:
        print('Wrong data type')
        exit()
    if HEADER[1] != 1:
        print('Wrong dimensions')
        exit()

    # Read Data
    DATA = np.fromfile(fname,dtype=np.int32,offset=offset,count=HEADER[2])
    offset = offset + HEADER[2]*4
    return DATA, offset

def read_int(fname,offset):
    # Read Header
    HEADER = np.fromfile(fname,dtype=np.int32,count=3,offset=offset)
    offset = offset + 3*4

    #Process Header
    # print(HEADER)
    if HEADER[0] != 0:
        print('Wrong data type')
        exit()
    if HEADER[1] != 0:
        print('Wrong dimensions')
        exit()
    if HEADER[2] != 1:
        print('Wrong dimensions')
        exit()

    # Read Data
    VALUE = int(np.fromfile(fname,dtype=np.int32,count=1,offset=offset)[0])
    offset = offset + 4


    return VALUE, offset

def read_scalar(fname,offset):
    # Read Header
    HEADER = np.fromfile(fname,dtype=np.int32,count=3,offset=offset)
    offset = offset + 3*4

    # Process Header
    # print(HEADER)
    if HEADER[0] != 1:
        print('Wrong data type')
        exit()
    if HEADER[1] != 0:
        print('Wrong dimensions')
        exit()
    if HEADER[2] != 1:
        print('Wrong dimensions')
        exit()
    VALUE = int(np.fromfile(fname,dtype=np.double,count=1,offset=offset)[0])
    offset = offset + 8

    return VALUE, offset

def read_matrix(fname,offset):
    # Read Header 
    HEADER = np.fromfile(fname,dtype=np.int32,count=4,offset=offset)
    offset = offset + 4*4
    # Process Header 
    print(HEADER)
    if HEADER[0] != 1:
        print('Wrong data type')
        exit()
    if HEADER[1] != 2:
        print('Wrong dimensions')
        exit()

    n = HEADER[2]
    m = HEADER[3]

    DATA = np.zeros((n,m))
    # Read Data
    for j in range(m):
        DATA[:,j] = np.fromfile(fname,dtype=np.double,offset=offset,count=n)
        offset = offset + n*8
    
    return DATA, offset

def read_CRS(fname,offset):
    # Read Header 
    HEADER = np.fromfile(fname,dtype=np.int32,count=5,offset=offset)
    offset = offset + 5*4
    # Process Header 
    print(HEADER)
    if HEADER[0] != 1:
        print('Wrong data type')
        exit()
    if HEADER[1] != -1:
        print('Wrong dimensions')
        exit()

    n1 = HEADER[2]
    n2 = HEADER[3]
    n3 = HEADER[4]
    DATA = {}

    DATA['vals'] = np.zeros(n1)
    DATA['col_ind'] = np.zeros(n2)
    DATA['row_ptr'] = np.zeros(n3)

    # Read Data
    DATA['vals'] = np.fromfile(fname,dtype=np.double,offset=offset,count=n1)
    offset = offset + n1*8
    DATA['col_ind'] = np.fromfile(fname,dtype=np.int32,offset=offset,count=n2)
    offset = offset + n2*4
    DATA['row_ptr'] = np.fromfile(fname,dtype=np.int32,offset=offset,count=n3)
    offset = offset + n3*4
    print(DATA)
    return DATA, offset


fname = '../data/aero_nodes.xyz.meshdef'

offset = 0
i = 0

FILE_DATA={}

FILE_DATA['Xs'], offset = read_vector(fname,offset)
FILE_DATA['Ys'], offset = read_vector(fname,offset)
FILE_DATA['Zs'], offset = read_vector(fname,offset)

FILE_DATA['normsX'], offset = read_vector(fname,offset)
FILE_DATA['normsY'], offset = read_vector(fname,offset)
FILE_DATA['normsZ'], offset = read_vector(fname,offset)


FILE_DATA['meshType'], offset = read_int(fname,offset)

print(FILE_DATA['meshType'])
if FILE_DATA['meshType'] == 0:
    FILE_DATA['Xv'], offset = read_vector(fname,offset)
    FILE_DATA['Yv'], offset = read_vector(fname,offset)
    FILE_DATA['Zv'], offset = read_vector(fname,offset)

FILE_DATA['rbfmode'], offset = read_int(fname,offset)
FILE_DATA['r0'], offset = read_scalar(fname,offset)
FILE_DATA['ySymm'], offset = read_scalar(fname,offset)


print(FILE_DATA['rbfmode'])
if FILE_DATA['rbfmode'] == 1:
    FILE_DATA['nbase'], offset = read_int(fname,offset)
    FILE_DATA['phi_b'], offset = read_matrix(fname,offset)
    FILE_DATA['psi_r'], offset = read_matrix(fname,offset)
    FILE_DATA['LCRS'], offset = read_CRS(fname,offset)
    FILE_DATA['inew'], offset = read_intvector(fname,offset)
    FILE_DATA['radii'], offset = read_vector(fname,offset)
    FILE_DATA['scalings'], offset = read_vector(fname,offset)
    FILE_DATA['psi_v'], offset = read_CRS(fname,offset)
elif FILE_DATA['rbfmode'] == 0:
    FILE_DATA['Mmat'], offset = read_matrix(fname,offset)





print(FILE_DATA)