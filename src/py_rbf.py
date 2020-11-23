import numpy as np

# load data points
aero_file = '../data/aero_nodes.xyz'
struc_file = '../data/struc_nodes.xyz'
pressure_file = '../data/pressure.dat'
displacement_file = '../data/def.xyz'

aero_nodes = np.loadtxt(aero_file,skiprows=1)
struc_nodes = np.loadtxt(struc_file,skiprows=1)
pressure = np.loadtxt(pressure_file,skiprows=1)
displacements = np.loadtxt(displacement_file,skiprows=1)

r0 = 5

n_a = aero_nodes.shape[0]
n_s = struc_nodes.shape[0]

# Preallocate matrices
A_as = np.ones((n_a,n_s+4))
M_ss = np.zeros((n_s,n_s))
P_s = np.ones((4,n_s))

for i in range(n_s):
    for j in range(n_s):
        rad = (np.linalg.norm((struc_nodes[i]-struc_nodes[j])))/r0
        M_ss[i][j] = ((1-rad)**4) * (4*rad+1) # Wendland C2
    P_s[1:,i] = struc_nodes[i]

for i in range(n_a):
    for j in range(n_s):
        rad = np.linalg.norm((aero_nodes[i]-struc_nodes[j]))
        A_as[i][j+4] = rad
    A_as[i][1:4] = aero_nodes[i]

CssR = np.concatenate((P_s,M_ss))
CssL = np.concatenate((np.zeros((4,4)),np.transpose(P_s)))

Css = np.concatenate((CssL,CssR),axis=1)

M_inv = np.linalg.pinv(M_ss)
M_p = np.linalg.pinv(np.matmul(np.matmul(P_s,M_inv),np.transpose(P_s)))

Top = np.matmul(np.matmul(M_p,P_s),M_inv)
Bottom = M_inv - np.matmul(np.matmul(np.matmul(np.matmul(M_inv,np.transpose(P_s)),M_p),P_s),M_inv)

B = np.concatenate((Top,Bottom))
print(B.shape)

H = np.matmul(A_as,B)
print(H.shape)
print(pressure.shape)

U_ax = np.matmul(H,displacements[:,0])
U_ay = np.matmul(H,displacements[:,1])
U_az = np.matmul(H,displacements[:,2])

f = open("../data/aero_2.xyz","w")
f.write("{}\n".format(n_a))
for i in range(n_a):
    f.write("{} {} {}\n".format(U_ax[i],U_ay[i],U_az[i]))






