n_i = 5
n_j = 4
n_k = 3

fname = '../data/MB.blk'

f = open(fname,"w")

f.write("2 \t 1 \t 2.5 \n")
f.write("{} \t {} \t {} \n".format(n_i,n_j,n_k))
for k in range(n_k):
    for j in range(n_j):
        for i in range(n_i):
            f.write("{} \t {} \t {} \n".format(float(i),float(j),float(k)))
f.write("0 \t 0 \t 0\n")
f.write("0 \t 0 \t 0\n")
f.write("0 \t 0 \t 0\n")
f.write("2 \t 2 \t 3\n")
f.write("0 \t 0 \t 0\n")
f.write("0 \t 0 \t 0\n")


f.write("{} \t {} \t {} \n".format(n_i,n_j,n_k))
for k in range(n_k):
    for j in range(n_j):
        for i in range(n_i):
            f.write("{} \t {} \t {} \n".format(float(i),float(j+n_j),float(k)))
f.write("0 \t 0 \t 0\n")
f.write("0 \t 0 \t 0\n")
f.write("2 \t 1 \t 4\n")
f.write("0 \t 0 \t 0\n")
f.write("0 \t 0 \t 0\n")
f.write("0 \t 0 \t 0\n")




fname = '../data/SB.plt'

f = open(fname,"w")

f.write("VARIABLES = \"X\" \"Y\" \"Z\"")
f.write("ZONE I = {} \t J = {} \t K = {} F = POINT\n".format(n_i,n_j,n_k))
for k in range(n_k):
    for j in range(n_j):
        for i in range(n_i):
            f.write("{} \t {} \t {} \n".format(float(i),float(j),float(k)))