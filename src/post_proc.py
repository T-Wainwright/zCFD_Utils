import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_table('../../cases/IEA_15MW/RANS/IEA_15_RANS_report.csv',delimiter=' ')

plt.plot(data.Cycle[:100],data["rho"][:100])
# plt.plot(data.Cycle[:100],data["rhoV[2]"][:100])
plt.xlabel('Cycle')
plt.ylabel('rho')
plt.yscale("log")
plt.show()

print(data.keys())
# cp = pd.read_table('../../cases/IEA_15_P1_OUTPUT/cp.csv',delimiter=',')

# print(cp.keys)
# plt.plot(cp["Points:0"],cp.cp)
# plt.xlabel('Cycle')
# plt.ylabel('rho')
# plt.show()