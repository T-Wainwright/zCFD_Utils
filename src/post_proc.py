import matplotlib.pyplot as plt
import pandas as pd
from paramiko import SSHClient
from scp import SCPClient

ssh = SSHClient()
ssh.load_system_host_keys()
ssh.connect('bc4login.acrc.bris.ac.uk', username='tw15748')

scp = SCPClient(ssh.get_transport())

remote_path1 = 'scratch/jobs/zcfd/CT/M0.225_Validation/CT8_M0225_report.csv'
remote_path2 = 'scratch/jobs/zcfd/CT/M0.439_Validation/CT8_M0439_report.csv'

scp.get(remote_path1, local_path='../data/')
scp.get(remote_path2, local_path='../data/')

data1 = pd.read_table('../data/CT8_M0225_report.csv', delimiter=' ')
data2 = pd.read_table('../data/CT8_M0439_report.csv', delimiter=' ')


plt.plot(data1.Cycle, data1["rhoOmega"], markevery=100)
plt.plot(data2.Cycle, data2["rhoOmega"], markevery=100)

# plt.plot(data.Cycle[:100],data["rhoV[2]"][:100])

plt.xlabel('Cycle')
plt.ylabel('rho')
plt.yscale("log")
plt.title('Convergence')

# plt.locator_params(numticks=12)

# fig, (ax1, ax2) = plt.subplots(1,2)
# fig.suptitle('Convergence')
# ax1.semilogy(data1.Cycle,data1["rho"])
# ax2.plot(data2.Cycle,data2["rho"])
# ax2.set(yl)

plt.show()

# cp = pd.read_table('../../cases/IEA_15_P1_OUTPUT/cp.csv',delimiter=',')

# print(cp.keys)
# plt.plot(cp["Points:0"],cp.cp)
# plt.xlabel('Cycle')
# plt.ylabel('rho')
# plt.show()
