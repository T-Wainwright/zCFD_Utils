# Download report csv from remote cluster, extract solver convergence data and plot convergence

import matplotlib.pyplot as plt
import pandas as pd
from paramiko import SSHClient
from scp import SCPClient

# remote path and local paths
remote_path1 = 'scratch/jobs/zcfd/CT/M0.225_Validation/CT8_M0225_report.csv'  # Path to remote data
local_path = '../data/'  # Path to where you want data stored

# Set up ssh link
ssh = SSHClient()
ssh.load_system_host_keys()
ssh.connect('bc4login.acrc.bris.ac.uk', username='tw15748')  # Bluecrustal login info
scp = SCPClient(ssh.get_transport())

# Download and load report file
scp.get(remote_path1, local_path=local_path)
data = pd.read_table('../data/CT8_M0225_report.csv', delimiter=' ')

# Plot convergence data
plt.plot(data.Cycle, data["rhoOmega"], markevery=100)

plt.xlabel('Cycle')
plt.ylabel('rho')
plt.yscale("log")
plt.title('Convergence')

plt.show()