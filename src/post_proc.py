import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_table('../../cases/CT8_1M_P2_OUTPUT/CT8_1M_report.csv',delimiter=' ')

plt.plot(data.Cycle,data.rho)
plt.xlabel('Cycle')
plt.ylabel('rho')
plt.show()