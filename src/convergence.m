% T = readtable('../../cases/MDO_250K/MDO_250K_report.csv')
T = readtable('../../cases/IEA_15_P1_OUTPUT/IEA_15_report.csv')

plot(T.Cycle,T.rho)
