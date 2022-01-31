# Post processing script for Mexico reference wind turbine cases
# T-Wainwright 2021

from paraview.simple import *
from paraview.vtk.numpy_interface import dataset_adapter as dsa
from paraview.vtk.numpy_interface import algorithms as algs
import numpy as np
import pylab as pl

def write_tec(results, data_dir, tecName):
    # Dump out tecplot format data
    f = open(data_dir + tecName + '.dat', "w")

    f.write("TITLE=\"{}\"\n".format(tecName))
    f.write("VARIABLES=\"x/c\" \"cp\" \n")
    for i in range(n_spanwise_stations):
        f.write("ZONE T=\"{}% span\"\n".format(int(results[i]['spanwise_pctg'])))
        for j in range(len(results[i]['y/c'])):
            f.write("{} \t {} \n".format(results[i]['y/c'][j], results[i]['cp'][j]))


def cp_plot(results, data_dir, fname, exp_data):
    fig = pl.figure(figsize=(25,30), dpi=100, facecolor='w', edgecolor='k')
    fig.suptitle('Mexico Rotor Blade (' + r'$\mathbf{V_{\infty}}$' + '= 15m/s)', fontsize=28, fontweight='normal', color='#5D5858')

    for case in results:
        ax = fig.add_subplot(3,2,case+1)
        ax.set_title(r'$\mathbf{C_P}$' + ' at ' r'$\mathbf{r/R}$' + ' = {:.0f}'.format(results[case]['spanwise_pctg']*100) + '%', fontsize=24, fontweight='normal', color='#5d5858')
        ax.grid(True)
        ax.set_xlabel('$\mathbf{x/c}$', fontsize=24, fontweight='bold', color = '#5D5858')
        ax.set_ylabel('$\mathbf{C_p}$', fontsize=24, fontweight='bold', color = '#5D5858')
        ax.plot(results[case]['y/c'], results[case]['cp'],label='zCFD')
        ax.invert_yaxis()
        
        plot_experimental(ax, case, exp_data)
        ax.legend(loc='upper right', shadow=True)

        legend = ax.legend(loc='best', scatterpoints=1, numpoints=1, shadow=False, fontsize=16)

    fig.savefig(data_dir + "Cp_plots.png")
        


def load_experimental(data_dir, fname):
    exp_data = {}
    raw_data = np.loadtxt(data_dir + fname)

    line = 0
    i = 0
    while line < raw_data.shape[0]:
        exp_data[i] = {}
        exp_data[i]['spanwise_pctg'] = raw_data[line, 0]
        num_entries = int(raw_data[line, 1])
        exp_data[i]['num_entries'] = num_entries
        line = line + 1
        exp_data[i]['y/c'] = raw_data[line:line + num_entries, 0]
        exp_data[i]['cp'] = raw_data[line:line + num_entries, 1]
        line = line + num_entries
        i = i+1

    return exp_data


def plot_experimental(ax, case, exp_data):
    ax.scatter(exp_data[case]['y/c'], exp_data[case]['cp'], label='Experimental')

# User input variables

blade_radius = 2.25
root_cutout = 0.21
p_inf = 103037
rho_inf = 1.225
spanwise_locations_pctg = [0.25, 0.35, 0.6, 0.82, 0.92]
omega = 44.45

case_name = 'Mexico_SST'
data_dir = '/home/tom/Documents/University/Coding/cases/Mexico/Mexico_SST_P140_OUTPUT/'

# Calculate spanwise cut locations

spanwise_locations_m = []

for loc in spanwise_locations_pctg:
    m = (loc * blade_radius)
    spanwise_locations_m.append(m)
    print(spanwise_locations_m)

n_spanwise_stations = len(spanwise_locations_m)

#--------------------------------------------------------------------------------------#
#-------------------------------Paraview Section---------------------------------------#
#--------------------------------------------------------------------------------------#

# load data
reader = OpenDataFile(data_dir + "Mexico_SST_wall.pvd")
UpdatePipeline()

# Convert cell data to point data

cellDataToPointData = CellDatatoPointData(Input=reader)
cellDataToPointData.ProcessAllArrays
UpdatePipeline()

mergeBlocks = MergeBlocks(Input=cellDataToPointData)

# Extract streamwise slice locations, normalise y and save csv

results = {}

for i in range(n_spanwise_stations):
    zSlice = Slice(Input=mergeBlocks)
    zSlice.SliceType = 'Plane'
    zSlice.HyperTreeGridSlicer = 'Plane'
    zSlice.SliceOffsetValues = [0.0]

    zSlice.SliceType.Origin = [0.0, 0.0, spanwise_locations_m[i]]
    zSlice.SliceType.Normal = [0.0, 0.0, 1.0]

    UpdatePipeline()

    sortedLine = PlotOnSortedLines(Input=zSlice)
    UpdatePipeline()
    mergeBlocks2 = MergeBlocks(Input=sortedLine)
    UpdatePipeline()

    # stick data into useful array

    rawData = servermanager.Fetch(mergeBlocks2)
    data = dsa.WrapDataObject(rawData)

    results[i] = {}
    results[i]['spanwise_pctg'] = spanwise_locations_pctg[i]
    results[i]['spanwise_m'] = spanwise_locations_m[i]
    results[i]['p'] = data.PointData['p']
    results[i]['y'] = data.Points[:, 1]

    v_local = np.linalg.norm([omega * spanwise_locations_m[i], 15])
    print(v_local)

    results[i]['cp'] = np.zeros_like(results[i]['p'])
    results[i]['y/x'] = np.zeros_like(results[i]['y'])

    results[i]['cp'] = [(p - p_inf) / (0.5 * rho_inf * v_local ** 2) for p in results[i]['p']]

    y_min = min(results[i]['y'])
    y_max = max(results[i]['y'])
    delta = abs(y_min) + abs(y_max)

    results[i]['y/c'] = [(y - y_min) / (delta) for y in results[i]['y']]

    # Connect first and last results element to remove any gaps in cp plot
    results[i]['y/c'].append(results[i]['y/c'][0])
    results[i]['cp'].append(results[i]['cp'][0])


write_tec(results, data_dir, 'Mexico_V15')
exp_data = load_experimental(data_dir, 'Experimental_V15_raw.dat')
cp_plot(results, data_dir, 'test', exp_data)
