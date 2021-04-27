from paraview.simple import *
from paraview.vtk.numpy_interface import dataset_adapter as dsa
from paraview.vtk.numpy_interface import algorithms as algs
import numpy as np 

# User input variables

blade_radius = 2.04
root_cutout = 0.21
p_inf = 103037
rho_inf = 1.225
spanwise_locations_pctg = [0.25, 0.35, 0.6, 0.82, 0.92]
omega = 44.45

# Calculate spanwise cut locations

spanwise_locations_m = []

for loc in spanwise_locations_pctg:
    m = (loc*blade_radius) + root_cutout
    spanwise_locations_m.append(m)

n_spanwise_stations = len(spanwise_locations_m)


# load data
reader = OpenDataFile("Mexico_SST_P140_OUTPUT/Mexico_SST_wall.pvd")
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

    # stick data into useful array

    rawData = servermanager.Fetch(zSlice)
    data = dsa.WrapDataObject(rawData)

    results[i] = {}
    results[i]['p'] = data.PointData['p']
    results[i]['y'] = data.Points[:,1]

    # print(data.PointData['p'])

    v_local = np.linalg.norm([omega * spanwise_locations_m[i], 15])

    results[i]['cp'] = np.zeros_like(results[i]['p'])
    results[i]['y/x'] = np.zeros_like(results[i]['y'])

    results[i]['cp'] = [(p - p_inf)/(0.5*rho_inf*v_local**2) for p in results[i]['p']]
    
    y_min = min(results[i]['y'])
    y_max = max(results[i]['y'])
    delta = abs(y_min) + abs(y_max)

    results[i]['y/c'] = [(y - y_min)/(delta) for y in results[i]['y']]
    
# Dump out tecplot format data
f = open('MEXICO_V15.dat',"w")

f.write("TITLE=\"MEXICO DATA\"\n")
f.write("VARIABLES=\"x/c\" \"cp\" \n")
for i in range(n_spanwise_stations):
    f.write("ZONE\n")
    for j in range(len(results[i]['y/c'])):
        f.write("{} \t {} \n".format(results[i]['y/c'][j], results[i]['cp'][j]))