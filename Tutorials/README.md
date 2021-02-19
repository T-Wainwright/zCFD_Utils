# Quick start zCFD guide

zCFD runs in it's own python virtual environment. This means it should be fully self contained, with all modules and libraries shipped as is. In order to launch the virtual environment, run:

```
source $PATH_TO_ZCFD/bin/activate
```

This must be run from outside the virtual environment, personally I set an alias in my `.bash_aliases` or `.bashrc` to do this automatically:

```
alias zcfd="$PATH_TO_ZCFD/bin/activate"  
```

The `$` symbol preceeding `PATH_TO_ZCFD` indicates it is just an *environment variable*, and is simply the file address of the zCFD root directory. You can get it very simply by navigating to the root directory and running `pwd`, which will print the path to the current directory you are in.

Once in the zcfd virtual environment the command line should gain the zCFD prefix as see below.

```
(zCFD) user1:
```
## Running locally
---

To run zCFD, invoke the smart use the Smartlaunch command:

```
run_zcfd --ntask 10 -p $PROBLEM_NAME -c $CASE_NAME
```

There's a little to unpack here. The official guide states: 

"The optimimum number of execution tasks $ntask should match the total number of sockets (usually two per node) **not** the number of cores."

From personal experience this may be true for HPC nodes, but if you're running the code locally for development purposes, then best results are achieved by setting this to the total number of CPU's you want to use. To find the total number of CPU's you have available, run the `lscpu` command which should give you following readout:

```
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Byte Order:                      Little Endian
Address sizes:                   39 bits physical, 48 bits virtual
CPU(s):                          12
On-line CPU(s) list:             0-11
Thread(s) per core:              2
Core(s) per socket:              6
Socket(s):                       1
Vendor ID:                       GenuineIntel
CPU family:                      6
Model:                           158
Model name:                      Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz
...
```
```
ntasks = CPU(s) = Core(s) per socket * Sockets * Thread(s) per code
```

`$PROBLEM_NAME` is the path to the .h5 mesh file, and `$CASE_NAME` is the path to the .py control dictionary. The .py file must be located in the directory you call `run_zcfd` in.

For example, to run the NACA0012 test case, locally on the above specified machine:

```
run_zcfd --ntask 12 -p meshes/NACA0012.h5 -c NACA0012.py
```

## Running on a cluster
---
The `mycluster` utility sould be used for running zcfd on clusters. A comprehensive guide for how to use mycluster can be found in the official zCFD guide, which can be found at: https://docs.zenotech.com/2020.04.108/mycluster.html

## Visualising results
---
zCFD will default to outputting in vtk (visualisation toolkit) format, for visualisation through paraview. Paraview is an open source visualisation tool, which can be downloaded for free from: https://www.paraview.org/download/

The Paraview Data format (PVD) uses a single .pvd file to point to the relevant vtk files located in the visualisation directory. For the majority of cases this is the only file you need to open in paraview (although all output files need to be present in the file structure output by the solver).
The output directory will typically have the following structure:

* OUTPUT_DIRECTORY
  * VOLUME_MESH_OUTPUT.pvd
  * BOUNDARY_OUTPUTS.pvd
  * LOGGING (Directory)
    * LOGFILE.txt
  * VISUALISATION (Directory)
    * DATA_FILES.vtp/.pvtp

The number of boundary files depends on the number of discrete boundary conditions present in your mesh, for example: wall, farfield, periodic etc...
The results files are all located in the VISUALISATION directory, but only the .pvd files at this level need to be opened.

## Paraview zCFD reader
---
zCFD comes preshipped with a paraview plugin to read zCFD meshes from native .h5 format. To load this plugin you must first ensure you have a compatible version of paraview downloaded. The version zCFD is compiled against can be found in `zcfd/lib/paraview-VERSION`, the downloads section of paraview allows you to select a specific version to download, and you can have as many versions downloaded as you want. This method ensures compatibility with the zCFD distribution you have. You do not need a developer version of paraview, the precompiled binaries are all you need.

Once you have downloaded and installed the compatible paraview version, **in the zCFD virtual environment** run:

```
pvserver
```
Which should then give you the following readout:

```
Waiting for client...                                                                                                   Connection URL: cs://IT076523:11111                                                                                     Accepting connection(s): IT076523:11111 
```

In the paraview GUI, select `connect` in the network bar, bringing up the  *Choose Server Configuration* dialogue box. Click `Add Server`, and give your connection a name, keep the host as *localhost*, and set the port to match the port in the above address (11111) in this example. Click configure, then save as manual start up. Then you just need to select the connection you created, and click connect. 

If successful the pipeline browser icon should change from *builtin:* to your local server name. Additionally the command window where zCFD is running should have updated to read `Client connected.`

In the `tools` menu select `manage plugins`, and load `zCFDReader` from the remote plugins menu. You should now be able to load native zCFD .h5 mesh files into paraview for visualisation.
