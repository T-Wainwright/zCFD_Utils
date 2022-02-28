# DPW5 Example Case

The Drag Prediction Workshop (DPW) is a regular workshop organised by the AIAA where the CFD industry participate in attempting to accurately predict the drag behaviour of the NASA Common Research Model (CRM) under predefined conditions. The meshes for this example are taken from the 5th installment of the workshop as these are small enough to run on a local machine.

More information about the workshop can be found [here](https://aiaa-dpw.larc.nasa.gov/Workshop5/workshop5.html)
___

## Steps to run the case locally

1. Activate the zCFD virtual environment, and navigate to data location

```
source $PATH_TO_zCFD/bin/activate

cd $PATH_TO_zCFD_UTILS/Tutorials/Examples/DPW5
```

2. Convert the mesh file to zCFD h5 format
```
ugridconvert L1.T.rev01.p3d.hex.r8.ugrid L1.T.rev01.p3d.hex.h5
```

3. Launch pvserver to check converted mesh file
```
pvserver
```

4. Launch the paraview GUI, select connect in the network bar, bringing up the Choose Server Configuration dialogue box. Click Add Server, and give your connection a name, keep the host as localhost, and set the port to match the port in the above address (11111) in this example. Click configure, then save as manual start up. Then you just need to select the connection you created, and click connect.

5. Open the zCFD mesh, and select the zCFD reader

6. In the 'properties' toolbar, progressively select and deselect the 'zones', to ensure the zone numbering matches that in the input deck (remember range is non inclusive of the last element)

```
sym = [ii for ii in range(1, 9)]    = [1, 2, 3, 4, 5, 6, 7, 8]
wall = [ii for ii in range(9, 14)]  = [9, 10, 11, 12, 13]
ff = [ii for ii in range(14, 19)]   = [14, 15, 16, 17, 18]
```
7. Close paraview, and check the input deck is valid

```
validate_input DPW5.py
```
8. Check the input deck maps boundaries properly to the mesh (they should already match if we've visually checked properly)

```
validate_input DPW5.py -m L1.T.rev01.p3d.hex.h5
```
9. Run the solver (here using 1 process, but feel free to play about with this)

```
run_zcfd -n 1 -p L1.T.rev01.p3d.hex.h5 -c DPW5.py 
```
10. Launch jupyter lab to monitor (from within the zCFD environment) run the command, then run the link provided in a browser. If the solver is actively running, you will need to repeat step 1 in a new terminal window before running this step.
```
jupyter-lab --no-browser
```
11. Launch paraview either and open `$PATH_TO_zCFD_UTILS/Tutorials/Examples/DWP5/DPW5_P1_OUTPUT/DPW5.pvd` to view the volume output. Open `$PATH_TO_zCFD_UTILS/Tutorials/Examples/DWP5/DPW5_P1_OUTPUT/DPW5_wall.pvd` to view the surface output on the wall, the same follows for the periodic and farfield surfaces.
