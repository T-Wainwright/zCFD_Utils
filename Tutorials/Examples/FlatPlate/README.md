# Flate Plate Example Case

The zero pressure gradient flat plate is a common benchmarking test for turbulence modelling in viscous flows, to ensure boundary layer effects are accurately captured. More information on the specifics of the case can be found [here](https://zcfd.zenotech.com/validation/plate).
___

## Steps to run the case locally

1. Activate the zCFD virtual environment, and navigate to data location

```
source $PATH_TO_zCFD/bin/activate

cd $PATH_TO_zCFD_UTILS/Tutorials/Examples/FlatPlate
```
2. Check the input deck is valid

```
validate_input plate_coarse.py
```
3. Check the input deck maps boundaries properly to the mesh

```
validate_input plate_coarse.py -m plate_coarse.h5
```
4. Run the solver (here using 1 process, but feel free to play about with this)

```
run_zcfd -n 1 -p plate_coarse.h5 -c plate_coarse.py
```
5. Launch jupyter lab to monitor (from within the zCFD environment) run the command, then run the link provided in a browser. If the solver is actively running, you will need to repeat step 1 in a new terminal window before running this step.
```
jupyter-lab --no-browser
```
6. Launch paraview either and open `$PATH_TO_zCFD_UTILS/Tutorials/Examples/FlatPlate/plate_coarse_P1_OUTPUT/CT0_250K.pvd` to view the volume output. Open `$PATH_TO_zCFD_UTILS/Tutorials/Examples/FlatPlate/plate_coarse_P1_OUTPUT/plate_coarse_wall.pvd` to view the surface output on the wall, the same follows for the symmetry and farfield surfaces.
