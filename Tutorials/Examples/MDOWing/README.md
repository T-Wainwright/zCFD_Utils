# MDO Wing Example Case

The Brite-Euram MDO wing is an optimisation test case designed to be reflective of an A380 style wing. The mesh for this case has been generated using Chris Allen's structured multiblock mesh generator, then converted with the converter in `zCFD_Utils`.

___

## Steps to run the case locally

1. Activate the zCFD virtual environment, and navigate to data location

```
source $PATH_TO_zCFD/bin/activate

cd $PATH_TO_zCFD_UTILS/Tutorials/Examples/MDOWing
```
2. Check the input deck is valid

```
validate_input MDO_125K.py
```
3. Check the input deck maps boundaries properly to the mesh

```
validate_input MDO_125K.py -m MDO_125K.blk.h5
```
4. Run the solver (here using 1 process, but feel free to play about with this)

```
run_zcfd -n 1 -p MDO_125K.blk.h5 -c MDO_125K.py
```
5. Launch jupyter lab to monitor (from within the zCFD environment) run the command, then run the link provided in a browser. If the solver is actively running, you will need to repeat step 1 in a new terminal window before running this step.
```
jupyter-lab --no-browser
```
6. Launch paraview either and open `$PATH_TO_zCFD_UTILS/Tutorials/Examples/MDOWing/MDO_125K_P1_OUTPUT/MDO_125K.pvd` to view the volume output. Open `$PATH_TO_zCFD_UTILS/Tutorials/Examples/MDOWing/MDO_125K_P1_OUTPUT/MDO_125K_wall.pvd` to view the surface output on the wall, the same follows for the symmetry and farfield surfaces.
