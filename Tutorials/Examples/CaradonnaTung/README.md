# Caradonna Tung Example Case

The Caradonna Tung rotor is a standard rotor validation case, looking at up to transonic flow around a 6ft rotor in a wind tunnel. The mesh used here is 180 degree periodic domain, generated using Chris Allen's structured multiblock mesh generator, and converted using the `mesh_utils` module in `zCFD_utils`.

[source paper](https://ntrs.nasa.gov/api/citations/19820004169/downloads/19820004169.pdf)

___

## Steps to run the case locally

1. Activate the zCFD virtual environment, and navigate to data location

```
source $PATH_TO_zCFD/bin/activate

cd $PATH_TO_zCFD_UTILS/Tutorials/Examples/CaradonnaTung
```
2. Check the input deck is valid

```
validate_input CT8_250K.py
```
3. Check the input deck maps boundaries properly to the mesh

```
validate_input CT0_250K.py -m CT0_250K.blk.h5
```
4. Run the solver (here using 1 process, but feel free to play about with this)

```
run_zcfd -n 1 -p CT0_250K.blk.h5 -c CT0_250K.py 
```
5. Launch jupyter lab to monitor (from within the zCFD environment) run the command, then run the link provided in a browser. If the solver is actively running, you will need to repeat step 1 in a new terminal window before running this step.
```
jupyter-lab --no-browser
```
6. Launch paraview either and open `$PATH_TO_zCFD_UTILS/Tutorials/Examples/CaradonnaTung/CT0_250K_P1_OUTPUT/CT0_250K.pvd` to view the volume output. Open `$PATH_TO_zCFD_UTILS/Tutorials/Examples/CaradonnaTung/CT0_250K_P1_OUTPUT/CT0_250K_wall.pvd` to view the surface output on the wall, the same follows for the periodic and farfield surfaces.
