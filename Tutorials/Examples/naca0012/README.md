# NACA0012 Example Case

This case simulates a NACA0012 under varying angles of attack. It is again a standard test case for CFD validation, with more information available [here](https://zcfd.zenotech.com/validation/naca0012).

___

## Steps to run the case locally

1. Activate the zCFD virtual environment, and navigate to data location

```
source $PATH_TO_zCFD/bin/activate

cd $PATH_TO_zCFD_UTILS/Tutorials/Examples/naca0012
```
2. Check the input deck is valid

```
validate_input n0012_897_a0p0.py
```
3. Check the input deck maps boundaries properly to the mesh

```
validate_input n0012_897_a0p0.py -m naca0012_0897.h5
```
4. Run the solver (here using 1 process, but feel free to play about with this)

```
run_zcfd -n 1 -p naca0012_0897.h5 -c n0012_897_a0p0.py
```
5. Launch jupyter lab to monitor (from within the zCFD environment) run the command, then run the link provided in a browser. If the solver is actively running, you will need to repeat step 1 in a new terminal window before running this step.
```
jupyter-lab --no-browser
```
6. Launch paraview either and open `$PATH_TO_zCFD_UTILS/Tutorials/Examples/naca0012/n0012_897_a0p0_P1_OUTPUT/n0012_897_a0p0.pvd` to view the volume output. Open `$PATH_TO_zCFD_UTILS/Tutorials/Examples/naca0012/n0012_897_a0p0_P1_OUTPUT/n0012_897_a0p0_wall.pvd` to view the surface output on the wall, the same follows for the and farfield and symmetry surfaces.
