# zCFD_Utils
Utilities and tutorials for working with zCFD Solver

Tom Wainwright

University of Bristol

tom.wainwright@bristol.ac.uk

## Installing zCFD_Utils
This is required for using the codebase within the zCFD virtual environment. If you are running the tutorials from here these steps are not required.

1. Download zCFD from [here](https://zcfd.zenotech.com/download)

2. Unpack the download file:
```
./zCFD-icc-sse-impi-2021.11.765-Linux-64bit.sh
```
- If this doesn't work, you may need to make the file executable:
```
chmod u+x zCFD-icc-sse-impi-2021.11.765-Linux-64bit.sh
```

3. Clone this repository into a location OUTSIDE of the zCFD folder system
```
git clone https://github.com/T-Wainwright/zCFD_Utils.git
```

4. Launch into the zCFD virtual environment
```
source $PATH_TO_ZCFD/bin/activate
```

5. Navigate to the root folder of zCFD_Utils, and install using pip:

```
cd $PATH_TO_zCFD_Utils/
pip install .
```
## Tutorial pages
---
* Quick start guide
  
  *  *Note this assumes 0 experience working with ANY CFD solver, and is intended for undergraduate/ graduate level students using zCFD for the first time.*

* Mesh description notebook
  
  An interactive notebook for understanding the zCFD mesh format, with demonstrations of some tools which can be used along side it. The notebook requires `h5py` and `numpy` modules to function. The mesh used as a demonstration can be found in the `data` directory.
  
* Example cases
  *  2D zero pressure gradient flat plate
  *  2D NACA0012 Aerofoil
  *  3D MDO Wing
  *  3D Caradonna Tung rotor
  *  3D DPW5 Case (Advanced)
