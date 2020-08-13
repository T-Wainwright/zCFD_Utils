# zCFD_Utils
Utilities for working with zCFD Solver

Tom Wainwright

University of Bristol

tom.wainwright@bristol.ac.uk

## Current modules:

* mesh_utils:
  * Unpack and manipulate both ZCFD and CBA meshes
  * Convert meshes to tecplot .plt format
  * Stitch together multible mesh blocks/ multiblock meshes
  * Extract surfaces for RBF mesh deformation
  * Convert single block CBA meshes into zcfd format
* converter
  * Convert multiblock CBA meshes into zcfd format

* Examples included:
  * Omesh.blk
    * Single block
    * 2D
    * NACA0012 Aerofoil
    * O- Mesh topology
  * Cmesh.blk- Multiblock
    * 3 Blocks
    * 2D
    * RAE2822 Aerofoil
    * CH- Mesh topology
  * MDO_125K.blk
    * 8 Blocks
    * 3D
    * BRITE-EURAM MDO wing
    * CH- Mesh topology
  * CT0_250K.blk
    * 36 Blocks
    * 3D
    * Caradonna-Tung rotor, 0 &deg
    * CH- Mesh attached topology with H expansion blocks