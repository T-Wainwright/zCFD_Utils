import zutil


parameters = {
    # units for dimensional quantities
    "units": "SI",
    # reference state
    "reference": "IC_1",
    "time marching": {
        "unsteady": {"total time": 1.0, "time step": 1.0, "order": "second"},
        "scheme": {"name": "runge kutta", "stage": 5},
        "multigrid": 10,
        "cfl": 2.0,
        "cfl ramp factor": {"initial": 1.0, "growth": 1.1},
        "cycles": 10000,
    },
    "report": {"frequency": 10},
    "equations": "RANS",
    "RANS": {
        "order": "second",
        "limiter": "vanalbada",
        "precondition": True,
        "turbulence": {"model": "sst"},
    },
    "material": "air",
    "air": {
        "gamma": 1.4,
        "gas constant": 287.0,
        "Sutherlands const": 110.4,
        "Prandtl No": 0.72,
        "Turbulent Prandtl No": 0.9,
    },
    "IC_1": {
        "temperature": zutil.to_kelvin(540.0),
        "pressure": 101325.0,
        "V": {"vector": [1.0, 0.0, 0.0], "Mach": 0.2},
        "Reynolds No": 5.0e6,
        "Reference Length": 1.0,
        "turbulence intensity": 3.8e-2,
        "eddy viscosity ratio": 0.01,
        "ambient turbulence intensity": 1.0e-20,
        "ambient eddy viscosity ratio": 1.0e-20,
    },
    "IC_2": {
        "reference": "IC_1",
        "total pressure ratio": 1.02828,
        "total temperature ratio": 1.008,
    },
    "BC_1": {"zone": [0], "type": "symmetry"},
    "BC_2": {"zone": [1], "type": "symmetry"},
    "BC_3": {"zone": [5], "type": "wall", "kind": "noslip"},
    "BC_4": {"zone": [2], "type": "inflow", "condition": "IC_2", "kind": "default"},
    "BC_5": {"zone": [3], "type": "farfield", "condition": "IC_1", "kind": "riemann"},
    "BC_6": {"zone": [6], "type": "farfield", "condition": "IC_1", "kind": "riemann"},
    "BC_7": {"zone": [4], "type": "symmetry"},
    "write output": {
        "format": "vtk",
        "surface variables": [
            "V",
            "p",
            "T",
            "rho",
            "yplus",
            "ut",
            "nu",
            "pressureforce",
            "frictionforce",
            "eddy",
        ],
        "volume variables": ["V"],
    },
}