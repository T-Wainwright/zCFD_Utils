import zutil

alpha = 10.0


def my_transform(x, y, z):
    v = [x, y, z]
    v = zutil.rotate_vector(v, alpha, 0.0)
    return {"v1": v[0], "v2": v[1], "v3": v[2]}


parameters = {
    "restart": False,
    # units for dimensional quantities
    "units": "SI",
    # reference state
    "reference": "IC_1",
    "time marching": {
        "unsteady": {"total time": 1.0, "time step": 1.0, "order": "second"},
        "scheme": {"name": "runge kutta", "stage": 5},
        "multigrid": 2,
        "multigrid cycles": 20000,
        "cfl": 1.5,
        "cfl transport": 1.0,
        "cfl coarse": 1.0,
        "cfl ramp factor": {"initial": 0.05, "growth": 1.01},
        "cycles": 50000,
    },
    "equations": "RANS",
    "RANS": {
        "order": "euler_second",
        "limiter": "vanalbada",
        "precondition": True,
        "preconditioner": {"minimum mach number": 0.25},
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
        "V": {"vector": zutil.vector_from_angle(alpha, 0.0), "Mach": 0.15},
        #'viscosity' : 0.0,
        "Reynolds No": 6.0e6,
        "Reference Length": 1.0,
        "turbulence intensity": 5.2e-2,
        "eddy viscosity ratio": 1.0,
        "ambient turbulence intensity": 5.2e-2,
        "ambient eddy viscosity ratio": 1.0,
    },
    "BC_1": {
        #'ref' : 3,
        "zone": [0, 1],
        "type": "symmetry",
    },
    "BC_2": {
        #'ref' : 3,
        "zone": [4],
        "type": "wall",
        "kind": "noslip",
    },
    "BC_3": {
        #'ref' : 9,
        "zone": [2, 3, 5],
        "type": "farfield",
        "condition": "IC_1",
        "kind": "riemann",
    },
    "report": {
        "frequency": 10,
        #'Scale residuals by volume' : True,
        "forces": {
            "FR_1": {
                "name": "wall",
                "zone": [4],
                "transform": my_transform,
                "reference area": 1.0,
            }
        },
    },
    "write output": {
        "format": "vtk",
        "surface variables": [
            "V",
            "p",
            "T",
            "rho",
            "cp",
            "cf",
            "pressureforce",
            "pressuremoment",
            "pressuremomentx",
            "pressuremomenty",
            "frictionforce",
            "frictionmoment",
            "frictionmomentx",
            "frictionmomentz",
            "yplus",
            "var_6",
            "var_7",
        ],
        "volume variables": ["V", "p", "T", "rho", "mach", "cp", "eddy"],
    },
}
