
import zutil

alpha = 0.0

sym = [ii for ii in range(1, 9)]
wall = [ii for ii in range(9, 14)]
ff = [ii for ii in range(14, 19)]


def my_transform(x, y, z):
    v = [x, y, z]
    v = zutil.rotate_vector(v, alpha, 0.0)
    return {'v1': v[0], 'v2': v[1], 'v3': v[2]}


parameters = {

    'restart': False,

    # units for dimensional quantities
    'units': 'SI',

    # reference state
    'reference': 'IC_1',

    'time marching': {
        'unsteady': {
            'total time': 1.0,
            'time step': 1.0,
            'order': 'second',
        },
        'scheme': {
            'name': 'runge kutta',
            'stage': 5,
        },
        'multigrid': 1,
        'multigrid cycles': 2000,
        'cfl': 2.5,
        'cfl transport': 1.0,
        'cfl coarse': 1.0,
        'cfl ramp factor': {'initial': 0.05, 'growth': 1.01},
        'cycles': 5000,
    },

    'equations': 'RANS',

    # 'euler': {
    #     # Spatial accuracy (options: first, second)
    #     'order': 'second',
    #     # Optional (default 'vanalbada'):
    #     # MUSCL limiter (options: vanalbada)
    #     'limiter': 'vanalbada',
    #     # Optional (default False):
    #     # Use low speed mach preconditioner
    #     'precondition': True,
    #     # Optional (default False):
    #     # Use linear gradients
    #     'linear gradients': False,
    #     # Optional (default 'HLLC'):
    #     # Scheme for inviscid flux: HLLC or Rusanov
    #     'Inviscid Flux Scheme': 'HLLC',
    # },

    'RANS': {
        'order': 'second',
        'limiter': 'vanalbada',
        'precondition': False,
        'turbulence': {
            'model': 'sst',
        },
    },

    'material': 'air',

    'air': {
        'gamma': 1.4,
        'gas constant': 287.0,
        'Sutherlands const': 110.4,
        'Prandtl No': 0.72,
        'Turbulent Prandtl No': 0.9,
    },
    'IC_1': {
        'temperature': zutil.to_kelvin(540.0),
        'pressure': 101325.0,
        'V': {
            'vector': zutil.vector_from_angle(alpha, 0.0),
            'Mach': 0.85,
        },
        # 'viscosity' : 0.0,
        'Reynolds No': 6.0e6,
        'Reference Length': 1.0,
        'turbulence intensity': 5.2e-2,
        'eddy viscosity ratio': 1.0,
        'ambient turbulence intensity': 5.2e-2,
        'ambient eddy viscosity ratio': 1.0,
    },
    'BC_1': {
        # 'ref' : 3,
        'zone': sym,
        'type': 'symmetry',
    },
    'BC_2': {
        # 'ref' : 3,
        'zone': wall,
        'type': 'wall',
        'kind': 'noslip',
    },
    'BC_3': {
        # 'ref' : 9,
        'zone': ff,
        'type': 'farfield',
        'condition': 'IC_1',
        'kind': 'riemann',
    },

    'report': {
        'frequency': 10,
        # 'Scale residuals by volume' : True,
        'forces': {
            'FR_1': {
                'name': 'wall',
                'zone': [4],
                'transform': my_transform,
                'reference area': 1.0,
            },
        },
    },

    'write output': {
        'format': 'vtk',
                  'surface variables': ['V', 'p', 'T', 'rho', 'cp', 'pressureforce', 'pressuremoment', 'pressuremomentx', 'pressuremomenty'],
                  'volume variables': ['V', 'p', 'T', 'rho', 'mach', 'cp'],
        'frequency': 1000,
    },
}
