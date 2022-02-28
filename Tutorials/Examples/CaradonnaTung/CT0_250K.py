import math
import zutil

alpha = 0


def my_transform(x, y, z):
    v = [x, y, z]
    v = zutil.rotate_vector(v, alpha, 0.0)
    return {'v1': v[0], 'v2': v[1], 'v3': v[2]}


parameters = {

    # units for dimensional quantities
    'units': 'SI',

    # reference state
    'reference': 'IC_1',

    'partitioner': 'metis',

    # time marching properties
    'time marching': {
        'unsteady': {
            'total time': 1.0,
            'time step': 1.0,
            'order': 'second',
            'start': 1,
        },
        'scheme': {
            'name': 'runge kutta',
            'stage': 5,
        },
        # multigrid levels including fine mesh
        'multigrid': 5,
        'cfl': 2,
        'cfl coarse': 1.0,
        'cfl transport': 1.0,
        'cfl ramp factor': {'growth': 1.05, 'initial': 0.1, },
        'cycles': 5000,
    },

    'equations': 'RANS',

    'RANS': {
        'order': 'euler_second',
        'limiter': 'vanalbada',
        'precondition': True,
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

    'initial': 'IC_1',

    'IC_1': {
        'temperature': 289.75,
        'pressure': 103037,
        'V': {
            'vector': [0.0, 0.0, -1.0],
            'Mach': 0.2,
        },
        # 'viscosity' : 0.0,
        'Reynolds No': 3.93e6,
        'Reference Length': 1.0,
        'turbulence intensity': 0.01,
        'eddy viscosity ratio': 0.1,
    },
    'IC_2': {
        'temperature': 289.75,
        'pressure': 103037,
        'V': {
            'vector': [0.0, 0.0, -1.0],
            'Mach': 0.00001,
        },
        # 'viscosity' : 0.0,
        'Reynolds No': 3.93e6 * 0.00001 / 0.87,
        'Reference Length': 1.0,
        'turbulence intensity': 0.01,
        'eddy viscosity ratio': 0.1,
    },
    'IC_3': {
        'reference': 'IC_1',
        # static pressure/reference static pressure
        'static pressure ratio': 1.0,
    },
    'FZ_1': {
        'type': 'rotating',
        'zone': [0],
        'omega': 10,
        'axis': [0.0, 0.0, 1.0],
        'origin': [0.0, 0.0, 0.0],
    },
    'BC_1': {
        'ref': 7,
        'type': 'symmetry',
    },
    'BC_2': {
        'ref': 3,
        'zone': [4],
        'type': 'wall',
        'kind': 'wallfunction',
    },
    'BC_3': {
        'ref': 9,
        'zone': [2],
        'type': 'farfield',
        'condition': 'IC_2',
        'kind': 'riemann',
    },
    'BC_4': {
        'zone': [5],
        'type': 'periodic',
        'kind': {
            'rotated': {'theta': math.radians(180.0),
                        'axis': [0.0, 0.0, 1.0],
                        'origin': [0.0, 0.0, 0.0], },
        },
    },
    'BC_5': {
        'zone': [6],
        'type': 'periodic',
        'kind': {
            'rotated': {'theta': math.radians(180.0),
                        'axis': [0.0, 0.0, 1.0],
                        'origin': [0.0, 0.0, 0.0], },
        },

    },
    'write output': {
        'format': 'vtk',
                  'surface variables': ['V', 'p', 'T', 'rho', 'cp', 'mach', 'pressureforce', 'frictionforce'],
                  'volume variables': ['V', 'p', 'cp', 'T', 'rho', 'mach', 'cell_velocity', 'eddy'],
                  'frequency': 500,
    },
    'report': {
        'frequency': 10,
        'forces': {
            'FR_1': {
                'name': 'wall',
                'zone': [4],
                'transform': my_transform,
                'reference area': 10.0,
            },
        },
    },
}

############################
#
# Variable list
#
# var_1 to var_n
# p,pressure
# T, temperature
#
############################
