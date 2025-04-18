"""
General utilities for solver agnostic aerodynamic shape optimisation

Tom Wainwright 2021

University of Bristol

Currently this code is purely setup to couple the IEA 15MW turbine blade to the zCFD solver

tom.wainwright@bristol.ac.uk
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import py_rbf
import h5py
import yaml
import skopt
import json


class opt_driver():
    # Main driver class for surrogate model based optimisation
    def __init__(self, blade_surf: Blade_surf):
        # Read in offline data
        blade_props = Blade_geom(
            '/home/tom/Documents/University/Coding/zCFD_Utils/data/IEA-15-240-RWT_FineGrid.yaml')
        modal = Modal_data(
            '/home/tom/Documents/University/Coding/zCFD_Utils/data/UIUC Aerofoil Library/', load=True)
        cage_slice = load_cage_section(
            '/home/tom/Documents/University/Coding/zCFD_Utils/data/domain_ordered.ctr.asa.16')

        with open('/home/tom/Documents/University/Coding/zCFD_Utils/data/design_space.json') as jsonfile:
            self.design_space = json.load(jsonfile)

        self.cage = control_cage(
            blade_props, blade_surf, modal, 10, cage_slice, passengers=True)
        self.cage.generate_rbf()

    def get_deformations(self, dv_index):
        def_dict = self.design_space['design point: ' + str(dv_index)]
        deformations = self.cage.deformation_list_from_dv(def_dict)
        return deformations

    def evaluate_cost_func(self):
        pass


class aerofoil_points():
    # Class for reading in aerofoil points from Masters' database- This section of code is called for performing the SVD decomposition of the aerofoil data
    def __init__(self, fname: str):
        self.name = os.path.basename(fname)

        # read points
        self.points = np.loadtxt(fname, skiprows=3)

    def get_3D_slice(self):
        # Convert 2D xy slice into 3D by adding blank z column
        return add_z(self.points)


class control_cage():
    # Class to handle control cage behaviour
    # inputs required- blade_properties class, blade_surface class, modal data class, number of cross sections, cage cross sectional shape
    def __init__(self, blade_props: Blade_geom, blade_surface: Blade_surf, modal: Modal_data, n_sections: int, cage_slice: np.array, passengers=True):
        self.modal_data = modal
        self.blade_props = blade_props
        self.blade_surface = blade_surface

        # cage specific data
        self.cage_slice = cage_slice
        self.n_sections = n_sections
        self.n_pointsPerSection = len(cage_slice[:, 0])
        self.n_cagePoints = self.n_sections * self.n_pointsPerSection
        self.passengers = passengers
        if self.passengers:
            self.n_passengers = (self.n_sections - 1) * self.n_pointsPerSection

        # Construct cage from blade geoemetry

        chord_d = self.blade_props.outer_shape['chord']
        twist_d = self.blade_props.outer_shape['twist']
        pitch_axis_d = self.blade_props.outer_shape['pitch_axis']
        x_d = self.blade_props.outer_shape['reference_axis']['x']
        y_d = self.blade_props.outer_shape['reference_axis']['y']
        z_d = self.blade_props.outer_shape['reference_axis']['z']

        thick_d = {'grid': [], 'values': []}

        for section in self.blade_props.outer_shape['airfoil_position']['labels']:
            print(section)
            thick_d['values'].append(next(
                item for item in self.blade_props.airfoils if item["name"] == section)['relative_thickness'])

        for grid in self.blade_props.outer_shape['airfoil_position']['grid']:
            thick_d['grid'].append(grid)

        # Controls region of blade cage is placed over
        self.z_locations = np.linspace(0.15, 0.99, n_sections)

        # Interpolate blade geometry onto sections

        self.chord_i = np.interp(
            self.z_locations, chord_d['grid'], chord_d['values'])
        self.twist_i = np.interp(
            self.z_locations, twist_d['grid'], twist_d['values'])
        self.thick_i = np.interp(
            self.z_locations, thick_d['grid'], thick_d['values'])
        self.pitch_axis_i = np.interp(
            self.z_locations, pitch_axis_d['grid'], pitch_axis_d['values'])
        self.x_i = np.interp(self.z_locations, x_d['grid'], x_d['values'])
        self.y_i = np.interp(self.z_locations, y_d['grid'], y_d['values'])
        self.z_i = np.interp(self.z_locations, z_d['grid'], z_d['values'])

        self.cage = np.zeros((self.n_sections, self.n_pointsPerSection, 3))
        self.deformations = np.zeros_like(self.cage)

        if self.passengers:
            self.passenger_nodes = np.zeros(
                (self.n_sections - 1, self.n_pointsPerSection, 3))
            self.passenger_deformations = np.zeros_like(self.passenger_nodes)

        for i in range(self.n_sections):
            local_cage_x = (cage_slice[:, 0] -
                            self.pitch_axis_i[i]) * self.chord_i[i]
            local_cage_y = cage_slice[:, 1] * self.chord_i[i] * self.thick_i[i]
            local_cage_z = np.ones_like(local_cage_x) * self.z_i[i]

            local_cage_x_twist = local_cage_x * \
                np.cos(self.twist_i[i]) - local_cage_y * \
                np.sin(self.twist_i[i])
            local_cage_y_twist = local_cage_y * \
                np.cos(self.twist_i[i]) + local_cage_x * \
                np.sin(self.twist_i[i])

            # rotate 90 around reference axis
            for j in range(self.n_pointsPerSection):
                self.cage[i, j, :] = np.c_[local_cage_y_twist[j] + self.x_i[i],
                                           local_cage_x_twist[j] + self.y_i[i], local_cage_z[j] + 3]

        if self.passengers:
            self.setup_passenger_nodes()

        self.add_control_nodes()

    def setup_passenger_nodes(self):
        # Add passenger nodes to mesh- at present mean between sections in spanwise direction
        for i in range(self.n_sections - 1):
            for j in range(self.n_pointsPerSection):
                self.passenger_nodes[i, j, :] = (
                    self.cage[i, j, :] + self.cage[i + 1, j, :]) / 2

    def add_control_nodes(self):
        self.control_nodes = self.blade_surface.get_root()

    def generate_rbf(self):
        # rbf_cm (mesh1 = cage, mesh2 = modal data)
        self.rbf_cm = py_rbf.UoB_coupling(
            self.cage_slice, self.modal_data.naca0012)
        self.rbf_cm.generate_transfer_matrix(0.5, rbf='c2', polynomial=False)
        self.rbf_sc = py_rbf.UoB_coupling(
            self.blade_surface.points, self.dump_points())
        self.rbf_sc.generate_transfer_matrix(20, rbf='c2', polynomial=False)

    def deform_passenger_nodes(self):
        # Deform passenger nodes using sin^2 + cos^2 partition of unity
        for i in range(self.n_sections - 1):
            for j in range(self.n_pointsPerSection):
                self.passenger_deformations[i, j, :] = (self.deformations[i, j, :] * np.sin(
                    np.pi / 4) ** 2 + self.deformations[i + 1, j, :] * np.cos(np.pi / 4) ** 2)

    def write_passengers(self, fname: str):
        # Write passenger nodes to tecplot format
        f = open(fname, 'w')

        f.write('TITLE = \" Passenger Nodes\" \n')
        f.write('VARIABLES = \"X\" \"Y\" \"Z\" \n')
        f.write('ZONE I= {} J = {} F=point \n'.format(
            self.n_pointsPerSection, self.n_sections - 1))
        for i in range(self.n_sections - 1):
            for j in range(self.n_pointsPerSection):
                f.write('{} \t {} \t {}\n'.format(
                    self.passenger_nodes[i, j, 0], self.passenger_nodes[i, j, 1], self.passenger_nodes[i, j, 2]))

        f.close()

    def write_deformed_passengers(self, fname: str):
        # Write passenger node deformations to tecplot format
        f = open(fname, 'w')

        f.write('TITLE = \" Passenger Nodes Deformed\" \n')
        f.write('VARIABLES = \"X\" \"Y\" \"Z\" \n')
        f.write('ZONE I= {} J = {} F=point \n'.format(
            self.n_pointsPerSection, self.n_sections - 1))
        for i in range(self.n_sections - 1):
            for j in range(self.n_pointsPerSection):
                f.write('{} \t {} \t {}\n'.format(self.passenger_nodes[i, j, 0] + self.passenger_deformations[i, j, 0], self.passenger_nodes[i,
                        j, 1] + self.passenger_deformations[i, j, 1], self.passenger_nodes[i, j, 2] + self.passenger_deformations[i, j, 2]))

        f.close()

    def convert_section_deformations(self, deformations: np.array, section: int):
        # Scale deformations for section- similar process to casting blade, but only scalings and rotations are applied- no offset.
        local_cage_x = deformations[:, 0] * self.chord_i[section]
        local_cage_y = deformations[:, 1] * \
            self.chord_i[section] * self.thick_i[section]
        local_cage_z = np.zeros_like(local_cage_x)

        local_cage_x_twist = local_cage_x * \
            np.cos(self.twist_i[section]) - local_cage_y * \
            np.sin(self.twist_i[section])
        local_cage_y_twist = local_cage_y * \
            np.cos(self.twist_i[section]) + local_cage_x * \
            np.sin(self.twist_i[section])

        # rotate 90 around reference axis
        for j in range(self.n_pointsPerSection):
            self.deformations[section, j, :] = np.c_[
                local_cage_y_twist[j], local_cage_x_twist[j], local_cage_z[j]]

        if self.passengers:
            self.deform_passenger_nodes()

    def write_cage(self, fname: str):
        # Write control cage file to tecplot
        f = open(fname, 'w')

        f.write('TITLE = \" Control Cage\" \n')
        f.write('VARIABLES = \"X\" \"Y\" \"Z\" \n')
        f.write('ZONE I= {} J = {} F=point \n'.format(
            self.n_pointsPerSection, self.n_sections))
        for i in range(self.n_sections):
            for j in range(self.n_pointsPerSection):
                f.write('{} \t {} \t {}\n'.format(
                    self.cage[i, j, 0], self.cage[i, j, 1], self.cage[i, j, 2]))

        f.close()

    def write_control_nodes(self, fname: str):
        f = open(fname, "w")

        for i in range(len(self.control_nodes[:, 0])):
            f.write('{} \t {} \t {}\n'.format(
                self.control_nodes[i, 0], self.control_nodes[i, 1], self.control_nodes[i, 2]))
        f.close()

    def write_deformed(self, fname: str):
        # Write deformed control cage file to tecplot
        f = open(fname, 'w')

        f.write('TITLE = \" Deformed Control Cage\" \n')
        f.write('VARIABLES = \"X\" \"Y\" \"Z\" \n')
        f.write('ZONE I= {} J = {} F=point \n'.format(
            self.n_pointsPerSection, self.n_sections))
        for i in range(self.n_sections):
            for j in range(self.n_pointsPerSection):
                f.write('{} \t {} \t {}\n'.format(self.cage[i, j, 0] + self.deformations[i, j, 0], self.cage[i,
                        j, 1] + self.deformations[i, j, 1], self.cage[i, j, 2] + self.deformations[i, j, 2]))

        f.close()

    def dump_points(self):
        # Dump out control cage points into single array- needed to hook up with rbf coupler
        if self.passengers:
            points_array = np.zeros((self.n_cagePoints + self.n_passengers, 3))
        else:
            points_array = np.zeros((self.n_cagePoints, 3))
        ii = 0

        for i in range(self.n_sections):
            for j in range(self.n_pointsPerSection):
                points_array[ii, :] = self.cage[i, j, :]
                ii += 1

        if self.passengers:
            for i in range(self.n_sections - 1):
                for j in range(self.n_pointsPerSection):
                    points_array[ii, :] = self.passenger_nodes[i, j, :]
                    ii += 1

        points_array = np.r_[points_array, self.control_nodes]

        return points_array

    def dump_deformations(self):
        # Dump out control cage points into single array- needed to hook up with rbf coupler
        if self.passengers:
            points_array = np.zeros((self.n_cagePoints + self.n_passengers, 3))
        else:
            points_array = np.zeros((self.n_cagePoints, 3))
        ii = 0

        for i in range(self.n_sections):
            for j in range(self.n_pointsPerSection):
                points_array[ii, :] = self.deformations[i, j, :]
                ii += 1

        if self.passengers:
            for i in range(self.n_sections - 1):
                for j in range(self.n_pointsPerSection):
                    points_array[ii, :] = self.passenger_deformations[i, j, :]
                    ii += 1

        points_array = np.r_[points_array, np.zeros_like(self.control_nodes)]

        return points_array

    def get_sectional_deformation(self, mode, weight):
        deformed_aerofoil = deform_aerofoil(
            self.modal_data.naca0012, self.modal_data.U, mode, weight)
        deformations = add_z(deformed_aerofoil - self.modal_data.naca0012)
        sectional_deformation = self.rbf_cm.interp_12(deformations)

        return sectional_deformation

    def deform_from_dv(self, def_dict):
        n_modes = def_dict['n_modes']
        sectional_defs = np.zeros((self.n_pointsPerSection, 3))
        for i in range(self.n_sections):
            section_dict = def_dict['section ' + str(i)]
            for j in range(n_modes):
                mode = 'mode ' + str(j)
                # deform scaled cage by mode
                sectional_defs += self.get_sectional_deformation(
                    j, section_dict[mode] * 0.1)

            self.convert_section_deformations(sectional_defs, i)

        surf_def = self.rbf_sc.interp_12(self.dump_deformations())

        return surf_def

    def deformation_list_from_dv(self, def_dict):
        surf_def = self.deform_from_dv(def_dict)
        surf_def = surf_def.flatten()
        surf_def_list = surf_def.tolist()
        return surf_def_list


class Blade_geom:
    """
    This class renders one blade for the rotor.
    """

    def __init__(self, yaml_filename: str):
        """
        The constructor opens the YAML file and extracts the blade
        and airfoil information into instance attributes.

        Parameters
        ----------
        yaml_filename: str
            Filename that contains the geometry for the rotor.
        """
        geometry = yaml.load(open(yaml_filename, "r"), yaml.FullLoader)
        self.outer_shape = geometry["components"]["blade"]["outer_shape_bem"]
        self.airfoils = geometry["airfoils"]


class Blade_surf:
    # Class for reading and handling aerodynamic surface mesh in tecplot format
    def __init__(self, fname):
        self.points = np.loadtxt(fname, skiprows=3)
        self.n_p = len(self.points[:, 0])
        # Currently hard coded- be aware this could cause issues later.
        self.n_i = 161
        self.n_j = 385

    def get_root(self):
        return self.points[0:self.n_i, :]

    def write_surf(self, fname):
        # Write aerodynamic surface back to tecplot format
        f = open(fname, 'w')

        f.write('TITLE = \" IEA_original_surface\" \n')
        f.write('VARIABLES = \"X\" \"Y\" \"Z\" \n')
        f.write('ZONE I= {} J = {} F=point \n'.format(self.n_i, self.n_j))
        for i in range(self.n_sections):
            f.write('{} \t {} \t {}\n'.format(
                self.points[i, 0], self.points[i, 1], self.points[i, 2]))

        f.close()

    def write_deformed(self, fname, deformations):
        # Write deformed aerodynamic surface to tecplot format
        f = open(fname, 'w')

        f.write('TITLE = \" IEA_original_surface\" \n')
        f.write('VARIABLES = \"X\" \"Y\" \"Z\" \n')
        f.write('ZONE I= {} J = {} F=point \n'.format(self.n_i, self.n_j))
        for i in range(self.n_p):
            f.write('{} \t {} \t {}\n'.format(
                self.points[i, 0] + deformations[i, 0], self.points[i, 1] + deformations[i, 1], self.points[i, 2] + deformations[i, 2]))

        f.close()


class design_variables():
    # Class to handle design variables for optimisation problem
    def __init__(self, n_sections, n_modes, twist=False, chord=False) -> None:
        self.n_sections = n_sections
        self.n_modes = n_modes

        self.twist = twist
        self.chord = chord

        self.n_variables = self.n_modes * self.n_sections

        if twist:
            self.n_variables += self.n_sections

        if chord:
            self.n_variables += self.n_sections

        space = []

        for i in range(self.n_sections):
            for j in range(self.n_modes):
                space.append((-1.0, 1.0))
            if self.twist:
                space.append((-np.pi / 4, np.pi / 4))

        self.space = skopt.space.Space(space)

    def sample_space(self, samples_per_variable=10, n_opt_iterations=1000):
        # Sample design space using optimised latin hypercube sampling
        n_samples = samples_per_variable * self.n_variables
        lhs = skopt.sampler.Lhs(criterion='maximin',
                                iterations=n_opt_iterations)
        self.sample_points = lhs.generate(self.space.dimensions, n_samples)

    def get_deformations(self, index):
        # returns a dictionary of deformations to be applied to control cage
        def_dict = {}
        def_dict['n_modes'] = self.n_modes
        def_dict['twist'] = self.twist
        def_dict['chord'] = self.chord
        ii = 0

        for i in range(self.n_sections):
            section = 'section ' + str(i)
            def_dict[section] = {}
            for j in range(self.n_modes):
                def_dict[section]['mode ' +
                                  str(j)] = self.sample_points[index][ii]
                ii += 1
            if self.twist:
                def_dict[section]['twist'] = self.sample_points[index][ii]
                ii += 1

        return def_dict


class Modal_data():
    def __init__(self, data_dir, generate=False, load=False) -> None:
        # data_dir- point this one level above the dump of aerofoil data
        self.data_dir = data_dir
        if generate:
            self.generate_modes()
        if load:
            self.load_modes()

    def generate_modes(self):
        # perform SVD decomposition of aerofoil modes to generate modal data and save for offline use

        # load aerodynamic mode data
        aerofoil_names = glob.glob(self.data_dir + 'Smoothed/' + '*')

        aerofoils = {}

        for n in aerofoil_names:
            name = os.path.basename(n)
            aerofoils[name] = aerofoil_points(n)

        foil_keys = list(aerofoils.keys())

        n_foils = len(aerofoils)
        n_points = len(aerofoils[next(iter(aerofoils))].points[:, 0])
        m_def = int((n_foils * (n_foils - 1)) / 2)

        # create dZ matrix:
        dz = np.zeros([n_points, m_def])
        n = 0

        for i in range(n_foils):
            for j in range(i + 1, n_foils):
                dz[:, n] = aerofoils[foil_keys[i]].points[:, 1] - \
                    aerofoils[foil_keys[j]].points[:, 1]
                n += 1

        # Perform SVD to get mode shapes and energies

        print('Performing SVD')

        self.U, self.S, self.VH = np.linalg.svd(dz, full_matrices=False)

        print('Saving data')

        # Save SVD modal information for later
        f = h5py.File(self.data_dir + 'Modal_data.h5', "w")
        f.create_dataset('U', data=self.U)
        f.create_dataset('S', data=self.S)
        f.create_dataset('VH', data=self.VH)
        f.close()

        self.naca0012 = add_z(np.loadtxt(
            self.data_dir + 'Smoothed/NACA 0012-64.dat', skiprows=3))

    def load_modes(self):
        # Load SVD modes- saves having to rerun SVD each iteration
        f = h5py.File(self.data_dir + 'Modal_data.h5', 'r')
        self.U = np.array(f['U'])
        self.S = np.array(f['S'])
        self.VH = np.array(f['VH'])

        self.naca0012 = add_z(np.loadtxt(
            self.data_dir + 'Smoothed/NACA 0012-64.dat', skiprows=3))

        return


def load_cage_section(cage_path):
    # Load control cage cross section shape
    data = np.loadtxt(cage_path, skiprows=1)
    cage_slice = np.zeros_like(data)
    cage_slice[:, 0] = data[:, 0]
    cage_slice[:, 1] = data[:, 2]
    cage_slice[:, 2] = 0

    return cage_slice


def plot_aerofoil(foil):
    # Plot aerofoil to console
    plt.plot(foil[:, 0], foil[:, 1], 'b.')
    plt.axis('equal')


def plot_cage(cage):
    # Plot aerofoil to console
    fig, ax = plt.subplots()
    ax.plot(cage[:, 0], cage[:, 1], 'w.')
    ax.axis('equal')
    ax.set_title('Control Cage Points', color='white')
    ax.set_xlabel('x')
    ax.set_ylabel('z')

    # setting up X-axis label color to yellow
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')  # setting up Y-axis label color to blue

    # setting up X-axis tick color to red
    ax.tick_params(axis='x', colors='white')
    # setting up Y-axis tick color to black
    ax.tick_params(axis='y', colors='white')

    # setting up Y-axis tick color to red
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['right'].set_color('white')

    fig.savefig('/home/tom/Documents/University/Literature/Note Figures/Cage.png',
                format='png', transparent=True)


def plot_overlay(foil, cage_slice):
    # Plot aerofoil and control cage on same axis
    plt.plot(foil[:, 0], foil[:, 1], 'b.')
    plt.plot(cage_slice[:, 0], cage_slice[:, 1])
    plt.axis('equal')


def plot_deformed(points):
    plt.plot(points[:, 0], points[:, 1], 'r.')
    plt.axis('equal')


def deform_aerofoil(foil, U, mode, weight=1):
    d_points = np.zeros_like(foil)
    d_points[:, 0] = foil[:, 0]
    d_points[:, 1] = foil[:, 1] + weight * U[:, mode]
    d_points[:, 2] = foil[:, 2]
    return d_points


def plot_10_modes(modal_data, n=10):
    fig = plt.figure(figsize=(15, 14), dpi=300, facecolor='w', edgecolor='k')
    foil = modal_data.naca0012
    U = modal_data.U
    for i in range(n):
        ax = fig.add_subplot(int(np.ceil(n / 2)), 2, i + 1)
        d_points = deform_aerofoil(foil, U, i)
        ax.plot(foil[:, 0], foil[:, 1])
        ax.plot(d_points[:, 0], d_points[:, 1])
        ax.axis('equal')
        ax.set_xlabel('x/c', fontsize=20)
        ax.set_ylabel('z', fontsize=20)

        ax.set_title('Mode {}'.format(i), fontsize=20)
    fig.legend(('Naca0012', 'Deformed Naca0012'), loc='upper center',
               ncol=4, fontsize=20, bbox_to_anchor=(0.5, 1.05))

    fig.tight_layout()
    fig.savefig('/home/tom/Documents/University/Latex/SciTech2022OPTPresentation/figs/modes_lite.eps',
                format='eps', bbox_inches='tight')


def plot_10_modes_cage(foil, U, rbf, cage_slice, n=10):
    fig = plt.figure(figsize=(15, 10), dpi=300, facecolor='w', edgecolor='k')
    for i in range(n):
        d_points = np.zeros_like(cage_slice)
        ax = fig.add_subplot(int(np.ceil(n / 2)), 2, i + 1)
        d_points = deform_aerofoil(foil, U, i)
        deformations = d_points - foil
        cage_deformations = rbf.H @ deformations
        ax.plot(foil[:, 0], foil[:, 1])
        ax.plot(d_points[:, 0], d_points[:, 1])
        ax.plot(cage_slice[:, 0], cage_slice[:, 1], 'ro-')
        ax.plot(cage_slice[:, 0] + cage_deformations[:, 0],
                cage_slice[:, 1] + cage_deformations[:, 1], 'bo-')
        ax.axis('equal')
        ax.set_xlabel('x/c', fontsize=20)
        ax.set_ylabel('z', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)

        ax.set_title('Mode {}'.format(i), fontsize=18)
    fig.legend(('Naca0012', 'Deformed Naca0012', 'Control Cage', 'Deformed Control Cage'),
               loc='upper center', ncol=4, fontsize=18, bbox_to_anchor=(0.5, 1.025))
    fig.tight_layout()
    fig.savefig('/home/tom/Documents/University/Latex/Scitech2022_OPT/modes.eps',
                format='eps', bbox_inches='tight')


def plot_process(cage):
    fig = plt.figure(figsize=(15, 15), dpi=300, facecolor='w', edgecolor='k')

    cage_slice = cage.cage_slice
    foil = cage.modal_data.naca0012
    rbf = cage.rbf_cm
    U = cage.modal_data.U

    # Plot naca0012
    d_points = np.zeros_like(cage_slice)
    ax1 = fig.add_subplot(2, 2, 1)
    d_points = deform_aerofoil(foil, U, 0)

    ax1.plot(foil[:, 0], foil[:, 1])
    ax1.plot(cage_slice[:, 0], cage_slice[:, 1], 'ro-')
    ax1.axis('equal')
    ax1.set_xlabel('x/c', fontsize=20)
    ax1.set_ylabel('z', fontsize=20)
    ax1.legend(('Naca0012', 'Control Cage'), fontsize=15)
    ax1.set_title(
        'Couple control cage section to Naca0012 Aerofoil', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=20)

    # Plot cage
    ax2 = fig.add_subplot(2, 2, 2)
    deformations = d_points - foil
    cage_deformations = rbf.H @ deformations

    ax2.plot(d_points[:, 0], d_points[:, 1])
    ax2.plot(cage_slice[:, 0] + cage_deformations[:, 0],
             cage_slice[:, 1] + cage_deformations[:, 1], 'ro-')

    ax2.plot(cage_slice[:, 0], cage_slice[:, 1],
             c='silver', marker='o', ls='-')
    ax2.plot(foil[:, 0], foil[:, 1], c='silver')
    ax2.axis('equal')
    ax2.set_xlabel('x/c', fontsize=20)
    ax2.set_ylabel('z', fontsize=20)
    ax2.legend(('Deformed Naca0012', 'Deformed Control Cage'), fontsize=15)
    ax2.set_title('Deform Naca0012 and interpolate deformations', fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)

    # Plot deformed cage
    ax3 = fig.add_subplot(2, 2, 3)
    theta = np.deg2rad(-10)
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    cage_rot = (cage_slice + cage_deformations) @ rotation_matrix
    cage_rot_or = cage_slice @ rotation_matrix

    SNL_FFA = add_z(np.loadtxt(
        '/home/tom/Documents/University/Coding/zCFD_Utils/data/foils/SNL-FFA-W3-500.dat', skiprows=3))

    SNL_rot = SNL_FFA @ rotation_matrix

    # ax3.plot(cage_slice[:, 0], cage_slice[:, 1], c='silver', marker='o', ls='-')
    ax3.plot(cage_rot_or[:, 0], cage_rot_or[:, 1],
             c='blue', marker='o', ls='-')
    ax3.plot(cage_rot[:, 0], cage_rot[:, 1], 'ro-')
    ax3.plot(SNL_rot[:, 0], SNL_rot[:, 1], c='green')

    # ax3.plot(foil[:, 0], foil[:, 1], c='silver')
    # ax3.plot(d_points[:, 0], d_points[:, 1], c='silver')
    ax3.plot(cage_slice[:, 0] + cage_deformations[:, 0], cage_slice[:,
             1] + cage_deformations[:, 1], c='silver', marker='o', ls='-')
    ax3.axis('equal')
    ax3.set_xlabel('x/c', fontsize=20)
    ax3.set_ylabel('z', fontsize=20)
    ax3.legend(('Fitted Control cage', 'Transformed Control Cage Deformations',
               'Turbine Cross Section'), fontsize=15, loc='upper right')
    ax3.set_title(
        'Transform control cage deformations to turbine axis', fontsize=20)
    ax3.tick_params(axis='both', which='major', labelsize=20)

    # Plot deformed cage
    ax4 = fig.add_subplot(2, 2, 4)

    rbf_ca = py_rbf.UoB_coupling(SNL_rot, cage_slice @ rotation_matrix)
    rbf_ca.generate_transfer_matrix(0.5, 'c2', False)
    cage_deformations_rot = cage_deformations @ rotation_matrix

    surf_def = rbf_ca.H @ cage_deformations_rot
    SNL_def = SNL_rot + surf_def

    ax4.plot(cage_rot[:, 0], cage_rot[:, 1], 'ro-')
    ax4.plot(SNL_def[:, 0], SNL_def[:, 1], c='green')

    ax4.plot(SNL_rot[:, 0], SNL_rot[:, 1], c='silver')
    ax4.plot(cage_rot_or[:, 0], cage_rot_or[:, 1],
             c='silver', marker='o', ls='-')
    ax4.axis('equal')
    ax4.set_xlabel('x/c', fontsize=20)
    ax4.set_ylabel('z', fontsize=20)
    ax4.legend(('Transformed Control Cage Deformations',
               'Deformed Turbine Cross Section'), fontsize=15, loc='upper right')
    ax4.set_title('Deform blade surface', fontsize=20)
    ax4.tick_params(axis='both', which='major', labelsize=20)

    fig.tight_layout()
    fig.savefig('/home/tom/Documents/University/Literature/Note Figures/def_process.png',
                format='png', bbox_inches='tight', background='white')


def load_control_cage(fname):
    cage_nodes = np.loadtxt(fname, skiprows=3)
    return cage_nodes


def add_z(points):
    dim1 = len(points[:, 0])
    points_3D = np.zeros([dim1, 3])
    for i in range(dim1):
        points_3D[i, :] = [points[i, 0], points[i, 1], 0]

    return points_3D


# create classes
blade_props = Blade_geom(
    '/home/tom/Documents/University/Coding/zCFD_Utils/data/IEA-15-240-RWT_FineGrid.yaml')
blade_surf = Blade_surf(
    '/home/tom/Documents/University/Coding/zCFD_Utils/data/IEA_15MW_New_rot.srf.plt')
modal = Modal_data(
    '/home/tom/Documents/University/Coding/zCFD_Utils/data/UIUC Aerofoil Library/', load=True)

# cage_slice = load_cage_section('/home/tom/Documents/University/Coding/zCFD_Utils/data/domain_ordered.ctr.asa.16')

# cage = control_cage(blade_props, blade_surf, modal, 10, cage_slice, passengers=True)
# cage.generate_rbf()

# Optimisation controls
# dv = design_variables(10, 6)
# dv.sample_space()
# deformation_dict = dv.get_deformations(0)

# surf_def = cage.deform_from_dv(deformation_dict)


# # process visualization stuff
# naca0012 = add_z(np.loadtxt('/home/tom/Documents/University/Coding/zCFD_Utils/data/UIUC Aerofoil Library/Smoothed/NACA 0012-64.dat', skiprows=3))
# SNL_FFA = add_z(np.loadtxt('/home/tom/Documents/University/Coding/zCFD_Utils/data/foils/SNL-FFA-W3-500.dat', skiprows=3))


# # Create rotation matrix
# theta = np.deg2rad(-10)
# rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
# SNL_FFA = SNL_FFA @ rotation_matrix
# cage_rot = cage_slice @ rotation_matrix
# Couple control cage with naca0012 profile

# rbf = py_rbf.UoB_coupling(cage_slice, naca0012)
# rbf.generate_transfer_matrix(0.5, 'c2', False)

# # Couple aerodynamic surface to control cage
# rbf_ca = py_rbf.UoB_coupling(SNL_FFA, cage_slice @ rotation_matrix)
# rbf_ca.generate_transfer_matrix(0.5, 'c2', False)

# rbf_cs = py_rbf.UoB_coupling(blade_surf.points, cage.dump_points())
# rbf_cs.generate_transfer_matrix(20, 'c2', False)

# # perturb naca0012 by first mode

# deformed_aerofoil = deform_aerofoil(naca0012, U, 0, weight=0.5)
# deformations = add_z(deformed_aerofoil - naca0012)

# # Interpolate displacements to control cage

# # issue line- check out interp in py_rbf
# a = rbf.H @ deformations.copy()
# a_rot = a @ rotation_matrix

# # Create rbf system between cage and surface
# rbf_ca = py_rbf.UoB_coupling(SNL_FFA, cage_slice)
# rbf_ca.generate_transfer_matrix(0.5, 'c2', False)

# plt.plot(naca0012[:, 0], naca0012[:, 1])
# plt.plot(naca0012[:, 0] + deformations[:, 0], naca0012[:, 1] + deformations[:, 1])
# plt.plot(SNL_FFA[:, 0], SNL_FFA[:, 1])
# plt.plot(cage_rot[:, 0], cage_rot[:, 1], 'ro-')
# plt.plot(cage_rot[:, 0] + a_rot[:, 0], cage_rot[:, 1] + a_rot[:, 1], 'bo-')

# # interpolate deformations
# surf_def = rbf_ca.H @ a
# plt.plot(SNL_FFA[:, 0] + surf_def[:, 0], SNL_FFA[:, 1] + surf_def[:, 1])

# plt.legend(['Naca0102', 'Naca0012 Mode0', 'Control Cage', 'Control Cage Mode0', 'SNL-FFA-W3-500', 'SNL-FFA-W3-500 Mode0'])
# plt.show()

# naca0012 = add_z(np.loadtxt('/home/tom/Documents/University/Coding/zCFD_Utils/data/UIUC Aerofoil Library/Smoothed/NACA 0012-64.dat', skiprows=3))

# for i in range(cage.n_sections):
#     deformed_aerofoil = deform_aerofoil(naca0012, cage.modal_data.U, 2, weight=2)
#     deformations = add_z(deformed_aerofoil - naca0012)

#     a = cage.rbf_cm.H @ deformations.copy()

#     cage.convert_section_deformations(a, i)


# surf_def = cage.rbf_sc.H @ cage.dump_deformations()

# Write out deformations
# blade_surf.write_deformed('/home/tom/Documents/University/Coding/zCFD_Utils/data/IEA_15MW_New_rot_deformed.srf.plt', surf_def)
# cage.write_cage('/home/tom/Documents/University/Coding/zCFD_Utils/data/cage.plt')
# cage.write_deformed('../data/deformed_cage.plt')
# cage.write_control_nodes('/home/tom/Documents/University/Coding/zCFD_Utils/data/control_nodes.dat')


# dv = design_variables(10, 6)
# dv.sample_space()
# deformation_dict = dv.get_deformations(0)

# surf_def = cage.deform_from_dv(deformation_dict)
# blade_surf.write_deformed('/home/tom/Documents/University/Coding/zCFD_Utils/data/IEA_15MW_New_rot_deformed.srf.plt', surf_def)
# cage.write_cage('/home/tom/Documents/University/Coding/zCFD_Utils/data/cage.plt')
# cage.write_deformed('../data/deformed_cage.plt')
