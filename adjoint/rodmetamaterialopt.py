"""
Copyright (C) 2025 Zhenyu Zhao.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import scipy.optimize
import numpy as np
import os
import scipy.interpolate
import scipy.constants
import scipy.integrate

class RodMetaMaterialRegion2D:
    def __init__(self,fdtd_engine, rod_region, x_mesh = 0.02, y_mesh = 0.02, z_mesh = 0.02, z_start = -0.11, z_end = 0.11, rename = "RodMetaMaterialRegion" ):
        self.left_down_point = rod_region.left_down_point
        self.right_up_point= rod_region.right_up_point
        self.__last_params = None
        self.__lastest_params = None
        self.fdtd_engine = fdtd_engine
        self.rod_region = rod_region
        self.x_mesh = x_mesh
        self.y_mesh = y_mesh
        self.z_mesh = z_mesh
        self.x_min = self.left_down_point.x
        self.x_max = self.right_up_point.x
        self.y_min = self.left_down_point.y
        self.y_max = self.right_up_point.y
        self.z_min = z_start
        self.z_max = z_end
        self.x_size = int((self.x_max - self.x_min) / self.x_mesh) + 1
        self.y_size = int((self.y_max - self.y_min) / self.y_mesh) + 1
        self.z_size = int((self.z_max - self.z_min) / self.z_mesh) + 1
        self.x_positions = np.linspace(self.x_min, self.x_max, self.x_size)
        self.y_positions = np.linspace(self.y_min, self.y_max, self.y_size)
        self.z_positions = np.linspace(self.z_min, self.z_max, self.z_size)
        self.z_start = z_start
        self.z_end = z_end
        self.rename = rename
        self.index_region_name = self.rename + "_index"
        self.field_region_name = self.rename + "_field"
        self.__initialize()
        self.epsilon_figure = None
        self.field_figure = None

    def __initialize(self):

        self.fdtd_engine.add_index_region(self.left_down_point, self.right_up_point, z_min=self.z_min, z_max=self.z_max,
                                          dimension=2, index_monitor_name=self.index_region_name)
        # self.fdtd_engine.fdtd.eval(
        #     'select("{}");set("spatial interpolation","specified position");'.format(self.index_region_name))
        self.fdtd_engine.add_field_region(self.left_down_point, self.right_up_point, z_min=self.z_min, z_max=self.z_max,
                                          dimension=2, field_monitor_name=self.field_region_name)
        # self.fdtd_engine.fdtd.eval(
        #     'select("{}");set("spatial interpolation","specified position");'.format(self.field_region_name))
        self.fdtd_engine.add_mesh_region(self.left_down_point, self.right_up_point, x_mesh=self.x_mesh,
                                         y_mesh=self.y_mesh,
                                         z_mesh=self.z_mesh, z_min=self.z_min, z_max=self.z_max)

        self.fdtd_engine.fdtd.eval('select("FDTD");')
        self.fdtd_engine.fdtd.set('use legacy conformal interface detection', False)
        self.fdtd_engine.fdtd.set('conformal meshing refinement', 51)
        self.fdtd_engine.fdtd.set('meshing tolerance', 1.0 / 1.134e14)

    def update(self, params):
        self.rod_region.update(params)

    def get_E_distribution(self, if_get_spatial = 0):
        if (if_get_spatial == 0):
            self.field_figure = self.fdtd_engine.get_E_distribution(field_monitor_name = self.field_region_name, if_get_spatial = if_get_spatial)
            return self.field_figure
        else:
            return self.fdtd_engine.get_E_distribution(field_monitor_name = self.field_region_name, if_get_spatial = if_get_spatial)

    def get_E_distribution_in_CAD(self, data_name):
        self.fdtd_engine.get_E_distribution_in_CAD( field_monitor_name = self.field_region_name ,data_name = data_name)
        return data_name

    def get_epsilon_distribution_in_CAD(self, data_name):
        self.fdtd_engine.get_epsilon_distribution_in_CAD( index_monitor_name = self.index_region_name, data_name = data_name)
        return data_name

    def get_epsilon_distribution(self):
        return self.fdtd_engine.get_epsilon_distribution(index_monitor_name=self.index_region_name)

    def plot_epsilon_figure(self, filename = None):
        epsilon = np.real(np.mean(self.epsilon_figure[:,:,int(self.z_size/2),:] if type(self.epsilon_figure)!=type(None) else self.get_epsilon_distribution()[:,:,int(self.z_size/2),:], axis=-1))
        xx, yy = np.meshgrid(np.linspace(self.x_positions[0], self.x_positions[-1], epsilon.shape[0]),
                                         np.linspace(self.y_positions[0], self.y_positions[-1], epsilon.shape[1]))
        import matplotlib.pyplot as plt
        bar = plt.pcolormesh(xx, yy, epsilon.T , cmap="gray")
        plt.colorbar(bar)
        plt.xlabel('x (μm)')
        plt.ylabel('y (μm)')
        if (type(filename) != type(None)):
            if (filename[0:2] == './'):
                filepath = os.path.abspath('./') + '/' + filename[2:]
                filedir = os.path.split(filepath)[0]
                if not os.path.isdir(filedir):
                    os.makedirs(filedir)
                plt.savefig(filepath)
            elif (filename[0:3] == '../'):
                filepath = os.path.abspath('../') + '/' + filename[3:]
                filedir = os.path.split(filepath)[0]
                if not os.path.isdir(filedir):
                    os.makedirs(filedir)
                plt.savefig(filepath)
            else:
                plt.savefig(filename)
        plt.close()
        # plt.show()



    def plot_field_figure(self, filename = None):
        if type(self.field_figure) != type(None):
            field = np.abs(np.mean(self.field_figure[:, :,int(self.z_size/2), 0, :], axis=2))
        else:
            raise Exception("No field stored in the reiogn.")
        xx, yy = np.meshgrid(np.linspace(self.x_positions[0], self.x_positions[-1], field.shape[0]),
                             np.linspace(self.y_positions[0], self.y_positions[-1], field.shape[1]))
        import matplotlib.pyplot as plt
        bar = plt.pcolormesh(xx, yy, field.T, cmap="jet")
        plt.colorbar(bar)
        plt.xlabel('x (μm)')
        plt.ylabel('y (μm)')
        if (type(filename) != type(None)):
            if (filename[0:2] == './'):
                filepath = os.path.abspath('./') + '/' + filename[2:]
                filedir = os.path.split(filepath)[0]
                if not os.path.isdir(filedir):
                    os.makedirs(filedir)
                plt.savefig(filepath)
            elif (filename[0:3] == '../'):
                filepath = os.path.abspath('../') + '/' + filename[3:]
                filedir = os.path.split(filepath)[0]
                if not os.path.isdir(filedir):
                    os.makedirs(filedir)
                plt.savefig(filepath)
            else:
                plt.savefig(filename)
        plt.close()
        # plt.show()

class RodMetaMaterialRegion3D:
    def __init__(self,fdtd_engine, rod_region, x_mesh = 0.02, y_mesh = 0.02, z_mesh = 0.02, z_start = -0.11, z_end = 0.11, rename = "RodMetaMaterialRegion" ):
        self.left_down_point = rod_region.left_down_point
        self.right_up_point= rod_region.right_up_point
        self.__last_params = None
        self.__lastest_params = None
        self.fdtd_engine = fdtd_engine
        self.rod_region = rod_region
        self.x_mesh = x_mesh
        self.y_mesh = y_mesh
        self.z_mesh = z_mesh
        self.x_min = self.left_down_point.x
        self.x_max = self.right_up_point.x
        self.y_min = self.left_down_point.y
        self.y_max = self.right_up_point.y
        self.z_min = z_start
        self.z_max = z_end
        self.x_size = int((self.x_max - self.x_min) / self.x_mesh) + 1
        self.y_size = int((self.y_max - self.y_min) / self.y_mesh) + 1
        self.z_size = int((self.z_max - self.z_min) / self.z_mesh) + 1
        self.x_positions = np.linspace(self.x_min, self.x_max, self.x_size)
        self.y_positions = np.linspace(self.y_min, self.y_max, self.y_size)
        self.z_positions = np.linspace(self.z_min, self.z_max, self.z_size)
        self.z_start = z_start
        self.z_end = z_end
        self.rename = rename
        self.index_region_name = self.rename + "_index"
        self.field_region_name = self.rename + "_field"
        self.__initialize()
        self.epsilon_figure = None
        self.field_figure = None

    def __initialize(self):

        self.fdtd_engine.add_index_region(self.left_down_point, self.right_up_point, z_min=self.z_min, z_max=self.z_max,
                                          dimension=3, index_monitor_name=self.index_region_name)
        # self.fdtd_engine.fdtd.eval(
        #     'select("{}");set("spatial interpolation","specified position");'.format(self.index_region_name))
        self.fdtd_engine.add_field_region(self.left_down_point, self.right_up_point, z_min=self.z_min, z_max=self.z_max,
                                          dimension=3, field_monitor_name=self.field_region_name)
        # self.fdtd_engine.fdtd.eval(
        #     'select("{}");set("spatial interpolation","specified position");'.format(self.field_region_name))
        self.fdtd_engine.add_mesh_region(self.left_down_point, self.right_up_point, x_mesh=self.x_mesh,
                                         y_mesh=self.y_mesh,
                                         z_mesh=self.z_mesh, z_min=self.z_min, z_max=self.z_max)

        self.fdtd_engine.fdtd.eval('select("FDTD");')
        self.fdtd_engine.fdtd.set('use legacy conformal interface detection', False)
        self.fdtd_engine.fdtd.set('conformal meshing refinement', 51)
        self.fdtd_engine.fdtd.set('meshing tolerance', 1.0 / 1.134e14)

    def update(self, params):
        self.rod_region.update(params)

    def get_E_distribution(self, if_get_spatial = 0):
        if (if_get_spatial == 0):
            self.field_figure = self.fdtd_engine.get_E_distribution(field_monitor_name = self.field_region_name, if_get_spatial = if_get_spatial)
            return self.field_figure
        else:
            return self.fdtd_engine.get_E_distribution(field_monitor_name = self.field_region_name, if_get_spatial = if_get_spatial)

    def get_E_distribution_in_CAD(self, data_name):
        self.fdtd_engine.get_E_distribution_in_CAD( field_monitor_name = self.field_region_name ,data_name = data_name)
        return data_name

    def get_epsilon_distribution_in_CAD(self, data_name):
        self.fdtd_engine.get_epsilon_distribution_in_CAD( index_monitor_name = self.index_region_name, data_name = data_name)
        return data_name

    def get_epsilon_distribution(self):
        return self.fdtd_engine.get_epsilon_distribution(index_monitor_name=self.index_region_name)

    def plot_epsilon_figure(self, filename = None):
        epsilon = np.real(np.mean(self.epsilon_figure[:,:,int(self.z_size/2),:] if type(self.epsilon_figure)!=type(None) else self.get_epsilon_distribution()[:,:,int(self.z_size/2),:], axis=-1))
        xx, yy = np.meshgrid(np.linspace(self.x_positions[0], self.x_positions[-1], epsilon.shape[0]),
                                         np.linspace(self.y_positions[0], self.y_positions[-1], epsilon.shape[1]))
        import matplotlib.pyplot as plt
        bar = plt.pcolormesh(xx, yy, epsilon.T , cmap="gray")
        plt.colorbar(bar)
        plt.xlabel('x (μm)')
        plt.ylabel('y (μm)')
        if (type(filename) != type(None)):
            if (filename[0:2] == './'):
                filepath = os.path.abspath('./') + '/' + filename[2:]
                filedir = os.path.split(filepath)[0]
                if not os.path.isdir(filedir):
                    os.makedirs(filedir)
                plt.savefig(filepath)
            elif (filename[0:3] == '../'):
                filepath = os.path.abspath('../') + '/' + filename[3:]
                filedir = os.path.split(filepath)[0]
                if not os.path.isdir(filedir):
                    os.makedirs(filedir)
                plt.savefig(filepath)
            else:
                plt.savefig(filename)
        plt.close()
        # plt.show()



    def plot_field_figure(self, filename = None):
        if type(self.field_figure) != type(None):
            field = np.abs(np.mean(self.field_figure[:, :,int(self.z_size/2), 0, :], axis=2))
        else:
            raise Exception("No field stored in the reiogn.")
        xx, yy = np.meshgrid(np.linspace(self.x_positions[0], self.x_positions[-1], field.shape[0]),
                             np.linspace(self.y_positions[0], self.y_positions[-1], field.shape[1]))
        import matplotlib.pyplot as plt
        bar = plt.pcolormesh(xx, yy, field.T, cmap="jet")
        plt.colorbar(bar)
        plt.xlabel('x (μm)')
        plt.ylabel('y (μm)')
        if (type(filename) != type(None)):
            if (filename[0:2] == './'):
                filepath = os.path.abspath('./') + '/' + filename[2:]
                filedir = os.path.split(filepath)[0]
                if not os.path.isdir(filedir):
                    os.makedirs(filedir)
                plt.savefig(filepath)
            elif (filename[0:3] == '../'):
                filepath = os.path.abspath('../') + '/' + filename[3:]
                filedir = os.path.split(filepath)[0]
                if not os.path.isdir(filedir):
                    os.makedirs(filedir)
                plt.savefig(filepath)
            else:
                plt.savefig(filename)
        plt.close()
        # plt.show()


class AdjointForRodMetaMaterialRegion:
    def __init__(self, fdtd_engine, T_monitor_names, target_T, design_region, forward_source_names, backward_source_names, index_origin = 3.478, index_pert = 1.444, sim_name = "Adjoint", record_forward_field = 1, if_default_fom = 1, TE_fraction = 1):
        self.fdtd_engine = fdtd_engine
        self.design_region = design_region
        self.T_monitor_names = np.array([T_monitor_names]).flatten()
        self.target_T = np.reshape(np.array([target_T]), (np.shape(self.T_monitor_names)[0],-1))
        self.forward_source_names = np.array([forward_source_names]).flatten()
        self.backward_source_names = np.array([backward_source_names]).flatten()
        self.epsilon_origin = np.power(index_origin,2)
        self.epsilon_pert = np.power(index_pert,2)
        self.sim_name = sim_name
        self.record_forward_field = record_forward_field
        self.if_default_fom = if_default_fom
        self.TE_fraction = TE_fraction

        if (type(self.design_region) == RodMetaMaterialRegion2D):
            self.design_region_type = '2d'
        elif ((type(self.design_region) == RodMetaMaterialRegion3D)):
            self.design_region_type = '3d'
        else:
            raise Exception("The type of design_region should be RodMetaMaterialRegion2D or RodMetaMaterialRegion3D.")

        # calculate parameters for index
        self.__cal_index()

    def __cal_index(self):
        # for each parameter
        x_blocks_num = self.design_region.rod_region.matrix_mask.shape[0]
        y_blocks_num = self.design_region.rod_region.matrix_mask.shape[1]
        self.x_block_length = np.abs(self.design_region.left_down_point.x - self.design_region.right_up_point.x) / x_blocks_num
        self.y_block_length = np.abs(self.design_region.left_down_point.y - self.design_region.right_up_point.y) / y_blocks_num
        self.blocks_x_min_index = []
        self.blocks_x_max_index = []
        self.blocks_y_min_index = []
        self.blocks_y_max_index = []

        for i in range(0, x_blocks_num * y_blocks_num):
            x_index_for_blocks = i % x_blocks_num
            y_index_for_blocks = int(i / x_blocks_num)
            if (self.design_region.rod_region.matrix_mask[x_index_for_blocks, y_index_for_blocks] == 1):
                # calculate center
                x_relative_center = self.x_block_length / 2 + x_index_for_blocks * self.x_block_length
                y_relative_center = self.y_block_length * y_blocks_num - self.y_block_length / 2 - y_index_for_blocks * self.y_block_length
                # calculate min boundary and append
                self.blocks_x_min_index.append(
                    int((x_relative_center - self.x_block_length / 2) / self.design_region.x_mesh))
                # calculate max boundary and append
                self.blocks_x_max_index.append(
                    int((x_relative_center + self.x_block_length / 2) / self.design_region.x_mesh))
                # calculate min boundary and append
                self.blocks_y_min_index.append(
                    int((y_relative_center - self.y_block_length / 2) / self.design_region.y_mesh))
                # calculate max boundary and append
                self.blocks_y_max_index.append(
                    int((y_relative_center + self.y_block_length / 2) / self.design_region.y_mesh))

    def get_total_source_power(self, source_names):
        total_source_power = self.fdtd_engine.get_source_power(source_names[0])
        for i in range(1, np.shape(source_names)[0]):
            total_source_power += self.fdtd_engine.get_source_power(source_names[i])
        return total_source_power

    def get_forward_transmission_properties(self):
        mode_coefficients = []
        for i in range(0, np.shape(self.T_monitor_names)[0]):
            mode_coefficients.append(self.fdtd_engine.get_mode_coefficient(expansion_name=str(self.T_monitor_names[i])))
        mode_coefficients = np.array(mode_coefficients)
        forward_source_power = self.get_total_source_power(self.forward_source_names)
        self.T_fwd_vs_wavelengths = np.real(mode_coefficients * mode_coefficients.conj() / forward_source_power)
        self.phase_prefactors = mode_coefficients / 4.0 / forward_source_power

    def call_fom(self, params):
        """
         Calculate FoM(Figure of Merit) and return.
         (Reference: lumopt. https://github.com/chriskeraly/lumopt)
        """
        self.fdtd_engine.switch_to_layout()
        self.fdtd_engine.set_enable(self.forward_source_names.tolist())
        self.fdtd_engine.set_disable(self.backward_source_names.tolist())
        self.design_region.update(params)
        self.fdtd_engine.run()
        if (self.record_forward_field):
            self.forward_field = self.design_region.get_E_distribution()
        self.forward_field = self.design_region.get_E_distribution()

        self.get_forward_transmission_properties()

        if (self.if_default_fom == 1):
            wavelength = self.fdtd_engine.get_wavelength()
            wavelength_range = wavelength.max() - wavelength.min()
            if (wavelength.size > 1):
                T_fwd_integrand = np.abs(self.target_T) / wavelength_range
                const_term = np.trapz(y=T_fwd_integrand, x=wavelength, axis=1)
                T_fwd_error = np.abs(np.squeeze(self.T_fwd_vs_wavelengths) - self.target_T)
                T_fwd_error_integrand = T_fwd_error / wavelength_range
                error_term = np.trapz(y=T_fwd_error_integrand, x=wavelength, axis=1)
                self.fom = const_term - error_term
            else:
                self.fom = np.abs(self.target_T) - np.abs(self.T_fwd_vs_wavelengths.flatten() - self.target_T)
        else:
            wavelength = self.fdtd_engine.get_wavelength()
            wavelength_range = wavelength.max() - wavelength.min()
            if (wavelength.size > 1):
                self.fom = np.trapz(y=np.squeeze(self.T_fwd_vs_wavelengths), x=wavelength, axis=1)/ wavelength_range
            else:
                self.fom = self.T_fwd_vs_wavelengths.flatten()
        return self.fom

    def call_grad(self, params):
        """
        Calculate gradient and return.
        (Reference: lumopt. https://github.com/chriskeraly/lumopt)
        """
        if (self.design_region_type == '3d'):
            cell = self.design_region.x_mesh * 1e-6 * self.design_region.y_mesh * 1e-6 * self.design_region.z_mesh * 1e-6
        else:
            cell = self.design_region.x_mesh * 1e-6 * self.design_region.y_mesh * 1e-6
        self.fdtd_engine.switch_to_layout()
        self.fdtd_engine.set_disable(self.forward_source_names.tolist())
        self.fdtd_engine.set_disable(self.backward_source_names.tolist())
        omega = self.fdtd_engine.get_omega()
        wavelength = self.fdtd_engine.get_wavelength()
        wavelength_range = wavelength.max() - wavelength.min()

        d = np.diff(wavelength)
        T_fwd_partial_derivs = []
        for i in range(0, np.shape(self.backward_source_names)[0]):
            self.fdtd_engine.set_enable(self.backward_source_names[i])
            self.fdtd_engine.run()
            adjoint_source_power = self.fdtd_engine.get_source_power(self.backward_source_names[i])
            scaling_factor = np.conj(self.phase_prefactors[i]) * omega * 1j / np.sqrt(adjoint_source_power)
            adjoint_field = self.design_region.get_E_distribution()
            gradient_field = np.mean(np.sum(2.0 * cell * scipy.constants.epsilon_0 * self.forward_field * adjoint_field, axis=4), axis=2)
            averaged_gradient_field = np.zeros((gradient_field.shape[2], len(self.blocks_x_min_index)),
                                               dtype='complex_')
            for j in range(0, len(self.blocks_x_min_index)):
                x_averaged_gradient_field = np.mean(
                    gradient_field[self.blocks_x_min_index[j]:self.blocks_x_max_index[j], :, :], axis=0)
                averaged_gradient_field[:, j] = np.mean(
                    x_averaged_gradient_field[self.blocks_y_min_index[j]:self.blocks_y_max_index[j], :], axis=0)
            for wl in range(0, len(omega)):
                averaged_gradient_field[wl, :] = averaged_gradient_field[wl, :] * scaling_factor[wl]
            self.fdtd_engine.switch_to_layout()
            self.fdtd_engine.set_disable(self.backward_source_names[i])

            if (self.if_default_fom == 1):
                if (wavelength.size > 1):
                    T_fwd_error = self.T_fwd_vs_wavelengths[i] - self.target_T[i]
                    T_fwd_error_integrand = np.abs(T_fwd_error) / wavelength_range
                    const_factor = -1.0 * np.trapz(y = T_fwd_error_integrand, x = wavelength)
                    integral_kernel = np.sign(T_fwd_error) / wavelength_range
                    quad_weight = np.append(np.append(d[0], d[0:-1] + d[1:]),
                                            d[-1]) / 2
                    v = const_factor * integral_kernel.flatten() * quad_weight
                    T_fwd_partial_derivs.append(np.real(averaged_gradient_field).transpose().dot(v).flatten().real)
                else:
                    T_fwd_partial_derivs.append((-1.0*np.sign(self.T_fwd_vs_wavelengths[i] - self.target_T[i]) * np.real(averaged_gradient_field)).real)

            else:
                T_fwd_partial_derivs.append(np.real(averaged_gradient_field))

        if (self.if_default_fom == 1):
            T_fwd_partial_derivs = np.sum(np.array(T_fwd_partial_derivs), axis=0)
        else:
            T_fwd_partial_derivs = np.array(T_fwd_partial_derivs)

        if (self.design_region_type == '3d'):
            # convert delta epsilon to partial
            last_radius = self.design_region.rod_region.pixel_radius * params
            last_radius = np.clip(last_radius, 0.001, self.design_region.rod_region.pixel_radius)
            volume = self.x_block_length * self.y_block_length * (self.design_region.z_end - self.design_region.z_start)
            height = (self.design_region.rod_region.z_end - self.design_region.rod_region.z_start)
            epsilon_fraction = (self.epsilon_pert - self.epsilon_origin) / self.epsilon_pert
            volume_fraction = np.pi * height / volume
            T_fwd_partial_derivs = np.power(
                1 - self.TE_fraction * np.power(last_radius, 2) * volume_fraction * epsilon_fraction, 2) / \
                (2 * last_radius * (1 - self.TE_fraction + self.TE_fraction * self.epsilon_pert
                ) * volume_fraction * epsilon_fraction * self.design_region.rod_region.pixel_radius) * T_fwd_partial_derivs
        else:
            # convert delta epsilon to partial
            last_radius = self.design_region.rod_region.pixel_radius * params
            last_radius = np.clip(last_radius, 0.001, self.design_region.rod_region.pixel_radius)
            T_fwd_partial_derivs = 2*np.pi*last_radius * (self.epsilon_pert - self.epsilon_origin) / (
                    self.x_block_length * self.y_block_length) * T_fwd_partial_derivs

        # convert grad to params element
        return T_fwd_partial_derivs

    def reset_monitor_names(self, monitor_names):
        self.T_monitor_names = np.array([monitor_names]).flatten()
    def reset_forward_source_names(self, source_names):
        self.forward_source_names = np.array([source_names]).flatten()
    def reset_backward_source_names(self, source_names):
        self.backward_source_names = np.array([source_names]).flatten()
    def reset_target_T(self, target_T):
        self.target_T = np.reshape(np.array([target_T]), (np.shape(self.T_monitor_names)[0], -1))