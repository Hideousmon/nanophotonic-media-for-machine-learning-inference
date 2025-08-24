# Licensed under the GPLv3 License. See LICENSE file for details.
# Copyright (c) 2025 ZhenyuZhao

import numpy as np
import os


class RodMetaMaterialRegion2D:
    def __init__(self, fdtd_engine, rod_region, x_mesh=0.02, y_mesh=0.02, z_mesh=0.02, z_start=-0.11, z_end=0.11,
                 rename="RodMetaMaterialRegion"):
        self.left_down_point = rod_region.left_down_point
        self.right_up_point = rod_region.right_up_point
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
        self.fdtd_engine.add_field_region(self.left_down_point, self.right_up_point, z_min=self.z_min, z_max=self.z_max,
                                          dimension=2, field_monitor_name=self.field_region_name)
        self.fdtd_engine.add_mesh_region(self.left_down_point, self.right_up_point, x_mesh=self.x_mesh,
                                         y_mesh=self.y_mesh,
                                         z_mesh=self.z_mesh, z_min=self.z_min, z_max=self.z_max)

        self.fdtd_engine.fdtd.eval('select("FDTD");')
        self.fdtd_engine.fdtd.set('use legacy conformal interface detection', False)
        self.fdtd_engine.fdtd.set('conformal meshing refinement', 51)
        self.fdtd_engine.fdtd.set('meshing tolerance', 1.0 / 1.134e14)

    def update(self, params):
        self.rod_region.update(params)

    def get_E_distribution(self, if_get_spatial=0):
        if (if_get_spatial == 0):
            self.field_figure = self.fdtd_engine.get_E_distribution(field_monitor_name=self.field_region_name,
                                                                    if_get_spatial=if_get_spatial)
            return self.field_figure
        else:
            return self.fdtd_engine.get_E_distribution(field_monitor_name=self.field_region_name,
                                                       if_get_spatial=if_get_spatial)

    def get_E_distribution_in_CAD(self, data_name):
        self.fdtd_engine.get_E_distribution_in_CAD(field_monitor_name=self.field_region_name, data_name=data_name)
        return data_name

    def get_epsilon_distribution_in_CAD(self, data_name):
        self.fdtd_engine.get_epsilon_distribution_in_CAD(index_monitor_name=self.index_region_name, data_name=data_name)
        return data_name

    def get_epsilon_distribution(self):
        return self.fdtd_engine.get_epsilon_distribution(index_monitor_name=self.index_region_name)

    def plot_epsilon_figure(self, filename=None):
        epsilon = np.real(np.mean(
            self.epsilon_figure[:, :, int(self.z_size / 2), :] if type(self.epsilon_figure) != type(
                None) else self.get_epsilon_distribution()[:, :, int(self.z_size / 2), :], axis=-1))
        xx, yy = np.meshgrid(np.linspace(self.x_positions[0], self.x_positions[-1], epsilon.shape[0]),
                             np.linspace(self.y_positions[0], self.y_positions[-1], epsilon.shape[1]))
        import matplotlib.pyplot as plt
        bar = plt.pcolormesh(xx, yy, epsilon.T, cmap="gray")
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

    def plot_field_figure(self, filename=None):
        if type(self.field_figure) != type(None):
            field = np.abs(np.mean(self.field_figure[:, :, int(self.z_size / 2), 0, :], axis=2))
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


class RodMetaMaterialRegion3D:
    def __init__(self, fdtd_engine, rod_region, x_mesh=0.02, y_mesh=0.02, z_mesh=0.02, z_start=-0.11, z_end=0.11,
                 rename="RodMetaMaterialRegion"):
        self.left_down_point = rod_region.left_down_point
        self.right_up_point = rod_region.right_up_point
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

    def get_E_distribution(self, if_get_spatial=0):
        if (if_get_spatial == 0):
            self.field_figure = self.fdtd_engine.get_E_distribution(field_monitor_name=self.field_region_name,
                                                                    if_get_spatial=if_get_spatial)
            return self.field_figure
        else:
            return self.fdtd_engine.get_E_distribution(field_monitor_name=self.field_region_name,
                                                       if_get_spatial=if_get_spatial)

    def get_E_distribution_in_CAD(self, data_name):
        self.fdtd_engine.get_E_distribution_in_CAD(field_monitor_name=self.field_region_name, data_name=data_name)
        return data_name

    def get_epsilon_distribution_in_CAD(self, data_name):
        self.fdtd_engine.get_epsilon_distribution_in_CAD(index_monitor_name=self.index_region_name, data_name=data_name)
        return data_name

    def get_epsilon_distribution(self):
        return self.fdtd_engine.get_epsilon_distribution(index_monitor_name=self.index_region_name)

    def plot_epsilon_figure(self, filename=None):
        epsilon = np.real(np.mean(
            self.epsilon_figure[:, :, int(self.z_size / 2), :] if type(self.epsilon_figure) != type(
                None) else self.get_epsilon_distribution()[:, :, int(self.z_size / 2), :], axis=-1))
        xx, yy = np.meshgrid(np.linspace(self.x_positions[0], self.x_positions[-1], epsilon.shape[0]),
                             np.linspace(self.y_positions[0], self.y_positions[-1], epsilon.shape[1]))
        import matplotlib.pyplot as plt
        bar = plt.pcolormesh(xx, yy, epsilon.T, cmap="gray")
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

    def plot_field_figure(self, filename=None):
        if type(self.field_figure) != type(None):
            field = np.abs(np.mean(self.field_figure[:, :, int(self.z_size / 2), 0, :], axis=2))
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
