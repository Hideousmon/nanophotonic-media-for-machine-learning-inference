# Licensed under the GPLv3 License. See LICENSE file for details.
# Copyright (c) 2025 ZhenyuZhao

import numpy as np
from .rodregion import RodMetaMaterialRegion2D, RodMetaMaterialRegion3D
import scipy.constants
from .backend import get_backend, get_torch_device, get_library_name


class AdjointForRodComplexdirect:
    def __init__(self, fdtd_engine, T_monitor_names, design_region, forward_source_names, backward_source_names,
                 index_origin = 3.478, index_pert = 1.444, sim_name = "Adjoint",
                 em_grad_normalization = 0, backward_T_monitor_names = None, simulation_time = 5000,
                 eps_diff_func = None, reload_flag=False):
        self.fdtd_engine = fdtd_engine
        self.design_region = design_region
        self.T_monitor_names = np.array([T_monitor_names]).flatten()
        self.forward_source_names = np.array([forward_source_names]).flatten()
        self.backward_source_names = np.array([backward_source_names]).flatten()
        self.epsilon_origin = np.power(index_origin,2)
        self.epsilon_pert = np.power(index_pert,2)
        self.sim_name = sim_name
        self.original_epsilon_distribution = None
        self.eps_diff = None
        self.forward_params = None
        self.dx = 0.001/self.design_region.rod_region.pixel_radius
        self.simulation_time = simulation_time
        self.eps_diff_func = eps_diff_func
        self.reload_flag = reload_flag
        if backward_T_monitor_names is None:
            self.backward_T_monitor_names = []
        else:
            self.backward_T_monitor_names = np.array([backward_T_monitor_names]).flatten()
        self.em_grad_normalization = em_grad_normalization

        if type(self.design_region) == RodMetaMaterialRegion2D:
            self.design_region_type = '2d'
        elif type(self.design_region) == RodMetaMaterialRegion3D:
            self.design_region_type = '3d'
        else:
            raise Exception("The type of design_region should be RodMetaMaterialRegion2D or RodMetaMaterialRegion3D.")

        # calculate parameters for index
        self.__cal_index()

        self.backend = get_backend()
        self.library_name = get_library_name()
        self.torch_device_name = get_torch_device()
        self.device = None

        if self.library_name == "jax":
            self.forward_func = self.get_forward_function_jax()
        elif self.library_name == "torch":
            self.forward_func = self.get_forward_function_torch()
            self.device = self.backend.device(self.torch_device_name)
        else:
            raise Exception("Unknown backend.")

    def get_forward_function_jax(self):
        @self.backend.custom_vjp
        def forward(params):
            r, radian = self.forward_sim(np.array(params, dtype=np.float64))
            return r, radian

        def forward_solver_fwd(params):
            return self.forward_sim(params), (params,)

        def forward_solver_bwd(res, g):
            params = res
            grad = self.backend.numpy.array(self.backward_sim(g, params))
            grad = self.backend.numpy.sum(grad, axis=(1, 2))

            if self.em_grad_normalization:
                grad = self.backend.numpy.array(self.normalization(grad))

            return (grad,)

        forward.defvjp(forward_solver_fwd, forward_solver_bwd)

        return forward

    def get_forward_function_torch(self):
        class MyCustomFunction(self.backend.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                r, radian = self.backend.tensor(self.forward_sim(np.array(input.detach().cpu().numpy(), dtype=np.float64)),
                                          device=self.device, dtype=self.backend.float64)
                return r, radian

            @staticmethod
            def backward(ctx, grad_output_r, grad_output_radian):

                input, = ctx.saved_tensors
                grad = np.array(self.backward_sim((grad_output_r.detach().cpu().numpy(),
                                                   grad_output_radian.detach().cpu().numpy()), input.detach().cpu().numpy()))
                grad = np.sum(grad, axis=(1, 2))

                if self.em_grad_normalization:
                    grad = np.array(self.normalization(grad))

                grad = self.backend.tensor(grad, device=self.device, dtype=self.backend.float64)
                return (grad,)
        forward = MyCustomFunction.apply

        return forward

    @staticmethod
    def normalization(grad):
        np_grad = np.array(grad)
        grad_l2_norm = np.linalg.norm(np_grad)
        if grad_l2_norm != 0:
            return np_grad / grad_l2_norm
        else:
            return np_grad

    def forward(self, params):
        if self.library_name == "jax":
            r, radian = self.forward_func(params)
            return r*self.backend.numpy.exp(1j*radian)
        elif self.library_name == "torch":
            r, radian = self.forward_func(params)
            return r * self.backend.exp(1j * radian)
        else:
            raise Exception("Unknown backend.")

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
        transmission_coefficient_set = []
        N_set = []
        for i in range(0, np.shape(self.T_monitor_names)[0]):
            if self.T_monitor_names[i] in self.backward_T_monitor_names:
                mode_exp_data_set = self.fdtd_engine.fdtd.getresult(str(self.T_monitor_names[i]),
                                                                    'expansion for Output')

                transmission_coefficient_set.append(mode_exp_data_set['b'].flatten())
                N_set.append(mode_exp_data_set['N'].real.flatten())
                mode_coefficients.append((mode_exp_data_set['b'] * np.sqrt(mode_exp_data_set['N'].real)).flatten())
            else:
                mode_exp_data_set = self.fdtd_engine.fdtd.getresult(str(self.T_monitor_names[i]),
                                                                    'expansion for Output')
                transmission_coefficient_set.append(mode_exp_data_set['a'].flatten())
                N_set.append(mode_exp_data_set['N'].real.flatten())
                mode_coefficients.append((mode_exp_data_set['a'] * np.sqrt(mode_exp_data_set['N'].real)).flatten())

        mode_coefficients = np.array(mode_coefficients)
        N_set = np.array(N_set)

        forward_source_power = self.get_total_source_power(self.forward_source_names)
        transmission_coefficient_set = np.array(transmission_coefficient_set)
        transmission_coefficient_set_conj = np.conj(transmission_coefficient_set)
        real_part = np.real((transmission_coefficient_set + transmission_coefficient_set_conj) / 2)
        imag_part = np.real((transmission_coefficient_set - transmission_coefficient_set_conj) / 2j)
        power_value = np.power(real_part, 2) + np.power(imag_part, 2)
        self.radian_fwd_vs_wavelengths = np.arctan2(imag_part, real_part)
        self.T_fwd_vs_wavelengths = np.real(mode_coefficients * mode_coefficients.conj() / forward_source_power)
        self.phase_prefactors_radian = (- imag_part / power_value + real_part / power_value/ 1j) / np.sqrt(N_set)
        self.phase_prefactors_r = np.conj(mode_coefficients / forward_source_power)/2/np.sqrt(self.T_fwd_vs_wavelengths)
        self.eigen_mode_Ns = N_set


    def forward_sim(self, params):
        params = np.array(params, dtype=np.float64)
        self.forward_params = params
        self.fdtd_engine.switch_to_layout()
        self.fdtd_engine.set_enable_with_buffer(self.forward_source_names.tolist())
        self.fdtd_engine.set_disable_with_buffer(self.backward_source_names.tolist())
        self.design_region.update(params)
        self.fdtd_engine.eval_buffer()
        self.fdtd_engine.run(self.sim_name)
        self.forward_field = self.design_region.get_E_distribution()
        if self.eps_diff_func is None:
            self.original_epsilon_distribution = self.design_region.get_epsilon_distribution()
        self.get_forward_transmission_properties()
        self.fdtd_engine.switch_to_layout()
        self.fdtd_engine.set_disable_with_buffer(self.forward_source_names.tolist())
        self.fdtd_engine.eval_buffer()

        self.fom_T = self.T_fwd_vs_wavelengths
        self.fom_r = np.sqrt(self.fom_T)
        self.fom_radian = self.radian_fwd_vs_wavelengths

        return self.fom_r, self.fom_radian

    def backward_sim(self, bp_grad, params):
        self.fdtd_engine.switch_to_layout()

        omega = self.fdtd_engine.get_omega()
        r_grad = bp_grad[0]
        radian_grad = bp_grad[1]

        F_fwd_partial_derivs = []
        self.fdtd_engine.set_enable_with_buffer(self.backward_source_names)
        for i in range(0, np.shape(self.backward_source_names)[0]):
            prefix = (self.phase_prefactors_r[i] * r_grad[i] + self.phase_prefactors_radian[i] * radian_grad[i]
                      ) * 1j / np.sqrt(self.eigen_mode_Ns[i]) / 4.0
            prefix_amplitude = np.mean(np.abs(prefix))
            prefix_angle = np.mean(np.angle(prefix)) * 180 / np.pi
            self.fdtd_engine.reset_source_amplitude_with_buffer(self.backward_source_names[i], prefix_amplitude)
            self.fdtd_engine.reset_source_phase_with_buffer(self.backward_source_names[i], prefix_angle)

        self.fdtd_engine.eval_buffer()
        self.fdtd_engine.run()
        if (self.design_region_type == '3d'):
            cell = self.design_region.x_mesh * 1e-6 * self.design_region.y_mesh * 1e-6 * self.design_region.z_mesh * 1e-6
        else:
            cell = self.design_region.x_mesh * 1e-6 * self.design_region.y_mesh * 1e-6
        scaling_factor = omega

        self.adjoint_field = self.design_region.get_E_distribution()
        if self.eps_diff_func is None:
            self.eps_diff = self.cal_epsilon_diff_from_CAD(self.forward_params)
        else:
            self.eps_diff = self.eps_diff_func(self.reload_flag)
        gradient_field = np.mean(np.sum(2.0 * cell * scipy.constants.epsilon_0 * self.eps_diff * self.forward_field
                                        * self.adjoint_field, axis=4), axis=2)
        averaged_gradient_field = np.zeros((gradient_field.shape[2], len(self.blocks_x_min_index)),
                                           dtype='complex')

        for j in range(0, len(self.blocks_x_min_index)):
            x_averaged_gradient_field = np.mean(
                gradient_field[self.blocks_x_min_index[j]:self.blocks_x_max_index[j], :, :], axis=0)
            averaged_gradient_field[:, j] = np.mean(
                x_averaged_gradient_field[self.blocks_y_min_index[j]:self.blocks_y_max_index[j], :], axis=0)
        for wl in range(0, len(omega)):
            averaged_gradient_field[wl, :] = averaged_gradient_field[wl, :] * scaling_factor[wl]

        F_fwd_partial_derivs.append(np.real(averaged_gradient_field))

        F_fwd_partial_derivs = np.array(F_fwd_partial_derivs)

        F_fwd_partial_derivs = np.array(F_fwd_partial_derivs).transpose((2, 1, 0))

        return F_fwd_partial_derivs

    def cal_epsilon_diff_from_CAD(self, params):
        self.fdtd_engine.switch_to_layout()
        self.fdtd_engine.lumapi.putDouble(self.fdtd_engine.fdtd.handle, "dx", self.dx)
        # put original epsilon into CAD
        self.fdtd_engine.fdtd.putv('original_eps',
                                   np.expand_dims(self.original_epsilon_distribution, (-2)))
        # self.fdtd_engine.fdtd.redrawoff()

        perturbed_params = params + self.dx
        self.design_region.update(perturbed_params)
        self.fdtd_engine.eval("select(\"FDTD\");")
        self.fdtd_engine.eval("set(\"simulation time\", 1e-15);")
        self.fdtd_engine.run()
        self.fdtd_engine.get_epsilon_distribution_in_CAD(
            index_monitor_name=self.design_region.index_region_name,
            data_name="perturbed_epsilon")
        self.fdtd_engine.eval(
            "eps_diff_temp = (perturbed_epsilon - original_eps) / dx;")

        perturbed_epsilon = self.fdtd_engine.lumapi.getVar(self.fdtd_engine.fdtd.handle, "eps_diff_temp")
        self.fdtd_engine.eval("clear(perturbed_epsilon, dx, original_eps, eps_diff_temp);")
        # self.fdtd_engine.fdtd.redrawon()
        self.fdtd_engine.switch_to_layout()
        self.fdtd_engine.eval("select(\"FDTD\");")
        self.fdtd_engine.eval("set(\"simulation time\"," + str(self.simulation_time) + "e-15);")
        self.design_region.update(params)
        return perturbed_epsilon

    def reset_T_monitor_names(self, T_monitor_names):
        self.T_monitor_names = np.array([T_monitor_names]).flatten()

    def reset_forward_source_names(self, forward_source_names):
        self.forward_source_names = np.array([forward_source_names]).flatten()

    def reset_backward_source_names(self, backward_source_names):
        self.backward_source_names = np.array([backward_source_names]).flatten()
