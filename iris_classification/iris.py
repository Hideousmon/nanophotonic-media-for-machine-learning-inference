# Licensed under the GPLv3 License. See LICENSE file for details.
# Copyright (c) 2025 ZhenyuZhao

import numpy as np
from splayout import *
from nao import RodMetaMaterialRegion3D, AdjointForRodComplexdirect, min_feature_constrain
from batch_preprocess import get_batch_data
import os
import jax.numpy as jnp
import jax

def make_physical_model():
    fdtd = FDTDSimulation()
    rect_region = Waveguide(Point(-4, 0), Point(4, 0), width=8, material=Si, z_start=-0.11, z_end=0.11,
                            rename="rect_region")
    rect_region.draw_on_lumerical_CAD(fdtd)

    # input waveguides
    for i in range(0, 4):
        input_waveguide = Waveguide(Point(-4, -3 + 2 * i) - (3, 0), Point(-4, -3 + 2 * i), width=0.5,
                                    material=Si, z_start=-0.11, z_end=0.11)
        input_waveguide.draw_on_lumerical_CAD(fdtd)

    # output waveguides
    for i in range(0, 3):
        output_waveguide = Waveguide(Point(4, -2 + 2 * i) + (3, 0), Point(4, -2 + 2 * i), width=0.5,
                                     material=Si, z_start=-0.11, z_end=0.11)
        output_waveguide.draw_on_lumerical_CAD(fdtd)

    fdtd.add_fdtd_region(bottom_left_corner_point=Point(-4 - 1, -5), top_right_corner_point=Point(4 + 1, 5),
                         dimension=3, simulation_time=5000, height=0.8, use_gpu=1)

    # input sources
    for i in range(0, 4):
        fdtd.add_mode_source(Point(-4 - 0.5, 3 - 2 * i), width=1, source_name="Forward_" + str(i),
                             direction=FORWARD, mode_number=1,
                             wavelength_start=1.55, wavelength_end=1.55)

    # output monitors
    for i in range(0, 3):
        fdtd.add_mode_expansion(Point(4 + 0.5, 2 - 2 * i), mode_list=[1], width=1, expansion_name="Monitor_" + str(i),
                                points=1)

    # backward sources
    for i in range(0, 3):
        fdtd.add_mode_source(Point(4 + 0.5, 2 - 2 * i), width=1, source_name="Backward_" + str(i),
                             direction=BACKWARD, mode_number=1,
                             wavelength_start=1.55, wavelength_end=1.55)

    # add TO region
    pixels_region_mask_matrix = np.ones((20, 20))
    Pixel_Region = CirclePixelsRegionwithGroup(bottom_left_corner_point=Point(-4, -4), material=1.444,
                                      top_right_corner_point=Point(4, 4), pixel_radius=0.2,
                                      z_start=0.04, z_end=0.11,
                                      fdtd_engine=fdtd, matrix_mask=pixels_region_mask_matrix)
    Design_Region = RodMetaMaterialRegion3D(fdtd, Pixel_Region, x_mesh=0.025, y_mesh=0.025, z_mesh=0.01, z_start=0.04, z_end=0.11,
                                   rename="RodMetaMaterialRegion")

    monitor_list = ["Monitor_" + str(i) for i in range(0, 3)]
    forward_source_list = ["Forward_" + str(i) for i in range(0, 4)]
    backward_source_list = ["Backward_" + str(i) for i in range(0, 3)]

    fdtd.set_disable(forward_source_list)

    adjoint_list = []
    for i in range(0, 4):
        adjoint = AdjointForRodComplexdirect(fdtd, monitor_list, Design_Region, forward_source_list[i],
                                             backward_source_list, index_origin=3.478, index_pert=1.444,
                                             em_grad_normalization=0, simulation_time=5000)
        adjoint_list.append(adjoint)

    return fdtd, Pixel_Region, Design_Region, adjoint_list

if __name__ == '__main__':
    fdtd, pixel_region, opt_region, adjoint_list = make_physical_model()

    init_params = np.ones(20*20)*0.5
    if not os.path.isdir("./Models"):
        os.makedirs("./Models")
    np.save("./Models/" + "initial", init_params)

    def test_and_batch_train(params, test_features, test_targets, batch_features, batch_targets):
        trans1 = adjoint_list[0].forward(params)
        trans2 = adjoint_list[1].forward(params)
        trans3 = adjoint_list[2].forward(params)
        trans4 = adjoint_list[3].forward(params)

        transfer_m = jnp.concatenate([trans1, trans2, trans3, trans4], axis=1)


        # calculate transmission for test
        transfer_m_test = np.expand_dims(jax.lax.stop_gradient(transfer_m), axis=0)
        transfer_m_test = np.repeat(transfer_m_test, test_features.shape[0], axis=0)
        feature_data_test = np.exp(-1j * np.pi * np.expand_dims(jax.lax.stop_gradient(test_features), axis=(2)))
        transmission_test = np.power(np.abs(np.matmul(transfer_m_test, feature_data_test)), 2)[:, :,
                            0] / 4  # normalization for input


        # calculate transmission
        transfer_m = jnp.expand_dims(transfer_m, axis=0)
        transfer_m = jnp.repeat(transfer_m, batch_features.shape[0], axis=0)
        feature_data = jnp.exp(-1j * jnp.pi * jnp.expand_dims(batch_features, axis=(2)))
        transmission = jnp.power(jnp.abs(jnp.matmul(transfer_m, feature_data)), 2)[:, :,
                       0] / 4  # normalization for input

        # calculate test loss
        nmse_loss_test = np.mean(
            np.sum(np.power(transmission_test / np.expand_dims(np.sum(transmission_test, axis=1), 1) - test_targets, 2),
                    axis=1))
        global test_loss_list
        test_loss_list.append(nmse_loss_test)

        # calculate test acc
        argmax_fom_test = np.argmax(transmission_test, axis=1)
        argmax_target_test = np.argmax(jax.lax.stop_gradient(test_targets), axis=1)
        match_points_test = argmax_fom_test == argmax_target_test
        acc_test = np.sum(match_points_test) / transmission_test.shape[0]
        global test_acc_list
        test_acc_list.append(acc_test)

        # calculate loss
        nmse_loss = jnp.mean(
            jnp.sum(jnp.power(transmission / jnp.expand_dims(jnp.sum(transmission, axis=1), 1) - batch_targets, 2),
                    axis=1))

        # calculate acc
        argmax_fom = np.argmax(jax.lax.stop_gradient(transmission), axis=1)
        argmax_target = np.argmax(jax.lax.stop_gradient(batch_targets), axis=1)
        match_points = argmax_fom == argmax_target
        acc = np.sum(match_points) / jax.lax.stop_gradient(transmission).shape[0]
        global batch_acc_list
        batch_acc_list.append(acc)

        return nmse_loss


    val_and_grad = jax.value_and_grad(test_and_batch_train, argnums=0)

    train_data_features = np.load("../datasets/iris/train_data_features.npy")
    train_data_targets = np.load("../datasets/iris/train_data_targets.npy")
    test_data_features = np.load("../datasets/iris/test_data_features.npy")
    test_data_targets = np.load("../datasets/iris/test_data_targets.npy")

    MAX_ITERATION = 100
    BATCH_SIZE = 120
    BETA1 = 0.99
    BETA2 = 0.999
    EPSILON = 1e-8
    STEP_SIZE = 1e-2

    params = init_params.copy().flatten()
    MOPT = np.zeros(params.shape)
    VOPT = np.zeros(params.shape)

    for iteration in range(0, MAX_ITERATION):
        batch_features_list, batch_targets_list = get_batch_data(train_data_features, train_data_targets, BATCH_SIZE)
        iteration_acc_list = []
        iteration_loss_list = []
        iteration_test_acc_list = []
        iteration_test_loss_list = []

        # save key information in this epoch
        if not os.path.isdir("./Models"):
            os.makedirs("./Models")
        np.save("./Models/" + str(iteration), params)

        for i in range(0, len(batch_features_list)):
            batch_acc_list = []
            test_acc_list = []
            test_loss_list = []
            batch_loss, grad = val_and_grad(params, test_data_features, test_data_targets ,batch_features_list[i], batch_targets_list[i])
            batch_acc = np.mean(np.array(batch_acc_list))
            test_acc = np.mean(np.array(test_acc_list))
            test_loss = np.mean(np.array(test_loss_list))
            iteration_loss_list.append(batch_loss)
            iteration_acc_list.append(batch_acc)
            iteration_test_acc_list.append(test_acc)
            iteration_test_loss_list.append(test_loss)

            # params update
            grad = - np.array(grad)
            MOPT = BETA1 * MOPT + (1 - BETA1) * grad
            mopt_t = MOPT / (1 - BETA1 ** (iteration + 1))
            VOPT = BETA2 * VOPT + (1 - BETA2) * (np.square(grad))
            vopt_t = VOPT / (1 - BETA2 ** (iteration + 1))
            grad_adam = mopt_t / (np.sqrt(vopt_t) + EPSILON)

            params = params + grad_adam * STEP_SIZE
            params = min_feature_constrain(params, grad_adam * STEP_SIZE, np.ones((20, 20)), 0.2, 0.4,
                                           min_feature_size_positive=0.13, min_feature_size_negative=0.13,
                                           min_feature_size_negative_corner=0.13, etch_type="both")
            params = np.clip(params, 0, 1)

        train_accuracy = np.mean(np.array(iteration_acc_list))
        train_loss = np.mean(np.array(iteration_loss_list))
        test_accuracy = np.mean(np.array(iteration_test_acc_list))
        test_loss = np.mean(np.array(iteration_test_loss_list))
        print("Iteration: ", iteration, "Test Acc: ", test_accuracy, "Test Loss: ", test_loss)
        print("Iteration: ", iteration, "Train Acc: ", train_accuracy, "Train Loss: ", train_loss)

        # reload fdtd
        fdtd.eval("clear;")
        fdtd.save("./reload.fsp")
        fdtd = FDTDSimulation(load_file="./reload.fsp")
        fdtd.global_monitor_set_flag = 1
        fdtd.global_source_set_flag = 1
        fdtd.wavelength_start = 1.55e-6
        fdtd.wavelength_end = 1.55e-6
        fdtd.frequency_points = 1
        pixel_region.fdtd_engine = fdtd
        opt_region.fdtd_engine = fdtd
        for adj in adjoint_list:
            adj.fdtd_engine = fdtd

