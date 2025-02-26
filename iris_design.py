import numpy as np
from splayout import *
import sys
sys.path.append(".")
from adjoint.rodmetamaterialopt import RodMetaMaterialRegion2D, AdjointForRodMetaMaterialRegion
from adjoint.constraints import min_feature_constrain
import os
import time

def make_physical_model():
    fdtd = FDTDSimulation()
    effSi = 2.85189
    rect_region = Waveguide(Point(-4, 0), Point(4, 0), width=8, material=effSi, z_start=-0.11, z_end=0.11,
                            rename="rect_region")
    rect_region.draw_on_lumerical_CAD(fdtd)

    # input waveguides
    for i in range(0, 4):
        input_waveguide = Waveguide(Point(-4, -3 + 2 * i) - (3, 0), Point(-4, -3 + 2 * i) - (1, 0), width=0.5,
                                    material=effSi, z_start=-0.11, z_end=0.11)
        input_waveguide.draw_on_lumerical_CAD(fdtd)

    # input tapers
    for i in range(0, 4):
        input_taper = Taper(Point(-4, -3 + 2 * i) - (1, 0), Point(-4, -3 + 2 * i), start_width=0.5, end_width=2,
                            material=effSi, z_start=-0.11, z_end=0.11)
        input_taper.draw_on_lumerical_CAD(fdtd)

    # output waveguides
    for i in range(0, 3):
        output_waveguide = Waveguide(Point(4, -2 + 2 * i) + (3, 0), Point(4, -2 + 2 * i) + (1, 0), width=0.5,
                                     material=effSi, z_start=-0.11, z_end=0.11)
        output_waveguide.draw_on_lumerical_CAD(fdtd)

    # output tapers
    for i in range(0, 3):
        output_taper = Taper(Point(4, -2 + 2 * i), Point(4, -2 + 2 * i) + (1, 0), start_width=2, end_width=0.5,
                             material=effSi, z_start=-0.11, z_end=0.11)
        output_taper.draw_on_lumerical_CAD(fdtd)

    fdtd.add_fdtd_region(bottom_left_corner_point=Point(-4 - 1.8, -5.5), top_right_corner_point=Point(4 + 1.8, 5.5),
                         dimension=2, simulation_time=5000, height=0.8, z_symmetric=1)

    fdtd.eval("set(\"mesh refinement\", \"volume average\");")

    # input sources
    for i in range(0, 4):
        fdtd.add_mode_source(Point(-4 - 1.5, 3 - 2 * i), width=1, source_name="Forward_" + str(i),
                             direction=FORWARD, mode_number=2,
                             wavelength_start=1.55, wavelength_end=1.55)

    # output monitors
    for i in range(0, 3):
        fdtd.add_mode_expansion(Point(4 + 1.5, 2 - 2 * i), mode_list=[2], width=1, expansion_name="Monitor_" + str(i), points=1)

    # backward sources
    for i in range(0, 3):
        fdtd.add_mode_source(Point(4 + 1.5, 2 - 2 * i), width=1, source_name="Backward_" + str(i),
                             direction=BACKWARD, mode_number=2,
                             wavelength_start=1.55, wavelength_end=1.55)

    # add TO region
    pixels_region_mask_matrix = np.ones((40, 40))
    Pixel_Region = CirclePixelsRegion(bottom_left_corner_point=Point(-4, -4), material=SiO2,
                                      top_right_corner_point=Point(4, 4), pixel_radius=0.1,
                                      z_start=-0.11, z_end=0.11, relaxing_time=0.5,
                                      fdtd_engine=fdtd, matrix_mask=pixels_region_mask_matrix)
    Design_Region = RodMetaMaterialRegion2D(fdtd, Pixel_Region, x_mesh=0.02, y_mesh=0.02, z_mesh=0.02, z_start=-0.11, z_end=0.11,
                                   rename="RodMetaMaterialRegion")

    monitor_list = ["Monitor_0", "Monitor_1", "Monitor_2" ]
    forward_source_list = ["Forward_0", "Forward_1", "Forward_2", "Forward_3"]
    backward_source_list = ["Backward_0", "Backward_1", "Backward_2"]
    targets_list = [np.ones(1) * 0.5, np.ones(1) * 0.5, np.ones(1) * 0.5]

    Opt = AdjointForRodMetaMaterialRegion(fdtd, monitor_list, targets_list, Design_Region, forward_source_list,
                                             backward_source_list, index_origin=effSi, index_pert=1.444, if_default_fom=0)

    return fdtd, Design_Region, Opt

def set_inputs(data_input):
    phases = -data_input*180
    command = "select(\"Forward_0\");" + "set(\"phase\"," + str(phases[0]) + ");" + \
              "select(\"Forward_1\");" + "set(\"phase\"," + str(phases[1]) + ");" + \
              "select(\"Forward_2\");" + "set(\"phase\"," + str(phases[2]) + ");" + \
              "select(\"Forward_3\");" + "set(\"phase\"," + str(phases[3]) + ");"
    fdtd.switch_to_layout()
    fdtd.eval(command)

def generate_index_pool(data):
    index_pool = np.array(range(0, np.shape(data)[0]))
    return index_pool

def random_select_from_index_pool(index_pool, batch_size):
    selected_indexes = []
    selecting_size = batch_size if batch_size < index_pool.size else index_pool.size
    for i in range(0, selecting_size):
        selected_index = np.random.randint(0, index_pool.size)
        selected_indexes.append(index_pool[selected_index])
        index_pool = np.delete(index_pool, selected_index)
    return index_pool, selected_indexes

def cal_accuracy(foms, targets):
    if (np.argmax(np.abs(foms)) == np.argmax(targets)):
        return 1
    else:
        return 0

if __name__ == '__main__':
    fdtd, opt_region, Opt = make_physical_model()
    fdtd.save("TEMP")
    train_data_features = np.load("./dataset/train_data_features.npy")
    train_data_targets = np.load("./dataset/train_data_targets.npy")
    test_data_features = np.load("./dataset/test_data_features.npy")
    test_data_targets = np.load("./dataset/test_data_targets.npy")

    MAX_ITERATION = 40
    BATCH_SIZE = 5
    BETA1 = 0.9
    BETA2 = 0.999
    EPSILON = 1e-8
    STEP_SIZE = 1e-2

    RELOAD_FLAG = 0
    RELOAD_INDEX_POOL_PATH = "./current_index_pool.npy"
    RELOAD_ITER_PATH = "./current_iter.npy"
    RELOAD_PARAMS_PATH = "./current_params.npy"
    RELOAD_MOPT_PATH = "./current_mopt.npy"
    RELOAD_VOPT_PATH = "./current_vopt.npy"
    if (RELOAD_FLAG):
        ITERATION_BEGIN = np.load(RELOAD_ITER_PATH)
        params = np.load(RELOAD_PARAMS_PATH)
        MOPT = np.load(RELOAD_MOPT_PATH)
        VOPT = np.load(RELOAD_VOPT_PATH)
    else:
        ITERATION_BEGIN = 0
        np.random.seed(11328)
        params = np.random.rand(1600) / 1.25 + 0.1
        if not os.path.isdir("./Models"):
            os.makedirs("./Models")
        np.save("./Models/" + "initial", params)
        MOPT = np.zeros(params.shape)
        VOPT = np.zeros(params.shape)

    for iteration in range(ITERATION_BEGIN, MAX_ITERATION):
        # reload from the stop point
        if (RELOAD_FLAG):
            index_pool = np.load(RELOAD_INDEX_POOL_PATH)
            RELOAD_FLAG = 0
        else:
            index_pool = generate_index_pool(train_data_features)

        # train process
        print("Iteration:", iteration)
        train_accuracy_list = []
        train_loss_list = []
        while (index_pool.size != 0):
            print("Remained Pool Size:", index_pool.size)
            index_pool, selected_indexes = random_select_from_index_pool(index_pool, batch_size=BATCH_SIZE)
            gradient = np.zeros(params.size)
            for index in selected_indexes:
                time.sleep(0.1)
                set_inputs(train_data_features[index])
                Opt.reset_target_T(train_data_targets[index])
                time.sleep(0.1)
                foms = Opt.call_fom(params)
                # nonlinear function add
                nonlinear_output = 1/(1 + np.exp(-10*(foms - 0.5)))
                # calculate loss CEE
                total_output = nonlinear_output[0] + nonlinear_output[1] + nonlinear_output[2]
                cee_loss = - train_data_targets[index, 0] * np.log(nonlinear_output[0] / total_output) \
                           - train_data_targets[index, 1] * np.log(nonlinear_output[1] / total_output) \
                           - train_data_targets[index, 2] * np.log(nonlinear_output[2] / total_output)


                time.sleep(0.5)
                initial_grad =  np.squeeze(Opt.call_grad(params))
                # calculate true gradient with initial gradient
                nonlinear_grad = np.reshape(10*1/(1 + np.exp(-10*(foms - 0.5)))*(1 - 1/(1 + np.exp(-10*(foms - 0.5)))),(3,1))*initial_grad
                # CEE back
                total_grad = nonlinear_grad[0] + nonlinear_grad[1] + nonlinear_grad[2]
                cee_grad = - train_data_targets[index, 0] * (nonlinear_grad[0] / nonlinear_output[0] - total_grad / total_output) \
                           - train_data_targets[index, 1] * (nonlinear_grad[1] / nonlinear_output[1] - total_grad / total_output) \
                           - train_data_targets[index, 2] * (nonlinear_grad[2] / nonlinear_output[2] - total_grad / total_output)

                gradient += -cee_grad
                train_accuracy_list.append(cal_accuracy(nonlinear_output, train_data_targets[index]))
                train_loss_list.append(cee_loss)

            # adam optimizer
            MOPT = BETA1 * MOPT + (1 - BETA1) * gradient
            mopt_t = MOPT / (1 - BETA1 ** (iteration + 1))
            VOPT = BETA2 * VOPT + (1 - BETA2) * (np.square(gradient))
            vopt_t = VOPT / (1 - BETA2 ** (iteration + 1))
            grad_adam = mopt_t / (np.sqrt(vopt_t) + EPSILON)

            # params update
            params = params + grad_adam * STEP_SIZE
            # constraints
            params = min_feature_constrain(params, grad_adam * STEP_SIZE, np.ones((40, 40)), 0.1, 0.2,
                                           min_feature_size_positive=0.08, min_feature_size_negative=0.05,
                                           min_feature_size_negative_corner=0.05, etch_type="both")
            params = np.clip(params, 0, 1)


            np.save("current_params", params)
            np.save("current_iter", iteration)
            np.save("current_index_pool", index_pool)
            np.save("current_mopt", MOPT)
            np.save("current_vopt", VOPT)

        # logging the training process
        train_accuracy = np.mean(train_accuracy_list)
        train_loss = np.mean(train_loss_list)
        print("Iter: ", iteration, "Train Acc: ", train_accuracy, "Train Loss: ", train_loss)

        # test process
        test_accuracy_list = []
        test_loss_list = []
        for data_index in range(0, np.shape(test_data_features)[0]):
            print("Test Index:", data_index)
            set_inputs(test_data_features[data_index])
            Opt.reset_target_T(test_data_targets[data_index])
            foms = Opt.call_fom(params)
            # if data_index % 10 == 0:
            #     print(test_data_targets[data_index, :], foms)
            # calculate loss CEE
            nonlinear_output = 1 / (1 + np.exp(-10 * (foms - 0.5)))
            total_output = nonlinear_output[0] + nonlinear_output[1] + nonlinear_output[2]
            cee_loss = - test_data_targets[data_index, 0] * np.log(nonlinear_output[0] / total_output) \
                       - test_data_targets[data_index, 1] * np.log(nonlinear_output[1] / total_output) \
                       - test_data_targets[data_index, 2] * np.log(nonlinear_output[2] / total_output)

            test_accuracy_list.append(cal_accuracy(nonlinear_output, test_data_targets[data_index]))
            test_loss_list.append(cee_loss)

        # logging the test process
        test_accuracy = np.mean(test_accuracy_list)
        test_loss = np.mean(test_loss_list)
        print("Iteration: ", iteration, "Test Acc: ", test_accuracy, "Test Loss: ", test_loss)

        # save key information in this epoch
        if not os.path.isdir("./Models"):
            os.makedirs("./Models")
        np.save("./Models/" + str(iteration), params)