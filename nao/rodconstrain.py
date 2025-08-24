# Licensed under the GPLv3 License. See LICENSE file for details.
# Copyright (c) 2025 ZhenyuZhao

import numpy as np


def min_feature_constrain(proportion_params , proportion_params_step_lengths, matrix_mask, radius_max ,block_length, min_feature_size_positive, min_feature_size_negative, min_feature_size_negative_corner = None, etch_type = "positive", edge_constant = 1):
    if type(min_feature_size_negative_corner) == type(None):
        min_feature_size_negative_corner = min_feature_size_negative
    if (type(matrix_mask) != type(None)):
        enable_positions = np.where(np.transpose(matrix_mask) == 1)
        if (len(np.transpose(enable_positions)) != len(proportion_params)):
            raise Exception("The proportion_params can not match the matrix_mask!")
        masked_matrix = matrix_mask.copy().astype(np.double)
        for i, position in enumerate(np.transpose(enable_positions)):
            masked_matrix[position[1], position[0]] = proportion_params[i]

        constrained_proportion_params = proportion_params.copy().astype(np.double)
        masked_matrix = np.pad(masked_matrix, (1, 1), 'constant', constant_values=(edge_constant,edge_constant))

        # adapt constrains
        if (etch_type == "positive"):
            for i, position in enumerate(np.transpose(enable_positions)):
                if (masked_matrix[position[1] + 1, position[0] + 1] <= 0):
                    masked_matrix[position[1] + 1, position[0] + 1] = 0
                if (masked_matrix[position[1] + 1, position[0] + 1]*radius_max*2 < min_feature_size_positive):
                    if (proportion_params_step_lengths[i] > 0):
                        masked_matrix[position[1] + 1, position[0] + 1] = min_feature_size_positive / radius_max / 2
                    elif (proportion_params_step_lengths[i] <= 0):
                        masked_matrix[position[1] + 1, position[0] + 1] = 0
                if (masked_matrix[position[1] + 1, position[0] + 1] > 1):
                    masked_matrix[position[1] + 1, position[0] + 1] = 1

            # derive new params
            for i, position in enumerate(np.transpose(enable_positions)):
                constrained_proportion_params[i] = masked_matrix[position[1] + 1, position[0] + 1]

        elif (etch_type == "negative"):
            for i, position in enumerate(np.transpose(enable_positions)):
                if (masked_matrix[position[1] + 1, position[0] + 1] <= 0):
                    masked_matrix[position[1] + 1, position[0] + 1] = 0
                if (masked_matrix[position[1] + 1, position[0] + 1] > 1):
                    masked_matrix[position[1] + 1, position[0] + 1] = 1
                # left constrain
                if ( block_length - masked_matrix[position[1], position[0] + 1] * radius_max -
                        masked_matrix[position[1] + 1, position[0] + 1] * radius_max < min_feature_size_negative):
                    masked_matrix[position[1] + 1, position[0] + 1] = (block_length - masked_matrix[position[1], position[0] + 1] * radius_max -
                                                                      min_feature_size_negative)/radius_max
                # up constrain
                if ( block_length - masked_matrix[position[1] + 1, position[0] ] * radius_max -
                        masked_matrix[position[1] + 1, position[0] + 1] * radius_max < min_feature_size_negative):
                    masked_matrix[position[1] + 1, position[0] + 1] = (block_length - masked_matrix[position[1] + 1, position[0]] * radius_max -
                                                                      min_feature_size_negative)/radius_max

                # down constrain
                if (block_length - masked_matrix[position[1] + 1, position[0] + 2] * radius_max -
                        masked_matrix[position[1] + 1, position[0] + 1] * radius_max < min_feature_size_negative):
                    masked_matrix[position[1] + 1, position[0] + 1] = (block_length - masked_matrix[
                        position[1] + 1, position[0] + 2] * radius_max -
                                                                       min_feature_size_negative) / radius_max
                # right constrain
                if (block_length - masked_matrix[position[1] + 2, position[0] + 1] * radius_max -
                        masked_matrix[position[1] + 1, position[0] + 1] * radius_max < min_feature_size_negative):
                    masked_matrix[position[1] + 1, position[0] + 1] = (block_length - masked_matrix[
                        position[1] + 2, position[0] + 1] * radius_max -
                                                                       min_feature_size_negative) / radius_max

                # left up constrain
                if ( block_length*np.sqrt(2) - masked_matrix[position[1], position[0] ] * radius_max -
                        masked_matrix[position[1] + 1, position[0] + 1] * radius_max < min_feature_size_negative):
                    masked_matrix[position[1] + 1, position[0] + 1] = (block_length*np.sqrt(2) - masked_matrix[position[1], position[0]] * radius_max -
                                                                      min_feature_size_negative)/radius_max

                # right up constrain
                if (block_length * np.sqrt(2) - masked_matrix[position[1] + 2, position[0]] * radius_max -
                        masked_matrix[position[1] + 1, position[0] + 1] * radius_max < min_feature_size_negative):
                    masked_matrix[position[1] + 1, position[0] + 1] = (block_length * np.sqrt(2) - masked_matrix[
                        position[1] + 2, position[0]] * radius_max -
                                                                   min_feature_size_negative) / radius_max

                # right down constrain
                if (block_length * np.sqrt(2) - masked_matrix[position[1] + 2, position[0] + 2] * radius_max -
                        masked_matrix[position[1] + 1, position[0] + 1] * radius_max < min_feature_size_negative):
                    masked_matrix[position[1] + 1, position[0] + 1] = (block_length * np.sqrt(2) - masked_matrix[
                        position[1] + 2, position[0] + 2] * radius_max -
                                                                       min_feature_size_negative) / radius_max
                # left down constrain
                if (block_length * np.sqrt(2) - masked_matrix[position[1], position[0] + 2] * radius_max -
                        masked_matrix[position[1] + 1, position[0] + 1] * radius_max < min_feature_size_negative):
                    masked_matrix[position[1] + 1, position[0] + 1] = (block_length * np.sqrt(2) - masked_matrix[
                        position[1], position[0] + 2] * radius_max -
                                                                       min_feature_size_negative) / radius_max

            # derive new params
            for i, position in enumerate(np.transpose(enable_positions)):
                constrained_proportion_params[i] = masked_matrix[position[1] + 1, position[0] + 1]
        elif (etch_type == "both"):
            for i, position in enumerate(np.transpose(enable_positions)):

                if (masked_matrix[position[1] + 1, position[0] + 1] <= 0):
                    masked_matrix[position[1] + 1, position[0] + 1] = 0
                if (masked_matrix[position[1] + 1, position[0] + 1]*radius_max*2 < min_feature_size_positive):
                    if (proportion_params_step_lengths[i] > 0):
                        masked_matrix[position[1] + 1, position[0] + 1] = min_feature_size_positive / radius_max / 2
                    elif (proportion_params_step_lengths[i] <= 0):
                        masked_matrix[position[1] + 1, position[0] + 1] = 0
                if (masked_matrix[position[1] + 1, position[0] + 1] > 1):
                    masked_matrix[position[1] + 1, position[0] + 1] = 1

                # left constrain
                if (block_length - masked_matrix[position[1], position[0] + 1] * radius_max -
                        masked_matrix[position[1] + 1, position[0] + 1] * radius_max < min_feature_size_negative):
                    masked_matrix[position[1] + 1, position[0] + 1] = (block_length - masked_matrix[
                        position[1], position[0] + 1] * radius_max -
                                                                       min_feature_size_negative) / radius_max
                # up constrain
                if (block_length - masked_matrix[position[1] + 1, position[0]] * radius_max -
                        masked_matrix[position[1] + 1, position[0] + 1] * radius_max < min_feature_size_negative):
                    masked_matrix[position[1] + 1, position[0] + 1] = (block_length - masked_matrix[
                        position[1] + 1, position[0]] * radius_max -
                                                                       min_feature_size_negative) / radius_max

                # down constrain
                if (block_length - masked_matrix[position[1] + 1, position[0] + 2] * radius_max -
                        masked_matrix[position[1] + 1, position[0] + 1] * radius_max < min_feature_size_negative):
                    masked_matrix[position[1] + 1, position[0] + 1] = (block_length - masked_matrix[
                        position[1] + 1, position[0] + 2] * radius_max -
                                                                       min_feature_size_negative) / radius_max
                # right constrain
                if (block_length - masked_matrix[position[1] + 2, position[0] + 1] * radius_max -
                        masked_matrix[position[1] + 1, position[0] + 1] * radius_max < min_feature_size_negative):
                    masked_matrix[position[1] + 1, position[0] + 1] = (block_length - masked_matrix[
                        position[1] + 2, position[0] + 1] * radius_max -
                                                                       min_feature_size_negative) / radius_max

                # left up constrain
                if (block_length * np.sqrt(2) - masked_matrix[position[1], position[0]] * radius_max -
                        masked_matrix[position[1] + 1, position[0] + 1] * radius_max < min_feature_size_negative_corner):
                    masked_matrix[position[1] + 1, position[0] + 1] = (block_length * np.sqrt(2) - masked_matrix[
                        position[1], position[0]] * radius_max -
                                                                       min_feature_size_negative_corner) / radius_max

                # right up constrain
                if (block_length * np.sqrt(2) - masked_matrix[position[1] + 2, position[0]] * radius_max -
                        masked_matrix[position[1] + 1, position[0] + 1] * radius_max < min_feature_size_negative_corner):
                    masked_matrix[position[1] + 1, position[0] + 1] = (block_length * np.sqrt(2) - masked_matrix[
                        position[1] + 2, position[0]] * radius_max -
                                                                       min_feature_size_negative_corner) / radius_max

                # right down constrain
                if (block_length * np.sqrt(2) - masked_matrix[position[1] + 2, position[0] + 2] * radius_max -
                        masked_matrix[position[1] + 1, position[0] + 1] * radius_max < min_feature_size_negative_corner):
                    masked_matrix[position[1] + 1, position[0] + 1] = (block_length * np.sqrt(2) - masked_matrix[
                        position[1] + 2, position[0] + 2] * radius_max -
                                                                       min_feature_size_negative_corner) / radius_max
                # left down constrain
                if (block_length * np.sqrt(2) - masked_matrix[position[1], position[0] + 2] * radius_max -
                        masked_matrix[position[1] + 1, position[0] + 1] * radius_max < min_feature_size_negative_corner):
                    masked_matrix[position[1] + 1, position[0] + 1] = (block_length * np.sqrt(2) - masked_matrix[
                        position[1], position[0] + 2] * radius_max -
                                                                       min_feature_size_negative_corner) / radius_max

                if (masked_matrix[position[1] + 1, position[0] + 1]*radius_max*2 < min_feature_size_positive):
                    masked_matrix[position[1] + 1, position[0] + 1] = 0

            # derive new params
            for i, position in enumerate(np.transpose(enable_positions)):
                constrained_proportion_params[i] = masked_matrix[position[1] + 1, position[0] + 1]

        else:
            raise Exception("Unknown etch_type is specified, it should be \'positive\' or \'negative\'!")

    elif (len(proportion_params.shape) != 2):
        raise Exception("The input matrix should be two-dimensional when matrix_mask not specified!")
    else:
        masked_matrix = proportion_params
        constrained_proportion_params = proportion_params.copy().astype(np.double)
        masked_matrix = np.pad(masked_matrix, (1, 1), 'constant', constant_values=(edge_constant,edge_constant))
        # adapt constrains
        if (etch_type == "positive"):
            for i in range(1, masked_matrix.shape[0] - 2):
                for j in range(1, masked_matrix.shape[1] - 2):
                    if (masked_matrix[i,j] <= 0):
                        masked_matrix[i,j] = 0
                    if (masked_matrix[i,j]* radius_max * 2 < min_feature_size_positive):
                        if (proportion_params_step_lengths[i-1,j-1] > 0):
                            masked_matrix[i,j] = min_feature_size_positive / radius_max / 2
                        elif (proportion_params_step_lengths[i-1,j-1] <= 0):
                            masked_matrix[i,j] = 0
                    if (masked_matrix[i,j] > 1):
                        masked_matrix[i,j] = 1

            # derive new params
            constrained_proportion_params = masked_matrix[1:-1,1:-1]

        elif (etch_type == "negative"):
            for i in range(1, masked_matrix.shape[0] - 2):
                for j in range(1, masked_matrix.shape[1] - 2):
                    if (masked_matrix[i,j] <= 0):
                        masked_matrix[i,j] = 0
                    if (masked_matrix[i,j] > 1):
                        masked_matrix[i,j] = 1
                    # left constrain
                    if (block_length - masked_matrix[i-1, j] * radius_max - masked_matrix[i, j] * radius_max < min_feature_size_negative):
                        masked_matrix[i,j] = (block_length - masked_matrix[i-1,j] * radius_max - min_feature_size_negative) / radius_max
                    # up constrain
                    if (block_length - masked_matrix[i, j-1] * radius_max - masked_matrix[i, j] * radius_max < min_feature_size_negative):
                        masked_matrix[i, j] = (block_length - masked_matrix[i, j-1] * radius_max - min_feature_size_negative) / radius_max
                    # down constrain
                    if (block_length - masked_matrix[i, j + 1] * radius_max - masked_matrix[i, j] * radius_max < min_feature_size_negative):
                        masked_matrix[i, j] = (block_length - masked_matrix[i, j + 1] * radius_max - min_feature_size_negative) / radius_max
                    # right constrain
                    if (block_length - masked_matrix[i + 1, j] * radius_max - masked_matrix[i, j] * radius_max < min_feature_size_negative):
                        masked_matrix[i, j] = (block_length - masked_matrix[i + 1, j] * radius_max - min_feature_size_negative) / radius_max
                    # left up constrain
                    if (block_length - masked_matrix[i - 1, j - 1] * radius_max - masked_matrix[
                        i, j] * radius_max < min_feature_size_negative_corner):
                        masked_matrix[i, j] = (block_length - masked_matrix[
                            i - 1, j - 1] * radius_max - min_feature_size_negative_corner) / radius_max
                    # right up constrain
                    if (block_length - masked_matrix[i + 1, j - 1] * radius_max - masked_matrix[
                        i, j] * radius_max < min_feature_size_negative_corner):
                        masked_matrix[i, j] = (block_length - masked_matrix[
                            i + 1, j - 1] * radius_max - min_feature_size_negative_corner) / radius_max
                    # right down constrain
                    if (block_length - masked_matrix[i + 1, j + 1] * radius_max - masked_matrix[
                        i, j] * radius_max < min_feature_size_negative_corner):
                        masked_matrix[i, j] = (block_length - masked_matrix[
                            i + 1, j + 1] * radius_max - min_feature_size_negative_corner) / radius_max
                    # left down constrain
                    if (block_length - masked_matrix[i - 1, j + 1] * radius_max - masked_matrix[
                        i, j] * radius_max < min_feature_size_negative_corner):
                        masked_matrix[i, j] = (block_length - masked_matrix[
                            i - 1, j + 1] * radius_max - min_feature_size_negative_corner) / radius_max

            # derive new params
            constrained_proportion_params = masked_matrix[1:-1, 1:-1]

        else:
            raise Exception("Unknown etch_type is specified, it should be \'positive' or \'negative'!")

    return constrained_proportion_params