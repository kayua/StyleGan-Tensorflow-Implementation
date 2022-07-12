# !/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'All'
__email__ = ' @gmail.com, @unipampa.edu.br '
__version__ = '{2}.{0}.{1}'
__data__ = '2021/11/21'
__credits__ = ['All']

import logging
import operator
import numpy


class Graph:

    def __init__(self, window_width, window_length, window_blocks, window_parameters, first_column_sort,
                 second_column_sort, threshold_prediction, temporary_path, recurrence_feature, feature_overlap):

        self.features_window_width = window_width
        self.features_window_length = window_length
        self.features_window_blocks = window_blocks
        self.features_window_parameters = window_parameters
        self.first_column_sort = first_column_sort
        self.tensor_features = None
        self.second_column_sort = second_column_sort
        self.threshold_prediction = threshold_prediction
        self.path_output_file = temporary_path
        self.recurrence_feature = recurrence_feature
        self.feature_overlap = feature_overlap
        self.snapshot_time_temp = 0

    def tensor_allocation_buffer(self):

        logging.debug('Starting allocation memory tensor')
        tensor_allocation_size_width = self.features_window_width
        tensor_allocation_size_width *= self.features_window_blocks
        tensor_allocation_size_length = self.features_window_length

        size_vector_zeros_generate = tensor_allocation_size_width
        size_vector_zeros_generate *= tensor_allocation_size_length
        size_vector_zeros_generate *= self.features_window_parameters

        logging.debug('Allocation tensor size {} bytes'.format(size_vector_zeros_generate * 4))
        matrix_feature = numpy.zeros(size_vector_zeros_generate, float)

        tensor_dimension_shape = '{}:{}'.format(tensor_allocation_size_width, tensor_allocation_size_length)
        tensor_dimension_shape = '{}:{}'.format(tensor_dimension_shape, tensor_allocation_size_length)

        logging.debug('Shape tensor {}'.format(tensor_dimension_shape))
        logging.debug('End tensor allocation process')

        return matrix_feature.reshape((tensor_allocation_size_width, tensor_allocation_size_length,
                                       self.features_window_parameters))

    @staticmethod
    def clean_tensor_buffer(tensor_buffer):

        logging.debug('Starting clean tensor buffer of extract tensor_feature')

        for axis_x, tensor_plane in enumerate(tensor_buffer):

            for axis_y, tensor_line in enumerate(tensor_plane):

                for axis_z, tensor_parameters in enumerate(tensor_line):
                    tensor_buffer[axis_x][axis_y][axis_z] = 0.

        logging.debug('End clean tensor buffer of extract tensor_feature')
        return tensor_buffer

    def add_in_tensor_feature(self, tensor_feature, memory_allocation, identifier_snapshot_time,
                              node_identifier_value, parameter_value):

        if identifier_snapshot_time > self.snapshot_time_temp:
            logging.debug('Window offset {}'.format(self.snapshot_time_temp))
            self.snapshot_time_temp = self.snapshot_time_temp
            self.snapshot_time_temp += self.features_window_length
            matrix_numpy_format = numpy.array(memory_allocation)

            logging.debug('Adding new tensor_feature in tensor block')
            tensor_feature.append(matrix_numpy_format)

            shape_matrix = numpy.array(self.tensor_features).shape
            logging.debug('New tensor_feature added to tensor Shape {}'.format(shape_matrix))

            memory_allocation = self.clean_tensor_buffer(memory_allocation)

        if (identifier_snapshot_time % self.features_window_length) != 0:

            new_column_identifier = (identifier_snapshot_time % self.features_window_length) - 1

            for position, value in enumerate(parameter_value):
                memory_allocation[node_identifier_value][new_column_identifier][position] = round(value, 3)

        else:

            for position, value in enumerate(parameter_value):
                memory_allocation[node_identifier_value][self.features_window_length - 1] = round(value, 3)

        return tensor_feature, memory_allocation

    def load_dataset_graph_file(self, dataset_filename):

        logging.info('Starting load dataset file {}'.format(dataset_filename))
        memory_allocated = self.load_graph_to_memory(dataset_filename)
        memory_allocation = self.tensor_allocation_buffer()
        tensor_feature = []

        for list_node in memory_allocated:
            tensor_feature, memory_allocation = self.add_in_tensor_feature(tensor_feature, memory_allocation,
                                                                           int(list_node[0]), int(list_node[1] % 32),
                                                                           list_node[2:])

        return numpy.array(tensor_feature)

    def write_new_dataset_features(self, tensor_feature, initial_position, file_pointer):

        original_tensor_shape = self.restore_original_tensor_dimension(tensor_feature)

        for axis_x, feature_frame in enumerate(original_tensor_shape):

            for axis_y, feature_line in enumerate(feature_frame):

                if float(feature_line[0].numpy()) > 0.001:

                    dataset_format_restore = ('{} {}'.format(axis_y + initial_position + 1, axis_x))
                    for axis_z, feature_dot in enumerate(feature_line):
                        dataset_format_restore += ' {} '.format(round(feature_dot.numpy(), 3))

                    file_pointer.write('{}\n'.format(dataset_format_restore))

    @staticmethod
    def create_file_dataset_output(filename_dataset_output):

        try:

            logging.debug('Creating file dataset output {}'.format(filename_dataset_output))
            file_dataset_output = open(filename_dataset_output, 'w')
            return file_dataset_output

        except FileExistsError:

            logging.error('File Exiting')
            exit(-1)

    @staticmethod
    def restore_original_tensor_dimension(multi_dimensional_tensor):

        original_tensor_features = []

        for feature_frame in multi_dimensional_tensor:
            original_tensor_features.extend(feature_frame)

        return original_tensor_features

    def get_features(self):

        return numpy.array(self.tensor_features)

    def reduction_feature_dimension(self):

        logging.info('Starting process tensor_feature reduction tensor_feature dimension')
        tensor_dimension_reduced = [[] for _ in range(self.features_window_blocks)]
        tensor_unique_channel = []
        format_width_feature = self.features_window_width
        format_width_feature *= self.features_window_blocks

        for feature_frame in self.tensor_features:

            feature_cut_points = range(0, format_width_feature, self.features_window_width)

            for feature_identifier, first_cut_point in enumerate(feature_cut_points):
                second_cut_point = first_cut_point
                second_cut_point += self.features_window_width
                feature_frame_reduced = feature_frame[first_cut_point:second_cut_point]
                tensor_dimension_reduced[feature_identifier].append(feature_frame_reduced)

        logging.debug('Concatenate multi channel tensor')

        for channel_feature in tensor_dimension_reduced:
            tensor_unique_channel.extend(channel_feature)

        self.tensor_features = numpy.array(tensor_unique_channel)
        logging.info('End process tensor_feature reduction tensor_feature dimension')
        logging.debug('New dimension features {}'.format(self.tensor_features.shape))

    @staticmethod
    def load_graph_to_memory(dataset_filename):

        temporary_tensor_graph = []

        try:

            dataset_pointer_file = open(dataset_filename, 'r')

            for identifier_line_data, dataset_read_line in enumerate(dataset_pointer_file.readlines()):
                dataset_line_list = dataset_read_line.split('\n')[0]
                dataset_line_list = dataset_line_list.split(' ')
                dataset_line_list = [float(dataset) for dataset in dataset_line_list if dataset != '']
                temporary_tensor_graph.append(dataset_line_list)

            return sorted(temporary_tensor_graph, key=operator.itemgetter(1))

        except FileNotFoundError:

            logging.error('File dataset not found {}'.format(dataset_filename))
            exit(-1)

    def get_window_width(self):
        return self.features_window_width

    def get_window_length(self):
        return self.features_window_length

    def get_window_blocks(self):
        return self.features_window_blocks

    def get_window_parameters(self):
        return self.features_window_parameters

    def get_first_column_sort(self):
        return self.first_column_sort

    def get_second_column_sort(self):
        return self.second_column_sort

    def get_threshold(self):
        return self.threshold_prediction

    def set_window_width(self, window_width):
        self.features_window_width = window_width

    def set_window_length(self, window_length):
        self.features_window_length = window_length

    def set_window_blocks(self, window_blocks):
        self.features_window_blocks = window_blocks

    def set_window_parameters(self, window_parameters):
        self.features_window_parameters = window_parameters

    def set_first_column_sort(self, first_column_sort):
        self.first_column_sort = first_column_sort

    def set_second_column_sort(self, second_column_sort):
        self.second_column_sort = second_column_sort

    def set_threshold(self, threshold):
        self.threshold_prediction = threshold
