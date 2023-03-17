//
//  neural_network.hpp
//  neural_network
//
//  Created by Sebastian on 09.03.23.
//

#include "training_data.hpp"

#pragma once

typedef struct neural_network_def {
    uint32_t* layer_size; // All sizes of each layer is stored here
    float** bias;
    float avg_cost; // Cost calculation of all outputs
    float** activation; // The Nodes which are activated at the end activation * weights
    float*** weights; // connection between the nodes
    int*** changed_weights;
} s_neural_network;


s_neural_network start(uint32_t input_size);
void feed_neural_network(s_neural_network& neural_network);
void delete_network(s_neural_network* neural_network);
float calculate_cost(s_neural_network* neural_network);
void train_network(s_neural_network& neural_network, digit_image_collection_t& training_data);
void print_activations(s_neural_network neural_network);
void print_weights(s_neural_network* neural_network);
void print_bias(s_neural_network neural_network);
void print_outputs(s_neural_network neural_network);
void adjust_weight(bool change_again, int x, int y, int z, s_neural_network& neural_network);


