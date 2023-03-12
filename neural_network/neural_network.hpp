//
//  neural_network.hpp
//  neural_network
//
//  Created by Sebastian on 09.03.23.
//

#pragma once

typedef struct neural_network_def {
    uint32_t* layer_size; // All sizes of each layer is stored here
    float** bias; 
    float** activation; // The Nodes which are activated at the end activation * weights
    float*** weights; // connection between the nodes
} s_neural_network;


s_neural_network start(uint32_t input_size);
void feed_neural_network(s_neural_network& neural_network);
void delete_network(s_neural_network* neural_network);




