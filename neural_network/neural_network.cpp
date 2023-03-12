//
//  neural_network.cpp
//  neural_network
//
//  Created by Sebastian on 09.03.23.
//
#include <iostream>
#include "neural_network.hpp"

void init_network(s_neural_network& neural_network, uint32_t input_size);
void print_activations(s_neural_network neural_network);
void print_weights(s_neural_network* neural_network);
void print_bias(s_neural_network neural_network);
float sigmoid(float x);
float random_float();
void print_outputs(s_neural_network neural_network);

s_neural_network start(uint32_t input_size)
{
    s_neural_network* neural_network = new s_neural_network;
    init_network(*neural_network, input_size);
    printf("Start Called \n");
    return *neural_network;
}


void init_network(s_neural_network& neural_network, uint32_t input_size)
{
    neural_network.layer_size = new uint32_t[4];
    neural_network.layer_size[0] = input_size;
    neural_network.layer_size[1] = 16;
    neural_network.layer_size[2] = 16;
    neural_network.layer_size[3] = 10;
    neural_network.activation = new float*[4];
    
    // Node initialization (activations) with 0.0f
    for(int a = 0; a < 4; a++)
    {
        neural_network.activation[a] = new float[neural_network.layer_size[a]];
        for(int b = 0; b < neural_network.layer_size[a]; b++)
        {
            neural_network.activation[a][b] = 0.1f;
        }
    }
    
    // Weights initializations 3D Array
    // Initializing float array. Weight has in summary 3 connections if there are 4 layers including input and output
    neural_network.weights = new float**[3];
    for(int a = 1; a <= 3; a++)
    {
        neural_network.weights[a-1] = new float*[neural_network.layer_size[a]]; // Init all avtivation connection
        for(int b = 0; b < neural_network.layer_size[a]; b++)
        {
            // Init connection to previous activation from current
            neural_network.weights[a-1][b] = new float[neural_network.layer_size[a-1]];
            for(int c = 0; c < neural_network.layer_size[a-1]; c++)
            {
                neural_network.weights[a-1][b][c] = random_float(); // Setting weight to 0.0f
            }
        }
    }
    
    neural_network.bias = new float*[3];
    for(int a = 1; a <= 3; a++)
    {
        neural_network.bias[a-1] = new float[neural_network.layer_size[a]];
        for(int b = 0; b < neural_network.layer_size[a]; b++)
        {
            neural_network.bias[a-1][b] = random_float();
        }
    }
    
    
    //print_activations(neural_network);
    //print_weights(&neural_network);
    //print_bias(neural_network);
    //print_outputs(neural_network);
}


float random_float() {
    return static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/1));
}

void feed_neural_network(s_neural_network& neural_network)
{
    
    for(int a = 1; a <= 3; a++)
    {
        for(int b = 0; b < neural_network.layer_size[a]; b++)
        {
            for(int c = 0; c < neural_network.layer_size[a-1]; c++)
            {
                neural_network.activation[a][b] = neural_network.activation[a-1][b] * neural_network.weights[a-1][b][c];
            }
            neural_network.activation[a-1][b] = sigmoid(neural_network.activation[a-1][b] + neural_network.bias[a-1][b]);
        }
    }
    //print_outputs(neural_network);
}


void delete_network(s_neural_network* neural_network)
{
    neural_network->layer_size = nullptr;
    delete [] neural_network->layer_size;
    neural_network->weights = nullptr;
    delete [] neural_network->weights;
    neural_network->activation = nullptr;
    delete [] neural_network->activation;
    neural_network->bias = nullptr;
    delete [] neural_network->bias;
    neural_network = nullptr;
}

float sigmoid(float x)
{
    return x / (1 + abs(x));
}


void print_bias(s_neural_network neural_network)
{
    for(int a = 1; a <= 3; a++)
    {
        for(int b = 0; b < neural_network.layer_size[a]; b++)
        {
            printf("%f ", neural_network.bias[a-1][b]);
        }
    }
    
}

void print_activations(s_neural_network neural_network)
{
    for(int a = 0; a <= 3; a++)
    {
        for(int b = 0; b < neural_network.layer_size[a]; b++)
        {
            printf("%f ", neural_network.activation[a][b]);
        }
        printf("\n");
    }
}



void print_weights(s_neural_network* neural_network)
{
    for(int a = 1; a <= 3; a++)
    {
        for(int b = 0; b < neural_network->layer_size[a]; b++)
        {
            for(int c = 0; c < neural_network->layer_size[a-1]; c++)
            {
                printf("%f ", neural_network->weights[a-1][b][c]);
            }
        }
    }
}

void print_outputs(s_neural_network neural_network)
{
    for(int a = 3; a <= 3; a++)
    {
        for(int b = 0; b < neural_network.layer_size[a]; b++)
        {
            printf("%f ", neural_network.activation[a][b]);
        }
    }
}



