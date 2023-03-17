//
//  neural_network.cpp
//  neural_network
//
//  Created by Sebastian on 09.03.23.
//
#include <iostream>
#include <math.h>
#include "neural_network.hpp"
#include "training_data.hpp"
#include <random>

void init_network(s_neural_network& neural_network, uint32_t input_size);
float sigmoid(float x);
float random_float();
float calculate_cost(s_neural_network& neural_network, digit_image_t training_data);



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
            neural_network.activation[a][b] = 0.0;
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
                neural_network.weights[a-1][b][c] = random_float(); // Setting random weight
            }
        }
    }
    
    neural_network.changed_weights = new int**[3];
    for(int a = 1; a <= 3; a++)
    {
        neural_network.changed_weights[a-1] = new int*[neural_network.layer_size[a]]; 
        for(int b = 0; b < neural_network.layer_size[a]; b++)
        {
            neural_network.changed_weights[a-1][b] = new int[neural_network.layer_size[a-1]];
            for(int c = 0; c < neural_network.layer_size[a-1]; c++)
            {
                neural_network.changed_weights[a-1][b][c] = 0;
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
}


float random_float() {
    // Seed the random engine with a random device
    std::random_device rd;
    std::mt19937 gen(rd());

    // Create a uniform distribution for the range [-1, 1]
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    // Generate a random number within the given range
    return dist(gen);
}


void feed_neural_network(s_neural_network& neural_network)
{
    for(int a = 1; a <= 3; a++)
    {
        for(int b = 0; b < neural_network.layer_size[a]; b++)
        {
            for(int c = 0; c < neural_network.layer_size[a-1]; c++)
            {
                neural_network.activation[a][b] += neural_network.activation[a-1][b] * neural_network.weights[a-1][b][c];
            }
            neural_network.activation[a][b] = sigmoid(neural_network.activation[a-1][b] + neural_network.bias[a-1][b]);
        }
    }
}


void train_network(s_neural_network& neural_network, digit_image_collection_t& training_data)
{
    for(int a = 0; a < training_data.size() - 59900; a++)
    {
        for(int b = 0; b < IMAGE_SIZE_Y; b++)
        {
            for(int c = 0; c < IMAGE_SIZE_X; c++)
            {
                neural_network.activation[0][b*IMAGE_SIZE_X+c] = training_data.at(a).matrix[b][c];
            }
        }
        feed_neural_network(neural_network);
        neural_network.avg_cost += calculate_cost(neural_network, training_data.at(a));
    }
    neural_network.avg_cost = neural_network.avg_cost / 100;
}


float calculate_cost(s_neural_network& neural_network, digit_image_t training_data)
{
    float expected_cost[] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    expected_cost[training_data.label.c_str()[0] - '0'] = 1.0f;
    float cost = 0.f;
    for(int a = 0; a < neural_network.layer_size[3]; a++)
    {
        cost += (neural_network.activation[3][a] - expected_cost[a]) * (neural_network.activation[3][a] - expected_cost[a]);
    }
    return cost;
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
    return 1 / (1 + exp(-x));
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

float calculate_cost(s_neural_network* neural_network)
{
    return 0.0f;
}

