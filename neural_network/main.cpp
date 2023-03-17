//
//  main.cpp
//  neural_network
//
//  Created by Sebastian on 09.03.23.
//

#include <iostream>
#include "training_data.hpp"
#include "neural_network.hpp"

int main(int argc, const char * argv[]) {
    s_neural_network neural_network = start(28*28);
    
    digit_image_collection_t training_data = load_mnist_data("/Users/Sebastian/Documents/AI/NumberRecognition/data/train-images.idx3-ubyte", "/Users/Sebastian/Documents/AI/NumberRecognition/label/train-labels.idx1-ubyte");
    digit_image_collection_t test_data = load_mnist_data("/Users/Sebastian/Documents/AI/NumberRecognition/data/t10k-images-idx3-ubyte", "/Users/Sebastian/Documents/AI/NumberRecognition/label/t10k-labels-idx1-ubyte");
    
    
    for(int i = 0; i < 50; i++)
    {
        for(int a = 1; a <= 3; a++)
        {
            for(int b = 0; b < neural_network.layer_size[a]; b++)
            {
                for(int c = 0; c < neural_network.layer_size[a-1]; c++)
                {
                    train_network(neural_network, training_data);
                    float cur_cost = neural_network.avg_cost;
                    neural_network.weights[a-1][b][c] += 0.3f;
                    train_network(neural_network, training_data);
                    float new_cost = neural_network.avg_cost;
                    if(cur_cost < new_cost)
                    {
                        neural_network.weights[a-1][b][c] -= 0.6f;
                    }
                    if(new_cost != cur_cost)
                    {
                        printf("%f ", neural_network.avg_cost);
                    }
                }
            }
        }
    }
    return 0;
}
