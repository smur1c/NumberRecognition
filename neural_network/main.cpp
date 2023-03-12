//
//  main.cpp
//  neural_network
//
//  Created by Sebastian on 09.03.23.
//

#include <iostream>
#include "neural_network.hpp"

int main(int argc, const char * argv[]) {
    s_neural_network neural_network = start(28*28);
    feed_neural_network(neural_network); 
    return 0;
}
