#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <string>

constexpr int IMAGE_SIZE_X = 28;
constexpr int IMAGE_SIZE_Y = 28;

typedef struct {
    float matrix[IMAGE_SIZE_Y][IMAGE_SIZE_X];
    std::string label;
} digit_image_t;

typedef std::vector<digit_image_t> digit_image_collection_t;

//a batch handler has a pointer to a collection
//its purpose is to get sublists of said collection


void print_digit_image(const digit_image_t& data);
digit_image_collection_t load_mnist_data(std::string data_file_path, std::string label_file_path);
