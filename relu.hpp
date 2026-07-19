#pragma once

#include <vector>


class Relu {
public:
    std::vector<float> data;
    Relu(std::vector<float> data) : data(data) {}

    void gates(){
        for (float& pos : data) {
            if (pos < 0) {
                pos = 0;
            }
        }
    }
};
