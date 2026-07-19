#pragma once
#include <vector>

class Tensor {
/*
 * A 4D tensor class that stores data in a contiguous memory buffer.
 * N : batch size
 * C : number of channels
 * H : height
 * W : width
 */
    public:
        int N, C, H, W;
        std::vector<float> data;

        Tensor(int n, int c, int h, int w) : N(n), C(c), H(h), W(w) {
            data.resize(static_cast<size_t>(N) * C * H * W);
        }

        int size() const {
            return N * C * H * W;
        }

        int index(int n, int c, int h, int w) const {
            return ((n * C + c) * H + h) * W + w;
        }

        float& at(int n, int c, int h, int w) {
            return data[index(n, c, h, w)];
        }

        const float& at(int n, int c, int h, int w) const {
            return data[index(n, c, h, w)];
        }
};
