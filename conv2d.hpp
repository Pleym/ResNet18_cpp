#pragma once
#include "tensor.hpp"
#include <cassert>

class Conv2d {
    /*
     * 2D convolution layer, no bias (every ResNet conv is followed by a
     * BatchNorm whose beta does the shifting).
     *
     * kernel_weight reuses Tensor with axes reinterpreted as (C_out, C_in, K, K):
     *   .N = number of filters = output channels
     *   .C = input channels each filter spans (must match the input tensor)
     */
    public:
        int stride;
        int padding;
        Tensor kernel_weight; // by value: a reference member would dangle if built from a temporary

        Conv2d(const Tensor& kernel_weight, int stride = 1, int padding = 1)
            : stride(stride), padding(padding), kernel_weight(kernel_weight) {}

        Tensor forward(const Tensor& image) {
            assert(image.C == kernel_weight.C); // filters must span exactly the input's channels

            int H_out = (image.H + 2 * padding - kernel_weight.H) / stride + 1;
            int W_out = (image.W + 2 * padding - kernel_weight.W) / stride + 1;

            Tensor output(image.N, kernel_weight.N, H_out, W_out);

            for (int n = 0; n < image.N; ++n){
                for (int co = 0; co < kernel_weight.N; ++co){ // one output channel per filter
                    for (int h = 0; h < H_out; ++h){
                        for (int w = 0; w < W_out; ++w){
                            float sum = 0;
                            // one output value = dot product between filter co and the
                            // K x K patch of the input, summed over ALL input channels
                            for (int ci = 0; ci < image.C; ++ci){
                                for (int kh = 0; kh < kernel_weight.H; ++kh){
                                    for (int kw = 0; kw < kernel_weight.W; ++kw){
                                        int h_in = h * stride - padding + kh;
                                        int w_in = w * stride - padding + kw;
                                        // position falling outside the image = zero padding, skip
                                        if (h_in >= 0 && h_in < image.H && w_in >= 0 && w_in < image.W){
                                            sum += image.at(n, ci, h_in, w_in) * kernel_weight.at(co, ci, kh, kw);
                                        }
                                    }
                                }
                            }
                            output.at(n, co, h, w) = sum;
                        }
                    }
                }
            }
            return output;
        }
};
