#include "tensor.hpp"
#include "relu.hpp"
#include "conv2d.hpp"
#include <iostream>
#include <cassert>

int main() {
    Tensor t(1, 3, 224, 224);
    assert(t.size() == 1 * 3 * 224 * 224);
    std::cout << "Size test succeded\n";
    assert(t.index(0, 1, 0, 1) == 50177);
    std::cout << "Index test succeded\n";
    assert(t.at(0, 0, 0, 0) == 0);
    assert(t.at(0, 1, 0, 1) == 0);
    std::cout << "At test succeded\n";
    t.at(0,0,0,0) = -5; t.at(0,0,0,1) = 3;
    Relu relu(t.data);
    relu.gates();
    assert(relu.data[0] == 0);
    assert(relu.data[1] == 3);
    std::cout << "Relu test succeded\n";

    // --- Conv2d tests: 4x4 image filled with 1..16 row by row ---
    Tensor img(1, 1, 4, 4);
    for (int i = 0; i < 16; ++i) img.data[i] = float(i + 1);

    // identity kernel (1 at center, 0 elsewhere): output must equal input exactly
    Tensor id_k(1, 1, 3, 3);
    id_k.at(0, 0, 1, 1) = 1;
    Tensor id_out = Conv2d(id_k).forward(img);
    assert(id_out.N == 1 && id_out.C == 1 && id_out.H == 4 && id_out.W == 4);
    for (int i = 0; i < 16; ++i) assert(id_out.data[i] == img.data[i]);
    std::cout << "Conv2d identity test succeded\n";

    // all-ones kernel: each output = sum of the 3x3 neighborhood
    Tensor ones_k(1, 1, 3, 3);
    for (float& v : ones_k.data) v = 1;
    Tensor sum_out = Conv2d(ones_k).forward(img);
    assert(sum_out.at(0, 0, 1, 1) == 54); // 1+2+3+5+6+7+9+10+11, window fully inside
    assert(sum_out.at(0, 0, 0, 0) == 14); // 1+2+5+6, corner: 5 of 9 taps in zero padding
    std::cout << "Conv2d padding/sum test succeded\n";

    // stride 2: output shrinks to 2x2, same corner window as stride 1
    Tensor s2_out = Conv2d(ones_k, 2).forward(img);
    assert(s2_out.H == 2 && s2_out.W == 2);
    assert(s2_out.at(0, 0, 0, 0) == 14);
    std::cout << "Conv2d stride test succeded\n";

    // 2 input channels: one filter must sum over ALL input channels
    Tensor img2(1, 2, 4, 4);
    for (int i = 0; i < 16; ++i) { img2.data[i] = 1; img2.data[16 + i] = 2; } // ch0 all 1s, ch1 all 2s
    Tensor k2(1, 2, 3, 3);
    for (float& v : k2.data) v = 1;
    Tensor out2 = Conv2d(k2).forward(img2);
    assert(out2.C == 1);
    assert(out2.at(0, 0, 1, 1) == 27); // 9 taps * 1 (ch0) + 9 taps * 2 (ch1)
    std::cout << "Conv2d multi-channel test succeded\n";
}
