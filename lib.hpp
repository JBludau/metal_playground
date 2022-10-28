#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <iostream>

std::ostream& operator<< (std::ostream& os, MTL::Device const & device);

void add_arrays (const float * inA, const float * inB, float* result, unsigned int length);

[[visible]] float add (const float A, const float B)

using TFuncSig = float(const float,const float);

[[kernel]] void add_arrays (device const float * inA,device const float * inB,device float* result,metal::visible_function_table<TFuncSig> table, uint i [[thread_position_in_grid]]);

void fill(float* b, unsigned int N,float value);

void fill_index(float* b, unsigned int N);
