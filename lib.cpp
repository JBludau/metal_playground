#include "lib.hpp"

std::ostream& operator<< (std::ostream& os, MTL::Device const & device){
  os << "name = " << device.name()->utf8String()  << '\n';
  os << "headless = " << device.headless()  << '\n';
  os << "removable = " << device.removable()  << '\n';
  os << "lowPower = " << device.lowPower()  << '\n';
  os << "hasUnifiedMemory = " << device.hasUnifiedMemory()  << '\n';
  os << "registryID = " << device.registryID()  << '\n';
  os << "maxTransferRate = " << device.maxTransferRate()  << '\n';
  os << "peerGroupID = " << device.peerGroupID()  << '\n';
  os << "peerIndex = " << device.peerIndex()  << '\n';
  os << "supportsDynamicLibraries = " << device.supportsDynamicLibraries()  << '\n';
  os << "supportsRaytracing = " << device.supportsRaytracing()  << '\n';
  os << "supportsFunctionPointers = " << device.supportsFunctionPointers()  << '\n';
  
  return os;
}

void add_arrays (const float * inA, const float * inB, float* result, unsigned int length)
{
  for (int i=0;i<length;++i)
    
    result[i] = inA[i] + inB[i];

}

// using add_external_signature = float(float,float)
// [[visible]] float add_external(const float A, const float B)
// {
  // return A+B;
// }

[[visible]] float add (const float A, const float B)
{
  return A + B;
}

[[kernel]] void add_arrays (device const float * inA,device const float * inB,device float* result,metal::visible_function_table<TFuncSig> table, uint i [[thread_position_in_grid]])
{
    //result[i] = add(inA[i],inB[i]);
    result[i] = table[0](inA[i],inB[i]);
}

void fill(float* b, unsigned int N,float value)
{
  for (unsigned int i=0;i<N;++i)
   b[i] = value; 
}


void fill_index(float* b, unsigned int N)
{
  for (unsigned int i=0;i<N;++i)
   b[i] = i; 
}
