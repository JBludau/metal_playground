#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <iostream>
#include <chrono>

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

const char* my_library_impl = R"(
[[visible]] float add (const float A, const float B)
{
  return A + B;
}

using TFuncSig = float(const float,const float);

[[kernel]] void add_arrays (device const float * inA,device const float * inB,device float* result,metal::visible_function_table<TFuncSig> table, uint i [[thread_position_in_grid]])
{
    //result[i] = add(inA[i],inB[i]);
    result[i] = table[0](inA[i],inB[i]);
}
)";

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
 
void driver_device(unsigned int N,unsigned int numRepetitions)
{
    using NS::StringEncoding::UTF8StringEncoding;

     MTL::Device* device = MTL::CreateSystemDefaultDevice();
     std::cout << *device << std::endl;

     NS::Array* devices =  MTL::CopyAllDevices();
     for (unsigned i=0;i<devices->count();++i)
  {
    std::cout << "device Number " << i << '\n';
    std::cout <<  *(devices->object<MTL::Device>(i)) << '\n'; 
  }

     MTL::Buffer* inA = device->newBuffer(N*sizeof(float),MTL::ResourceStorageModeShared);
     MTL::Buffer* inB = device->newBuffer(N*sizeof(float),MTL::ResourceStorageModeShared);
     MTL::Buffer* result = device->newBuffer(N*sizeof(float),MTL::ResourceStorageModeShared);

     fill_index(static_cast<float*>(inA->contents()),N);
     fill(static_cast<float*>(inB->contents()),N,0);
     fill(static_cast<float*>(result->contents()),N,0);

     NS::Error* pError = nullptr;
      MTL::Library* library = device->newLibrary( NS::String::string(my_library_impl, UTF8StringEncoding), nullptr, &pError );
    if ( !library )
    {
        __builtin_printf( "%s", pError->localizedDescription()->utf8String() );
        assert( false );
    }

    MTL::Function* myAdd = library->newFunction( NS::String::string("add", UTF8StringEncoding) );
    MTL::Function* myKernel = library->newFunction( NS::String::string("add_arrays", UTF8StringEncoding) );
    MTL::ComputePipelineState* computePipeline = device->newComputePipelineState(myKernel,&pError);
    MTL::VisibleFunctionTableDescriptor* visFuncTblDescr = MTL::VisibleFunctionTableDescriptor::alloc();
    visFuncTblDescr->setFunctionCount(1);
    MTL::VisibleFunctionTable* visFuncTable = computePipeline->newVisibleFunctionTable(visFuncTblDescr);
    MTL::FunctionHandle* myAddHandle = computePipeline->functionHandle(myAdd);
    visFuncTable->setFunction(myAddHandle,0);
  
    MTL::CommandQueue* queue = device->newCommandQueue();

    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int run=0;run<numRepetitions;++run)
    {
    MTL::CommandBuffer* buffer = queue->commandBuffer();

    MTL::ComputeCommandEncoder* encoder = buffer->computeCommandEncoder();

    encoder->setComputePipelineState(computePipeline);
    encoder->setBuffer(inA,0,0);
    encoder->setBuffer(inB,0,1);
    encoder->setBuffer(result,0,2);
    encoder->setVisibleFunctionTable(visFuncTable,0);

    MTL::Size gridSize(N,1,1);

    NS::UInteger threadGroupSize = computePipeline->maxTotalThreadsPerThreadgroup();

    if (threadGroupSize > N)
      threadGroupSize = N;

    encoder->dispatchThreads(gridSize,MTL::Size(threadGroupSize,1,1));

    encoder->endEncoding();
    
    buffer->commit();

    buffer->waitUntilCompleted();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double> >(end - start);


    float* raw_result = static_cast<float *>(result->contents());
    float* raw_A = static_cast<float *>(inA->contents());
    float* raw_B = static_cast<float *>(inB->contents());

    for (unsigned i = 0;i<N;++i)
    {
      if(raw_result[i]!=(i)) 
      {
        std::cout << "value at index " << i << " " << raw_result[i] <<
        " A " << raw_A[i] << 
        " B " << raw_B[i] << '\n';
      } 
    }

    std::cout << "This took " << duration.count() << " seconds for " << numRepetitions << " repetitions on the device: Bandwidth " << 3.0 * (double)N * sizeof(float) * numRepetitions / duration.count() << "\n";
}

void driver_host(unsigned int N, unsigned int numRepetitions)
{
  
  float* inA = new float[N];
  float* inB = new float[N];
  float* result = new float[N];

     fill_index(inA,N);
     fill_index(inB,N);
     fill(result,N,0);

    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int run=0;run<numRepetitions;++run)
    {
      add_arrays(inA,result,result,N); 
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::duration<double> >(end - start);

    std::cout << "This took " << duration.count() << " seconds for " << numRepetitions << " repetitions on the host: Bandwidth " << 3.0 * (double)N * sizeof(float) * numRepetitions / duration.count() << "\n";
}


int main(int argc, const char * argv[]) {

    unsigned int N = 1'000'000;
    // unsigned int N = 100;
    unsigned int numRepetitions = 10000;

    driver_device(N,numRepetitions);

    driver_host(N,numRepetitions);

    return 0;
}
