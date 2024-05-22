cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 89)
endif()

project(GraphM LANGUAGES CUDA CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)

find_package(Filesystem REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
# find_package(spdlog REQUIRED)

add_library(utils SHARED common/Timer.cpp common/Utils.cpp)
target_include_directories(utils PUBLIC common)
target_link_libraries(utils PUBLIC OpenMP::OpenMP_CXX CUDA::cudart CUDA::nvToolsExt)
target_compile_features(utils PUBLIC cxx_std_17 cuda_std_17)

add_library(graphm SHARED src/Graph.cpp src/DeviceGraph.cu src/GraphQueryHom.cu)
target_include_directories(graphm PUBLIC include)
target_link_libraries(graphm PUBLIC std::filesystem OpenMP::OpenMP_CXX CUDA::cudart rmm::rmm)
target_link_libraries(graphm PUBLIC fmt::fmt cereal::cereal utils)
target_compile_features(graphm PUBLIC cxx_std_17 cuda_std_17)
target_compile_options(graphm PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp,-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-g -G>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RELWITHDEBINFO>>:-lineinfo>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RELEASE>>:-O3>
)

add_executable(convert convert.cpp)
target_link_libraries(convert PUBLIC graphm)
target_compile_features(convert PUBLIC cxx_std_17 cuda_std_17)

add_executable(main main.cu)
target_link_libraries(main PUBLIC graphm)
target_compile_features(main PUBLIC cxx_std_17 cuda_std_17)

add_subdirectory(thirdparty)
add_subdirectory(test)
