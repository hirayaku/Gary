include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG main
)
set(BUILD_GMOCK OFF)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

ADD_EXECUTABLE(RadixSortTest
  RadixSortTest.cpp
)
TARGET_INCLUDE_DIRECTORIES(RadixSortTest PUBLIC common)
TARGET_LINK_LIBRARIES(RadixSortTest PUBLIC OpenMP::OpenMP_CXX gtest_main)

function(add_cuda_test TEST_NAME TEST_FILE)
  ADD_EXECUTABLE(${TEST_NAME} ${TEST_FILE})
  TARGET_INCLUDE_DIRECTORIES(${TEST_NAME} PRIVATE common include)
  TARGET_LINK_LIBRARIES(${TEST_NAME} PUBLIC graphm CUB::CUB OpenMP::OpenMP_CXX gtest_main)
  TARGET_COMPILE_FEATURES(${TEST_NAME} PUBLIC cxx_std_17 cuda_std_17)
  TARGET_COMPILE_OPTIONS(${TEST_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-fopenmp>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-g -G>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RELWITHDEBINFO>>:-lineinfo>
  )
endfunction()

add_cuda_test(CUDATest CUDATest.cu)
add_cuda_test(SearchTest SearchTest.cu)
add_cuda_test(CTACommTest CTACommTest.cu)
