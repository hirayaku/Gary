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
