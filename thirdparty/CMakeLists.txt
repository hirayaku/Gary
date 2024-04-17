include(FetchContent)

FetchContent_Declare(
    rmm
    GIT_REPOSITORY https://github.com/rapidsai/rmm
    GIT_TAG        deb4729
)

set(BUILD_TESTS OFF)
set(BUILD_BENCHMARKS OFF)
FetchContent_MakeAvailable(rmm)

FetchContent_Declare(
    cereal
    GIT_REPOSITORY https://github.com/USCiLab/cereal
    GIT_TAG        ebef1e9
)

set(JUST_INSTALL_CEREAL ON)
set(SKIP_PORTABILITY_TEST ON)
FetchContent_MakeAvailable(cereal)
