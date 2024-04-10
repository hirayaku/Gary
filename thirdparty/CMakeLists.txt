include(FetchContent)

FetchContent_Declare(
    rmm
    GIT_REPOSITORY https://github.com/rapidsai/rmm
    GIT_TAG        deb4729
)

option(BUILD_TESTS OFF)
FetchContent_MakeAvailable(rmm)