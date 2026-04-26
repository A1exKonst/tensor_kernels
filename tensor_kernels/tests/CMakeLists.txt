

find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(kernels_test "main_test.cpp")

target_link_libraries(kernels_test PRIVATE kernels_lib)
target_link_libraries(kernels_test PRIVATE GTest::gtest GTest::gtest_main)

enable_testing()
gtest_discover_tests(kernels_test)