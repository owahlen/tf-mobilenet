cmake_minimum_required(VERSION 3.14)

include_directories(/usr/local/cuda/targets/x86_64-linux/include)
add_library(flattenconcat SHARED FlattenConcat.cpp)
target_link_libraries(flattenconcat nvinfer cublas)
