cmake_minimum_required(VERSION 3.20.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
unset(CONFIG_STD_CPP11)
set(CONFIG_STD_CPP17 y)
    

find_package(Zephyr PATHS /home/EdgeVolution/tflite/zephyr)
#find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(EdgeVolution_tflite_micro)

# These samples use local static initialization. Since Zephyr doesn't support the
# C++ ABI for thread-safe initialization of local statics and the constructors don't
# appear to require thread safety, we turn it off in the C++ compiler.
# set(NO_THREADSAFE_STATICS $<TARGET_PROPERTY:compiler-cpp,no_threadsafe_statics>)
#zephyr_compile_options($<$<COMPILE_LANGUAGE:CXX>:${NO_THREADSAFE_STATICS}>)
# set_property(TARGET tgt PROPERTY CXX_STANDARD 17)


file(GLOB app_sources src/*)
target_sources(app PRIVATE ${app_sources})
