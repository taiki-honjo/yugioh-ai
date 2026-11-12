cmake_minimum_required(VERSION 3.16)
project(combo_benchmark LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Lua 5.4 REQUIRED)
find_package(glog REQUIRED)
find_package(Threads REQUIRED)
pybind11_add_module(ygopro_ygoenv NO_EXTRAS agent/ygoenv/ygoenv/ygopro/ygopro.cpp)
target_include_directories(ygopro_ygoenv PRIVATE ${CMAKE_SOURCE_DIR} agent/ygoenv include
    fmt/include sqlitecpp/include unordered-dense/include /usr/include/concurrentqueue ${LUA_INCLUDE_DIR})
target_link_libraries(ygopro_ygoenv PRIVATE /home/hatk8/projects/yugioh-ai/experiments/elfnotes-20261106/runtime/build/libocgcore.a
    /home/hatk8/projects/yugioh-ai/experiments/elfnotes-20261106/runtime/build/sqlitecpp/libSQLiteCpp.a /home/hatk8/projects/yugioh-ai/experiments/elfnotes-20261106/runtime/build/fmt/libfmt.a sqlite3 ${LUA_LIBRARIES} glog::glog Threads::Threads)
set_target_properties(ygopro_ygoenv PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/agent/ygoenv/ygoenv/ygopro")
