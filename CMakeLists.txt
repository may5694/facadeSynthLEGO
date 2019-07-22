cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_PREFIX_PATH /usr/local/libtorch_cuda)
set(CMAKE_CXX_STANDARD 14)

project(facadeSynthLEGO)
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

find_package(RapidJSON REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GDAL REQUIRED)
find_package(dlib 19.17 REQUIRED)

set(SOURCES
	src/main.cpp
	src/building.cpp
	src/dn_predict.cpp
	src/utils.cpp)

add_executable(facadeSynthLEGO ${SOURCES})
target_link_libraries(facadeSynthLEGO ${TORCH_LIBRARIES})
target_link_libraries(facadeSynthLEGO ${OpenCV_LIBS})
target_link_libraries(facadeSynthLEGO ${GDAL_LIBRARY})
target_link_libraries(facadeSynthLEGO stdc++fs)
target_link_libraries(facadeSynthLEGO ${DLIB_LIBRARIES} dlib cblas)

install(TARGETS facadeSynthLEGO DESTINATION .)
