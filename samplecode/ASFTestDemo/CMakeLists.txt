cmake_minimum_required(VERSION 3.0)		#指定cmake版本 
project(AFCFaceDemo_Linux)				#指定项目的名称，一般和项目的文件夹名称对应


set(CMAKE_CXX_STANDARD 11)				#设置c++标准

#设置debug
SET(CMAKE_BUILD_TYPE "Debug")  
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")  
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")  
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

set(OpenCV_DIR /home/share/local/lib)
find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

#指定头文件目录
include_directories(./)	
include_directories(./inc)

#指定静态和动态文件目录
link_directories(./linux_so)

#该指令的主要作用就是将指定的源文件生成链接文件，然后添加到工程中去。
#arcsoft_face生成的库文件名		[STATIC | SHARED | MODULE] 库类型
add_executable(face_compare 
        ./inc/amcomdef.h
        ./inc/arcsoft_face_sdk.h
		./inc/asvloffscreen.h
		./inc/merror.h
		./face_compare.cpp)

#-fPIC	在给定的作用域内设置一个命名的属性
set_property(TARGET face_compare PROPERTY POSITION_INDEPENDENT_CODE ON)


#链接库文件
target_link_libraries(face_compare	
		arcsoft_face
		arcsoft_face_engine
		${OpenCV_LIBS}
        )