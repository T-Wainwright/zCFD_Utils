cmake_minimum_required(VERSION 2.8.3)

FIND_PACKAGE(PythonInterp 3.10 REQUIRED)

project(mutlscale_cpp)

if(PYTHONINTERP_FOUND)
  if(UNIX AND NOT APPLE)
    FIND_PACKAGE(Boost COMPONENTS python${PYTHON_VERSION_SUFFIX})
    FIND_PACKAGE(PythonInterp 3 REQUIRED)
    FIND_PACKAGE(PythonLibs 3.10 REQUIRED)
    FIND_PACKAGE(Boost COMPONENTS numpy)
    FIND_PACKAGE(Eigen3)
  else()
    FIND_PACKAGE(Boost COMPONENTS python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
    FIND_PACKAGE(PythonInterp 3 REQUIRED)
    FIND_PACKAGE(PythonLibs 3.10 REQUIRED)
  endif()
else()
  message("Python not found")
endif()

# if(MEM_CHECK)
# # Enable GNU address sanitizer for memory checking
# if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
# SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
# SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")
# message(STATUS "Memory checks are: ON")
# else()
# message(FATAL_ERROR "Memory checks are only supported with the GCC toolchain")
# endif()
# endif()
message(STATUS "PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
message(STATUS "PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")
message(STATUS "PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
message(STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}")

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
LINK_LIBRARIES(${Boost_LIBRARIES} ${PYTHON_LIBRARIES}) # Deprecated but so convenient!

PYTHON_ADD_MODULE(multiscale multiscale.cpp)

# find_package(OpenMP)
# if(OpenMP_CXX_FOUND)
# target_link_libraries(multiscale PUBLIC OpenMP::OpenMP_CXX)
# endif()
