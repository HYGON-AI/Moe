# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT


cmake_minimum_required(VERSION 3.10)

set(MOE_C_HIP_COMPILER "aicc" CACHE STRING "HIP compiler used for local moe_c builds")
find_program(MOE_C_HIP_COMPILER_PATH NAMES ${MOE_C_HIP_COMPILER} aicc hipcc)
if(NOT MOE_C_HIP_COMPILER_PATH)
    message(FATAL_ERROR "Unable to find a HIP compiler. Tried ${MOE_C_HIP_COMPILER}, aicc, hipcc.")
endif()

set(CMAKE_C_COMPILER "${MOE_C_HIP_COMPILER_PATH}" CACHE FILEPATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "${MOE_C_HIP_COMPILER_PATH}" CACHE FILEPATH "CXX compiler" FORCE)

project(moe_wna16 VERSION 1.0.0 LANGUAGES C CXX)
message(STATUS "MOE_C_HIP_COMPILER_PATH: ${MOE_C_HIP_COMPILER_PATH}")

if(DEFINED ENV{ROCM_PATH})
    set(HIP_ROOT_DIR "$ENV{ROCM_PATH}")
else()
    set(HIP_ROOT_DIR "/opt/dtk")
endif()

set(Python3_EXECUTABLE ${PYTHON_PATH})
find_package(Python3 REQUIRED)
set(PYTHON_VERSION ${Python3_VERSION})

string(REPLACE "." ";" VERSION_LIST ${PYTHON_VERSION})
list(GET VERSION_LIST 0 PYTHON_VERSION_MAJOR)
list(GET VERSION_LIST 1 PYTHON_VERSION_MINOR)
set(PYTHON_VERSION "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
message(STATUS "Python version: ${PYTHON_VERSION}")
find_package(Torch REQUIRED)
find_package(Threads REQUIRED)
if(TARGET MIOpen)
    get_target_property(MIOPEN_LINK_LIBRARIES MIOpen INTERFACE_LINK_LIBRARIES)
    if(MIOPEN_LINK_LIBRARIES)
        foreach(_bad_rt
            "/usr/lib64/librt.so"
            "/usr/lib/x86_64-linux-gnu/librt.so"
            "/lib/x86_64-linux-gnu/librt.so"
        )
            list(REMOVE_ITEM MIOPEN_LINK_LIBRARIES "${_bad_rt}")
        endforeach()
        list(APPEND MIOPEN_LINK_LIBRARIES rt)
        set_property(TARGET MIOpen PROPERTY INTERFACE_LINK_LIBRARIES "${MIOPEN_LINK_LIBRARIES}")
    endif()
endif()

find_library(PYTHON_LIB python${PYTHON_VERSION} PATHS ${PYTHON_LIB_PATH})
message(STATUS "PYTHON_VERSION: ${PYTHON_VERSION}")
message(STATUS "PYTHON_LIB_PATH: ${PYTHON_LIB_PATH}")
message(STATUS "PYTHON_LIB: ${PYTHON_LIB}")
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
message("TORCH_INSTALL_PREFIX = ${TORCH_INSTALL_PREFIX}")

# aiter JIT builds with --offload-arch=native by default. Do the same here
# unless the caller explicitly requests a target architecture list.
set(MOE_C_HIP_ARCHITECTURES "" CACHE STRING "ROCm offload architectures for local moe_c builds")
if(NOT "${MOE_C_HIP_ARCHITECTURES}" STREQUAL "")
    set(HIP_ARCHITECTURES ${MOE_C_HIP_ARCHITECTURES})
elseif(DEFINED ENV{PYTORCH_ROCM_ARCH})
    set(HIP_ARCHITECTURES $ENV{PYTORCH_ROCM_ARCH})
else()
    set(HIP_ARCHITECTURES native)
endif()

string(REPLACE ";" "," ARCH_WITH_COMMAS "${HIP_ARCHITECTURES}")
string(REPLACE "gfx" "" TARGET_NUM "${ARCH_WITH_COMMAS}")

set(MOE_C_VGPR_ARCHITECTURES "${HIP_ARCHITECTURES}")
if("${MOE_C_VGPR_ARCHITECTURES}" STREQUAL "native")
    # Default native builds use the 768 VGPR hint. Explicit gfx92x targets still
    # select 512 below, while explicit gfx936/gfx938 targets select 768.
    set(MOE_C_VGPR_ARCHITECTURES "gfx938")
endif()
string(REPLACE ";" "," MOE_C_VGPR_ARCH_WITH_COMMAS "${MOE_C_VGPR_ARCHITECTURES}")
string(REPLACE "gfx" "" MOE_C_VGPR_TARGET_NUM "${MOE_C_VGPR_ARCH_WITH_COMMAS}")

message(STATUS "TARGET_NUM: ${TARGET_NUM}")
message(STATUS "MOE_C_VGPR_TARGET_NUM: ${MOE_C_VGPR_TARGET_NUM}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(TARGET_DIR ${CMAKE_BINARY_DIR})
file(MAKE_DIRECTORY ${TARGET_DIR})

set(COPY_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/copy_if_changed.sh")
set(SHARED_CSRC_DIR "${CMAKE_SOURCE_DIR}/csrc_for_aiter")
set(LOCAL_CSRC_DIR "${CMAKE_SOURCE_DIR}/csrc")
set(BUILD_CSRC_DIR "${CMAKE_BINARY_DIR}/csrc")

file(GLOB_RECURSE SHARED_SOURCE_FILES LIST_DIRECTORIES false CONFIGURE_DEPENDS "${SHARED_CSRC_DIR}/*")
file(GLOB LOCAL_SOURCE_FILES LIST_DIRECTORIES false CONFIGURE_DEPENDS "${LOCAL_CSRC_DIR}/*")
file(GLOB LOCAL_CU_FILES LIST_DIRECTORIES false CONFIGURE_DEPENDS "${LOCAL_CSRC_DIR}/*.cu")
set(SOURCE_FILE ${LOCAL_CU_FILES})

add_custom_command(
    OUTPUT ${TARGET_DIR}/copied
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${BUILD_CSRC_DIR}
    COMMAND bash ${COPY_SCRIPT} ${SHARED_CSRC_DIR} ${BUILD_CSRC_DIR}
    COMMAND bash ${COPY_SCRIPT} ${LOCAL_CSRC_DIR} ${BUILD_CSRC_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${TARGET_DIR}/copied
    DEPENDS ${SHARED_SOURCE_FILES} ${LOCAL_SOURCE_FILES}
    COMMENT "Composing local build csrc from csrc_for_aiter plus local entry files"
)

add_custom_target(copy_sources DEPENDS ${TARGET_DIR}/copied)
include(./cmake/utils.cmake)

message("TARGET_DIR = ${TARGET_DIR}")
message("SOURCE_FILE = ${SOURCE_FILE}")
set(SOURCE_FILE_LIST)
foreach(CU_FILE ${SOURCE_FILE})
    get_filename_component(CU_FILE_NAME ${CU_FILE} NAME)
    list(APPEND SOURCE_FILE_LIST "${BUILD_CSRC_DIR}/${CU_FILE_NAME}")
endforeach()
string(REPLACE ";" " " SOURCE_FILE_LIST "${SOURCE_FILE_LIST}")
message(STATUS "SOURCE_FILE_LIST: ${SOURCE_FILE_LIST}")
separate_arguments(SOURCE_FILE_LIST_AS_LIST UNIX_COMMAND "${SOURCE_FILE_LIST}")
message("SOURCE_FILE_LIST_AS_LIST = ${SOURCE_FILE_LIST_AS_LIST}")
set(SOURCE_FILE_LIST_BUILD_PREFIX ${SOURCE_FILE_LIST_AS_LIST})
message("SOURCE_FILE_LIST_BUILD_PREFIX = ${SOURCE_FILE_LIST_BUILD_PREFIX}")
set(HIPIFY_PROJECT_DIR ${BUILD_CSRC_DIR})
hipify_sources_target(HIP_FILE _${CMAKE_PROJECT_NAME} "${SOURCE_FILE_LIST_BUILD_PREFIX}")
message("HIP_FILE = ${HIP_FILE}")
add_custom_target(
    hipify
)
add_dependencies(hipify_${CMAKE_PROJECT_NAME} copy_sources)
add_dependencies(hipify hipify_${CMAKE_PROJECT_NAME})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -O3 -ffast-math -fno-strict-return -Wno-c++17-extensions -Wunused-result -w -fPIC -DUSE_ROCM --offload-arch=${ARCH_WITH_COMMAS} -DTARGET=${TARGET_NUM}")

set(ROCM_INC ${HIP_ROOT_DIR}/include)
set(PYTHON_INC ${PYTHON_INCLUDE_PATH}/)
set(SOURCE_INC ${BUILD_CSRC_DIR})
set(SOURCE_INC_FOR_AITER ${CMAKE_CURRENT_LIST_DIR}/csrc_for_aiter)
set(WHL_INC ${CMAKE_CURRENT_LIST_DIR}/csrc)
set(PYTORCH_ROOT ${TORCH_INCLUDE_PATH})
if(EXISTS "${PYTORCH_ROOT}/torch/include/torch/extension.h")
    set(PYTORCH_ROOT "${PYTORCH_ROOT}/torch")
endif()
set(PYTORCH_INC1  ${PYTORCH_ROOT}/include)
set(PYTORCH_INC2 ${PYTORCH_ROOT}/include/torch/csrc/api/include)
set(PYBIND_INC ${PYBIND11_INCLUDE_PATH}/include/pybind11)

set(MOE_C_VGPR_FLAG "-support-512-vgprs=true")
if("${MOE_C_VGPR_TARGET_NUM}" MATCHES "(^|,)936(,|$)" OR "${MOE_C_VGPR_TARGET_NUM}" MATCHES "(^|,)938(,|$)")
    set(MOE_C_VGPR_FLAG "-support-768-vgprs=true")
endif()
message(STATUS "MOE_C_VGPR_FLAG: ${MOE_C_VGPR_FLAG}")

set(HIP_COMPILE_FLAGS ${HIP_COMPILE_FLAGS} -std=c++17 -mllvm ${MOE_C_VGPR_FLAG} -mllvm -disable-machine-sink  -mcode-object-version=4 -ffast-math -g -O3 -fno-strict-return -Wno-c++17-extensions -Wunused-result -w -fPIC --offload-arch=${ARCH_WITH_COMMAS} -I${PYTHON_INC} -I${SOURCE_INC} -I${SOURCE_INC_FOR_AITER} -I${PYTORCH_INC1} -I${PYTORCH_INC2} -I${PYBIND_INC} -I${WHL_INC} -I${ROCM_INC} -DUSE_ROCM)
set(CMAKE_CXX_FLAGS ${HIP_COMPILE_FLAGS})

function(configure_moe_c_library TARGET_NAME OUTPUT_NAME)
    target_link_libraries(${TARGET_NAME} PRIVATE
        ${TORCH_LIBRARIES}
        ${TORCH_PYTHON_LIBRARY}
        Threads::Threads
        ${PYTHON_LIB}
    )
    target_include_directories(${TARGET_NAME} PRIVATE
        ${PYTHON_INC}
        ${SOURCE_INC}
        ${SOURCE_INC_FOR_AITER}
        ${WHL_INC}
        ${PYTORCH_INC1}
        ${PYTORCH_INC2}
        ${PYBIND_INC}
    )
    set_target_properties(${TARGET_NAME} PROPERTIES
        LINKER_LANGUAGE CXX
        LINK_DEPENDS_NO_SHARED TRUE
        OUTPUT_NAME "${OUTPUT_NAME}"
    )
endfunction()

function(add_moe_c_precision_target TARGET_NAME OUTPUT_NAME ENTRY_NAME)
    set(PRECISION_CU "${BUILD_CSRC_DIR}/${ENTRY_NAME}.cu")
    set(PRECISION_HIP "${BUILD_CSRC_DIR}/${ENTRY_NAME}.hip")
    set(PRECISION_OBJECT "${CMAKE_BINARY_DIR}/csrc/${TARGET_NAME}.o")
    set(PRECISION_ASM "${CMAKE_BINARY_DIR}/csrc/${TARGET_NAME}.s")
    add_custom_command(
        OUTPUT ${PRECISION_HIP}
        COMMAND bash ${CMAKE_SOURCE_DIR}/cmake/ensure_hip_entry.sh ${PRECISION_CU} ${PRECISION_HIP}
        DEPENDS hipify ${PRECISION_CU}
        COMMENT "Ensuring ${PRECISION_HIP} exists for split precision build"
    )
    add_custom_command(
        OUTPUT ${PRECISION_OBJECT}
        COMMAND ${MOE_C_HIP_COMPILER_PATH} ${HIP_COMPILE_FLAGS} -x hip -c ${PRECISION_HIP} -o ${PRECISION_OBJECT}
        DEPENDS ${PRECISION_HIP}
        COMMENT "Building ${TARGET_NAME} from ${PRECISION_HIP}"
    )
    add_custom_target(${TARGET_NAME}_kernel_compile DEPENDS ${PRECISION_OBJECT})
    add_dependencies(${TARGET_NAME}_kernel_compile hipify)
    add_custom_command(
        OUTPUT ${PRECISION_ASM}
        COMMAND ${MOE_C_HIP_COMPILER_PATH} ${HIP_COMPILE_FLAGS} -x hip -S ${PRECISION_HIP} -o ${PRECISION_ASM}
        DEPENDS ${PRECISION_HIP}
        COMMENT "Building ${TARGET_NAME} assembly from ${PRECISION_HIP}"
    )
    add_custom_target(${TARGET_NAME}_asm DEPENDS ${PRECISION_ASM})
    add_dependencies(${TARGET_NAME}_asm hipify)

    set_source_files_properties(${PRECISION_OBJECT} PROPERTIES GENERATED TRUE)
    add_library(${TARGET_NAME} SHARED EXCLUDE_FROM_ALL)
    add_dependencies(${TARGET_NAME} ${TARGET_NAME}_kernel_compile copy_sources)
    target_sources(${TARGET_NAME} PRIVATE ${PRECISION_OBJECT})
    configure_moe_c_library(${TARGET_NAME} ${OUTPUT_NAME})
endfunction()

add_moe_c_precision_target(moe_c_common moe_c_common moe_c_common)
add_moe_c_precision_target(moe_w8a8_int8 moe_w8a8_int8 moe_c_w8a8_int8)
add_moe_c_precision_target(moe_w8a8_fp8 moe_w8a8_fp8 moe_c_w8a8_fp8)
add_moe_c_precision_target(moe_w4a8 moe_w4a8 moe_c_w4a8)
add_moe_c_precision_target(moe_w4a16 moe_w4a16 moe_c_w4a16)
add_moe_c_precision_target(moe_w8a16 moe_w8a16 moe_c_w8a16)
add_moe_c_precision_target(moe_wna16_only moe_wna16_only moe_c_wna16)
add_moe_c_precision_target(moe_wfp4a16_groupwise moe_wfp4a16_groupwise moe_c_wfp4a16)
add_moe_c_precision_target(moe_wfp4a8_channelwise moe_wfp4a8_channelwise moe_c_wfp4a8_channelwise)
add_moe_c_precision_target(moe_wfp4a8_groupwise moe_wfp4a8_groupwise moe_c_wfp4a8_groupwise)

add_custom_target(moe_c_split
    DEPENDS
        moe_c_common
        moe_w8a8_int8
        moe_w8a8_fp8
        moe_w4a8
        moe_w4a16
        moe_w8a16
        moe_wna16_only
        moe_wfp4a16_groupwise
        moe_wfp4a8_channelwise
        moe_wfp4a8_groupwise
)

add_custom_target(clean_all
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_BINARY_DIR}/*"
    COMMENT "Remove generated build files"
)
