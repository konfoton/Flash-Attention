#!/usr/bin/env bash
set -euo pipefail

# Config
ARCH="${ARCH:-sm_86}"
OUT_LIB="${OUT_LIB:-libflash_attention.so}"

# Files (adjust if paths differ)
SRCDIR="$(dirname "$0")"
WRAPPER_CU="${SRCDIR}/wrapper.cu"
KERNEL_CU="${SRCDIR}/tensor_flash_attention.cu"

# Compile flags
NVCC_FLAGS="-O3 --std=c++17 -arch=${ARCH} -rdc=true -Xcompiler -fPIC"

echo "Compiling shared library (${OUT_LIB}) for arch ${ARCH}..."
nvcc ${NVCC_FLAGS} -shared "${WRAPPER_CU}" "${KERNEL_CU}" -o "${OUT_LIB}"

echo "Built ${OUT_LIB}"