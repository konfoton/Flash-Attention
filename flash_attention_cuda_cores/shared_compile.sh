#!/usr/bin/env bash
set -euo pipefail
ARCH="${ARCH:-sm_86}"
OUT_LIB="${OUT_LIB:-libflash_attention_cuda_cores.so}"
SRCDIR="$(dirname "$0")"
NVCC_FLAGS="-O3 --std=c++17 -arch=${ARCH} -rdc=true -Xcompiler -fPIC"

nvcc ${NVCC_FLAGS} -shared "${SRCDIR}/cuda_flash_attention.cu" "${SRCDIR}/wrapper.cu" -o "${SRCDIR}/${OUT_LIB}"
echo "Built ${SRCDIR}/${OUT_LIB}"
