#!/usr/bin/env bash
set -euo pipefail
ARCH="${ARCH:-sm_86}"
OUT_LIB="${OUT_LIB:-libnaive_attention.so}"
SRCDIR="$(dirname "$0")"
NVCC_FLAGS="-O3 --std=c++17 -arch=${ARCH} -rdc=true -Xcompiler -fPIC"

nvcc ${NVCC_FLAGS} -shared "${SRCDIR}/normal_attention.cu" "${SRCDIR}/wrapper.cu" -o "${SRCDIR}/${OUT_LIB}"
echo "Built ${SRCDIR}/${OUT_LIB}"
