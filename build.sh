#!/bin/bash

# Set the source file and output binary
SOURCE="celu.cu"
OUTPUT="celu"

# Compile the CUDA code with the appropriate architecture for A10 (sm_86)
nvcc -arch=sm_86 -o $OUTPUT $SOURCE

./$OUTPUT

# Print success message
echo "Compilation complete. Output binary: $OUTPUT"