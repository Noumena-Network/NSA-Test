FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Install Python, build dependencies and tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    cmake \
    ninja-build \
    ccache \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Install PyTorch nightly for CUDA 12.8
RUN pip install --break-system-packages --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install Triton build dependencies
RUN pip install --break-system-packages pybind11 wheel

# Build Triton from source with B200 SM 10.0a support
WORKDIR /tmp
RUN git clone https://github.com/triton-lang/triton.git && \
    cd triton && \
    git checkout main && \
    TRITON_CODEGEN_CUDA_COMPUTE_CAPABILITIES="100a" \
    CUDA_ARCHITECTURES="100a" \
    TORCH_CUDA_ARCH_LIST="10.0a" \
    pip install --break-system-packages -e . && \
    cd / && \
    rm -rf /tmp/triton

# Install test dependencies
RUN pip install --break-system-packages pytest

# Copy NSA code
WORKDIR /workspace
COPY . /workspace/nsa-test/

# Set environment variables
ENV PYTHONPATH=/workspace/nsa-test:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="10.0a"
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# Create test runner script
RUN cat > /workspace/run_tests.sh << 'EOF'
#!/bin/bash
set -e

echo "=== NSA B200 Test Suite ==="
echo "Date: $(date)"
echo ""

# Verify environment
echo "=== Environment Check ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
echo ""
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import triton; print(f'Triton: {triton.__version__}')"
python -c "import torch; props = torch.cuda.get_device_properties(0); print(f'GPU: {props.name}, Compute: {props.major}.{props.minor}, Shared memory: {props.shared_memory_per_block} bytes')"

cd /workspace/nsa-test

# Run tests
echo ""
echo "=== Running Tests ==="

# Track failures
FAILED=0

echo "1. Production Accuracy Test"
python -m pytest tests/test_accuracy.py -vs || FAILED=$((FAILED + 1))

echo ""
echo "2. Specification Conformance"
python -m pytest tests/test_spec_algorithms.py tests/test_spec_blocks.py -vs || FAILED=$((FAILED + 1))

echo ""
echo "3. Kernel Contracts"
python -m pytest tests/test_kernel_contracts.py -vs || FAILED=$((FAILED + 1))

echo ""
echo "4. Branch Tests"
python -m pytest tests/test_branches.py -vs || FAILED=$((FAILED + 1))

echo ""
echo "=== SUMMARY ==="
if [ $FAILED -eq 0 ]; then
    echo "✅ ALL CRITICAL TEST SUITES PASSED!"
    echo "Note: Skipping final 'run all tests' due to known CUDA context poisoning issue."
    exit 0
else
    echo "❌ $FAILED test suite(s) failed"
    exit 1
fi
EOF
RUN chmod +x /workspace/run_tests.sh

WORKDIR /workspace/nsa-test
CMD ["/workspace/run_tests.sh"]