#!/bin/bash

exe_file="copy"
src_file="copy.cu"

# Check if source file exists
if [ ! -f $src_file ]; then
    echo "Error: Source file $src_file does not exist"
    exit 1
fi

# Compile if executable doesn't exist
if [ ! -f $exe_file ]; then
    echo "Compiling $src_file..."
    nvcc -o $exe_file $src_file
    if [ $? -ne 0 ]; then
        echo "Error: Compilation failed"
        exit 1
    fi
fi

# Check if ncu is available
NCU_AVAILABLE=false
# Check if sudo is available
SUDO_AVAILABLE=false
if command -v sudo &> /dev/null; then
    SUDO_AVAILABLE=true
fi

# Allow user to set USE_SUDO_NCU=1 environment variable to use sudo for ncu
if [ "${USE_SUDO_NCU:-0}" = "1" ]; then
    if [ "$SUDO_AVAILABLE" = true ]; then
        USE_SUDO_NCU=true
    else
        echo "Warning: USE_SUDO_NCU=1 but sudo not available, will run ncu without sudo"
        USE_SUDO_NCU=false
    fi
else
    USE_SUDO_NCU=false
fi

if command -v ncu &> /dev/null; then
    NCU_AVAILABLE=true
    echo "ncu is available, will collect DRAM metrics"
    if [ "$USE_SUDO_NCU" = true ]; then
        echo "Note: Will use sudo for ncu"
    fi
else
    echo "Warning: ncu not found, skipping DRAM metrics collection"
    echo "Install NVIDIA Nsight Compute to enable DRAM metrics collection"
fi

echo ""
echo "=========================================="
echo "Running copy kernel with different sizes"
echo "=========================================="
echo ""

for ((n=10; n<=30; n+=2)); do
    num_elements=$((2**$n))
    
    echo "----------------------------------------"
    echo "Testing with 2^$n = $num_elements elements"
    echo "----------------------------------------"
    
    # Run the program normally
    ./$exe_file $num_elements
    
    # Run with ncu to collect DRAM metrics
    if [ "$NCU_AVAILABLE" = true ]; then
        echo ""
        echo "Collecting DRAM metrics with ncu..."
        
        # Determine ncu command (with or without sudo)
        # Only use sudo if explicitly requested AND sudo is available
        if [ "$USE_SUDO_NCU" = true ] && [ "$SUDO_AVAILABLE" = true ]; then
            NCU_CMD="sudo ncu"
        else
            NCU_CMD="ncu"
        fi
        
        # Run ncu and capture output
        # Note: --print-summary requires an argument (per-kernel, per-gpu, etc.)
        ncu_output=$($NCU_CMD \
            --kernel-name copy_baseline \
            --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum \
            --print-summary per-kernel \
            --target-processes all \
            ./$exe_file $num_elements 2>&1)
        
        ncu_exit_code=$?
        
        # Check for command not found error (e.g., sudo not available)
        if [ $ncu_exit_code -eq 127 ]; then
            if echo "$ncu_output" | grep -q "sudo: command not found"; then
                echo ""
                echo "Warning: sudo command not found (likely in container environment)"
                echo "Trying ncu without sudo..."
                # Retry without sudo
                ncu_output=$(ncu \
                    --kernel-name copy_baseline \
                    --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum \
                    --print-summary per-kernel \
                    --target-processes all \
                    ./$exe_file $num_elements 2>&1)
                ncu_exit_code=$?
            fi
        fi
        
        # Check for permission error
        if echo "$ncu_output" | grep -q "ERR_NVGPUCTRPERM"; then
            echo ""
            echo "ERROR: Permission denied for GPU Performance Counters"
            echo "ncu requires special permissions to access GPU performance counters."
            echo ""
            if [ "$SUDO_AVAILABLE" = true ]; then
                echo "Solutions:"
                echo "  1. Run the script with sudo: sudo $0"
                echo "  2. Or set USE_SUDO_NCU=1 and run: USE_SUDO_NCU=1 $0"
                echo "  3. Or configure system permissions (see https://developer.nvidia.com/ERR_NVGPUCTRPERM)"
            else
                echo "Note: Running in container environment (sudo not available)"
                echo "If you have root privileges, try running as root user"
                echo "Or configure system permissions (see https://developer.nvidia.com/ERR_NVGPUCTRPERM)"
            fi
            echo ""
            continue
        fi
        
        if [ $ncu_exit_code -eq 0 ]; then
            echo ""
            echo "--- ncu Raw Measurement Results ---"
            echo "$ncu_output"
            echo ""
        else
            echo "Warning: ncu profiling failed (exit code: $ncu_exit_code)"
            echo ""
            echo "ncu error output:"
            echo "$ncu_output" | tail -30
            echo ""
            if ! echo "$ncu_output" | grep -q "ERR_NVGPUCTRPERM"; then
                echo "Common issues:"
                echo "  1. Library version mismatch (GLIBC/GLIBCXX) - try running in the same environment where compiled"
                echo "  2. Kernel name mismatch - verify kernel name is 'copy_baseline'"
                echo "  3. GPU not available"
            fi
        fi
    fi
    
    echo ""
done

echo "=========================================="
echo "Testing completed"
echo "=========================================="
