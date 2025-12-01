#!/usr/bin/env python3
"""
Parse ncu output from run_copy.sh log file and plot bandwidth vs data size.
"""

import re
import sys
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(log_file):
    """
    Parse the log file to extract:
    - Data size (number of elements * sizeof(float))
    - DRAM bytes read (unit varies: byte, Kbyte, Mbyte, Gbyte)
    - DRAM bytes write (unit varies: byte, Kbyte, Mbyte, Gbyte)
    - GPU time duration (unit varies: us, ms)
    
    All units are converted to bytes and seconds respectively before calculating bandwidth.
    
    Returns a list of dictionaries with the extracted data.
    """
    results = []
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Split content by test sections
    # Pattern: "Testing with 2^n = num_elements elements"
    test_pattern = r'Testing with 2\^(\d+) = (\d+) elements'
    
    # Find all test sections
    test_matches = list(re.finditer(test_pattern, content))
    
    for i, test_match in enumerate(test_matches):
        start_pos = test_match.start()
        # Find the end of this test section (start of next test or end of file)
        if i + 1 < len(test_matches):
            end_pos = test_matches[i + 1].start()
        else:
            end_pos = len(content)
        
        test_section = content[start_pos:end_pos]
        
        # Extract test parameters
        n = int(test_match.group(1))
        num_elements = int(test_match.group(2))
        
        # Calculate data size in bytes (float = 4 bytes)
        data_size_bytes = num_elements * 4
        
        # Look for ncu output section
        if '--- ncu Raw Measurement Results ---' not in test_section:
            print(f"Warning: No ncu output found for 2^{n} = {num_elements} elements")
            continue
        
        # Extract metrics from ncu output
        # Pattern for metric lines: "dram__bytes_read.sum         Gbyte    4.30    4.30    4.30"
        # The format is: metric_name, unit, min, max, average
        metrics_pattern = r'(dram__bytes_read\.sum|dram__bytes_write\.sum|gpu__time_duration\.sum)\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
        
        metrics = {}
        for metric_match in re.finditer(metrics_pattern, test_section):
            metric_name = metric_match.group(1)
            metric_unit = metric_match.group(2)
            # Use Average value (group 5, the last column)
            metric_value = float(metric_match.group(5))
            
            metrics[metric_name] = {
                'value': metric_value,
                'unit': metric_unit
            }
        
        # Check if we got all required metrics
        required_metrics = ['dram__bytes_read.sum', 'dram__bytes_write.sum', 'gpu__time_duration.sum']
        if not all(m in metrics for m in required_metrics):
            print(f"Warning: Missing metrics for 2^{n} = {num_elements} elements")
            print(f"  Found metrics: {list(metrics.keys())}")
            continue
        
        # Convert units to bytes and seconds
        # Note: ncu uses decimal units (SI standard)
        # Memory units: byte, Kbyte, Mbyte, Gbyte
        def convert_bytes_to_bytes(value, unit):
            """Convert memory units to bytes."""
            unit_lower = unit.lower()
            if unit_lower == 'byte':
                return value
            elif unit_lower == 'kbyte':
                return value * 1e3
            elif unit_lower == 'mbyte':
                return value * 1e6
            elif unit_lower == 'gbyte':
                return value * 1e9
            else:
                raise ValueError(f"Unknown memory unit: {unit}")
        
        # Time units: us, ms
        def convert_time_to_seconds(value, unit):
            """Convert time units to seconds."""
            unit_lower = unit.lower()
            if unit_lower == 'us':
                return value * 1e-6
            elif unit_lower == 'ms':
                return value * 1e-3
            else:
                raise ValueError(f"Unknown time unit: {unit}")
        
        # Convert dram bytes
        bytes_read = convert_bytes_to_bytes(
            metrics['dram__bytes_read.sum']['value'],
            metrics['dram__bytes_read.sum']['unit']
        )
        bytes_write = convert_bytes_to_bytes(
            metrics['dram__bytes_write.sum']['value'],
            metrics['dram__bytes_write.sum']['unit']
        )
        
        # Convert time duration
        time_seconds = convert_time_to_seconds(
            metrics['gpu__time_duration.sum']['value'],
            metrics['gpu__time_duration.sum']['unit']
        )
        
        # Calculate bandwidth: (bytes_read + bytes_write) / time
        total_bytes = bytes_read + bytes_write
        bandwidth_bytes_per_sec = total_bytes / time_seconds
        bandwidth_gb_per_sec = bandwidth_bytes_per_sec / 1e9
        
        results.append({
            'n': n,
            'num_elements': num_elements,
            'data_size_bytes': data_size_bytes,
            'bytes_read': bytes_read,
            'bytes_write': bytes_write,
            'time_seconds': time_seconds,
            'bandwidth_bytes_per_sec': bandwidth_bytes_per_sec,
            'bandwidth_gb_per_sec': bandwidth_gb_per_sec
        })
    
    return results

def plot_bandwidth(results, output_file='bandwidth_plot.png'):
    """
    Plot bandwidth vs data size with log2 scale on x-axis.
    """
    if not results:
        print("Error: No data to plot")
        return
    
    # Sort by data size
    results.sort(key=lambda x: x['data_size_bytes'])
    
    data_sizes = [r['data_size_bytes'] for r in results]
    bandwidths = [r['bandwidth_bytes_per_sec'] for r in results]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot with log2 scale on x-axis
    plt.plot(data_sizes, bandwidths, 'o-', linewidth=2, markersize=8)
    
    # Set x-axis to log2 scale
    plt.xscale('log', base=2)
    
    # Format x-axis labels
    ax = plt.gca()
    ax.set_xlabel('Data Size (bytes, log₂ scale)', fontsize=12)
    ax.set_ylabel('Bandwidth (bytes/s)', fontsize=12)
    ax.set_title('DRAM Bandwidth vs Data Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis ticks to show powers of 2
    max_size = max(data_sizes)
    min_size = min(data_sizes)
    min_power = int(np.floor(np.log2(min_size)))
    max_power = int(np.ceil(np.log2(max_size)))
    
    # Create tick positions (powers of 2)
    tick_positions = [2**p for p in range(min_power, max_power + 1)]
    # Format labels: show 2^n format
    tick_labels = [f'2^{p}' for p in range(min_power, max_power + 1)]
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
    
    # Format y-axis to show GB/s
    y_ticks = ax.get_yticks()
    y_labels = [f'{y/1e9:.2f} GB/s' for y in y_ticks]
    ax.set_yticklabels(y_labels)
    
    # Add value annotations
    for r in results:
        plt.annotate(
            f"{r['bandwidth_gb_per_sec']:.2f} GB/s",
            (r['data_size_bytes'], r['bandwidth_bytes_per_sec']),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9,
            alpha=0.7
        )
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Print summary
    print("\nSummary:")
    print("-" * 80)
    print(f"{'Data Size (bytes)':<20} {'Bandwidth (GB/s)':<20} {'Time (ms)':<15}")
    print("-" * 80)
    for r in results:
        print(f"{r['data_size_bytes']:<20} {r['bandwidth_gb_per_sec']:<20.2f} {r['time_seconds']*1000:<15.2f}")
    print("-" * 80)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_bandwidth.py <log_file> [output_image]")
        print("Example: python3 plot_bandwidth.py run_copy.log bandwidth.png")
        sys.exit(1)
    
    log_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'bandwidth_plot.png'
    
    print(f"Parsing log file: {log_file}")
    results = parse_log_file(log_file)
    
    if not results:
        print("Error: No valid data found in log file")
        sys.exit(1)
    
    print(f"Found {len(results)} valid test results")
    plot_bandwidth(results, output_file)

if __name__ == '__main__':
    main()

