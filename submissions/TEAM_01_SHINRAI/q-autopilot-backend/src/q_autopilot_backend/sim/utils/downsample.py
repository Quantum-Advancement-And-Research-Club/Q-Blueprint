import numpy as np


def downsample_series(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple:
    """LTTB (Largest-Triangle-Three-Buckets) downsampling - matches frontend."""
    n = len(x)
    if n <= max_points:
        return x, y
    
    # Simple LTTB implementation
    bucket_size = n / (max_points - 2)
    sampled_x = [x[0]]
    sampled_y = [y[0]]
    
    for i in range(1, max_points - 1):
        # Define bucket range
        start = int((i - 1) * bucket_size) + 1
        end = min(int(i * bucket_size) + 1, n - 1)
        
        if start >= end:
            continue
        
        # Point from previous bucket
        prev_x = sampled_x[-1]
        prev_y = sampled_y[-1]
        
        # Next bucket's average point (for area calculation)
        next_start = int(i * bucket_size) + 1
        next_end = min(int((i + 1) * bucket_size) + 1, n)
        if next_start < next_end:
            next_x = np.mean(x[next_start:next_end])
            next_y = np.mean(y[next_start:next_end])
        else:
            next_x = x[-1]
            next_y = y[-1]
        
        # Find point in current bucket that maximizes triangle area
        max_area = -1
        max_idx = start
        for j in range(start, end):
            area = abs((prev_x - next_x) * (y[j] - prev_y) - (prev_x - x[j]) * (next_y - prev_y))
            if area > max_area:
                max_area = area
                max_idx = j
        
        sampled_x.append(x[max_idx])
        sampled_y.append(y[max_idx])
    
    # Add last point
    sampled_x.append(x[-1])
    sampled_y.append(y[-1])
    
    return np.array(sampled_x), np.array(sampled_y)