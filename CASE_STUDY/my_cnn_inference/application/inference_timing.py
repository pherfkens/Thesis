import time
import torch
import numpy as np
from framework.processing import forward
from framework.models_pytorch import move_data_to_gpu


def measure_inference_time(model, generator, cuda=True, num_warmup=5, num_iterations=None):
    """
    Measure pure inference time on test dataset
    
    Args:
        model: Trained model in eval mode
        generator: Test data generator
        cuda: Whether to use GPU
        num_warmup: Number of warmup iterations (GPU optimization)
        num_iterations: Number of iterations to measure (None = all data)
    
    Returns:
        dict with timing statistics
    """
    model.eval()
    
    # Warmup GPU
    print("🔥 Warming up GPU...")
    generate_func = generator.generate_testing(data_type='testing', max_iteration=num_warmup)
    for i, data in enumerate(generate_func):
        if i >= num_warmup:
            break
        
        # Quick forward pass for warmup
        batch_x, batch_event, *_ = data
        batch_x = move_data_to_gpu(batch_x, cuda)
        
        with torch.no_grad():
            _ = model(batch_x)
    
    # Clear GPU cache
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print("⏱️  Measuring inference time...")
    
    # Actual timing
    generate_func = generator.generate_testing(data_type='testing', max_iteration=num_iterations)
    
    batch_times = []
    sample_times = []
    total_samples = 0
    
    overall_start = time.time()
    
    for iteration, data in enumerate(generate_func):
        batch_x, batch_event, *_ = data
        batch_size = batch_x.shape[0]
        
        # Move to GPU
        batch_x = move_data_to_gpu(batch_x, cuda)
        
        # Time the forward pass
        if cuda:
            torch.cuda.synchronize()  # Ensure GPU operations complete
        
        batch_start = time.time()
        
        with torch.no_grad():
            outputs = model(batch_x)
        
        if cuda:
            torch.cuda.synchronize()  # Ensure GPU operations complete
        
        batch_end = time.time()
        
        batch_time = batch_end - batch_start
        batch_times.append(batch_time)
        
        # Calculate per-sample time
        sample_time = batch_time / batch_size
        sample_times.extend([sample_time] * batch_size)
        total_samples += batch_size
        
        if iteration % 10 == 0:
            print(f"  Processed batch {iteration+1}, samples: {total_samples}")
    
    overall_end = time.time()
    
    # Calculate statistics
    batch_times = np.array(batch_times)
    sample_times = np.array(sample_times)
    
    results = {
        'total_time_s': overall_end - overall_start,
        'total_samples': total_samples,
        'total_batches': len(batch_times),
        
        # Per-sample metrics (most important for reporting)
        'mean_sample_time_ms': np.mean(sample_times) * 1000,
        'std_sample_time_ms': np.std(sample_times) * 1000,
        'median_sample_time_ms': np.median(sample_times) * 1000,
        'min_sample_time_ms': np.min(sample_times) * 1000,
        'max_sample_time_ms': np.max(sample_times) * 1000,
        
        # Per-batch metrics
        'mean_batch_time_ms': np.mean(batch_times) * 1000,
        'std_batch_time_ms': np.std(batch_times) * 1000,
        
        # Throughput metrics
        'samples_per_second': total_samples / (overall_end - overall_start),
        'batches_per_second': len(batch_times) / (overall_end - overall_start),
    }
    
    return results


def measure_single_forward_pass_time(model, input_shape, cuda=True, num_runs=100):
    """
    Measure time for a single forward pass with synthetic data
    Useful for reporting model complexity independent of data loading
    
    Args:
        model: Trained model
        input_shape: Shape of input tensor (batch_size, seq_len, features)
        cuda: Whether to use GPU
        num_runs: Number of runs to average
    """
    model.eval()
    
    # Create synthetic input
    dummy_input = torch.randn(*input_shape)
    if cuda:
        dummy_input = dummy_input.cuda()
        model = model.cuda()
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if cuda:
        torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(num_runs):
        if cuda:
            torch.cuda.synchronize()
        
        start = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        
        if cuda:
            torch.cuda.synchronize()
        
        end = time.time()
        times.append(end - start)
    
    times = np.array(times)
    
    return {
        'mean_forward_time_ms': np.mean(times) * 1000,
        'std_forward_time_ms': np.std(times) * 1000,
        'median_forward_time_ms': np.median(times) * 1000,
        'min_forward_time_ms': np.min(times) * 1000,
        'max_forward_time_ms': np.max(times) * 1000,
    }


def comprehensive_timing_report(model, generator, input_shape, cuda=True):
    """
    Generate comprehensive timing report for research paper
    """
    print("📊 COMPREHENSIVE INFERENCE TIMING REPORT")
    print("=" * 50)
    
    # 1. Real dataset timing
    print("\n1️⃣ Real Dataset Inference Timing:")
    dataset_times = measure_inference_time(model, generator, cuda)
    
    print(f"   Total samples processed: {dataset_times['total_samples']}")
    print(f"   Mean inference time per sample: {dataset_times['mean_sample_time_ms']:.2f} ± {dataset_times['std_sample_time_ms']:.2f} ms")
    print(f"   Median inference time per sample: {dataset_times['median_sample_time_ms']:.2f} ms")
    print(f"   Throughput: {dataset_times['samples_per_second']:.1f} samples/second")
    
    # 2. Pure forward pass timing
    print("\n2️⃣ Pure Forward Pass Timing (synthetic data):")
    forward_times = measure_single_forward_pass_time(model, input_shape, cuda)
    
    print(f"   Input shape: {input_shape}")
    print(f"   Mean forward pass time: {forward_times['mean_forward_time_ms']:.2f} ± {forward_times['std_forward_time_ms']:.2f} ms")
    print(f"   Median forward pass time: {forward_times['median_forward_time_ms']:.2f} ms")
    
    # 3. Hardware info
    print("\n3️⃣ Hardware Information:")
    if cuda and torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("   Device: CPU")
    
    print(f"   PyTorch Version: {torch.__version__}")
    
    # 4. Model complexity
    from framework.pytorch_utils import count_parameters
    params = count_parameters(model)
    print(f"   Model Parameters: {params:,} ({params/1e6:.2f}M)")
    
    return {
        'dataset_timing': dataset_times,
        'forward_timing': forward_times,
        'model_params': params
    }


# Usage in your inference script
def add_timing_to_inference():
    """
    Add this to your Inference.py for proper timing measurement
    """
    # After loading model and before evaluation
    print("\n" + "="*60)
    print("INFERENCE TIMING MEASUREMENT")
    print("="*60)
    
    # Determine input shape from your data
    # For your models, this is typically (batch_size, seq_len, mel_bins)
    input_shape = (32, 3001, 64)  # Adjust based on your model
    
    # Run comprehensive timing
    timing_results = comprehensive_timing_report(
        model=model, 
        generator=generator, 
        input_shape=input_shape,
        cuda=config.cuda
    )
    
    # Save timing results
    import json
    with open('inference_timing_report.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            return obj
        
        json.dump(convert_numpy(timing_results), f, indent=2)
    
    print(f"\n💾 Detailed timing results saved to: inference_timing_report.json")
    
    return timing_results