import time
import torch
import numpy as np
import os
import sys

# Add your framework path here
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

import framework.config as config
from framework.models_pytorch import AD_CNN

def move_data_to_gpu(x, cuda, using_float=False):
    if using_float:
        x = torch.Tensor(x)
    else:
        if 'float' in str(x.dtype):
            x = torch.Tensor(x)
        elif 'int' in str(x.dtype):
            x = torch.LongTensor(x)
        else:
            raise Exception("Error!")

    if cuda:
        x = x.cuda()
    return x

def measure_single_sample_inference_ad_cnn(model, generator, cuda=True, num_samples=100):
    """
    Measure inference time for single samples using AD_CNN model (mel spectrogram only)
    
    Args:
        model: The trained AD_CNN model
        generator: DataGenerator instance 
        cuda: Whether to use GPU
        num_samples: Number of samples to test for averaging
    
    Returns:
        dict: Timing statistics
    """
    model.eval()
    
    # Generate test data
    generate_func = generator.generate_testing(data_type='testing')
    
    inference_times = []
    preprocessing_times = []
    total_times = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_data in generate_func:
            if sample_count >= num_samples:
                break
            
            # Extract only mel spectrogram data (first element)
            batch_x = batch_data[0]  # Only mel spectrogram, no loudness
            
            # Process each sample in the batch individually
            batch_size = batch_x.shape[0]
            
            for i in range(batch_size):
                if sample_count >= num_samples:
                    break
                    
                # Extract single sample (only mel spectrogram)
                single_x = batch_x[i:i+1]  # Keep batch dimension
                
                # Measure preprocessing time (data movement to GPU)
                preprocess_start = time.time()
                single_x_gpu = move_data_to_gpu(single_x, cuda)
                preprocess_end = time.time()
                
                # Measure pure inference time
                if cuda:
                    torch.cuda.synchronize()  # Ensure GPU operations are complete
                    
                inference_start = time.time()
                
                # Forward pass - AD_CNN only takes mel spectrogram input
                scene, event, ISOPls, ISOEvs, pleasant, eventful, chaotic, vibrant, \
                uneventful, calm, annoying, monotonous = model(single_x_gpu)
                
                if cuda:
                    torch.cuda.synchronize()  # Ensure GPU operations are complete
                    
                inference_end = time.time()
                
                # Calculate times
                preprocess_time = preprocess_end - preprocess_start
                inference_time = inference_end - inference_start
                total_time = preprocess_time + inference_time
                
                preprocessing_times.append(preprocess_time)
                inference_times.append(inference_time)
                total_times.append(total_time)
                
                sample_count += 1
                
                if sample_count % 10 == 0:
                    print(f"Processed {sample_count}/{num_samples} samples")
    
    # Calculate statistics
    stats = {
        'num_samples': len(inference_times),
        'inference_time': {
            'mean_ms': np.mean(inference_times) * 1000,
            'std_ms': np.std(inference_times) * 1000,
            'min_ms': np.min(inference_times) * 1000,
            'max_ms': np.max(inference_times) * 1000,
            'median_ms': np.median(inference_times) * 1000
        },
        'preprocessing_time': {
            'mean_ms': np.mean(preprocessing_times) * 1000,
            'std_ms': np.std(preprocessing_times) * 1000,
            'min_ms': np.min(preprocessing_times) * 1000,
            'max_ms': np.max(preprocessing_times) * 1000,
            'median_ms': np.median(preprocessing_times) * 1000
        },
        'total_time': {
            'mean_ms': np.mean(total_times) * 1000,
            'std_ms': np.std(total_times) * 1000,
            'min_ms': np.min(total_times) * 1000,
            'max_ms': np.max(total_times) * 1000,
            'median_ms': np.median(total_times) * 1000
        }
    }
    
    return stats

def print_timing_stats(stats):
    """Print timing statistics in a readable format"""
    print("\n" + "="*60)
    print("AD_CNN SINGLE SAMPLE INFERENCE TIMING STATISTICS")
    print("="*60)
    print(f"Number of samples tested: {stats['num_samples']}")
    print()
    
    print("PURE INFERENCE TIME (model forward pass only):")
    print(f"  Mean:   {stats['inference_time']['mean_ms']:.3f} ± {stats['inference_time']['std_ms']:.3f} ms")
    print(f"  Median: {stats['inference_time']['median_ms']:.3f} ms")
    print(f"  Range:  {stats['inference_time']['min_ms']:.3f} - {stats['inference_time']['max_ms']:.3f} ms")
    print()
    
    print("PREPROCESSING TIME (data movement to GPU):")
    print(f"  Mean:   {stats['preprocessing_time']['mean_ms']:.3f} ± {stats['preprocessing_time']['std_ms']:.3f} ms")
    print(f"  Median: {stats['preprocessing_time']['median_ms']:.3f} ms")
    print(f"  Range:  {stats['preprocessing_time']['min_ms']:.3f} - {stats['preprocessing_time']['max_ms']:.3f} ms")
    print()
    
    print("TOTAL TIME (preprocessing + inference):")
    print(f"  Mean:   {stats['total_time']['mean_ms']:.3f} ± {stats['total_time']['std_ms']:.3f} ms")
    print(f"  Median: {stats['total_time']['median_ms']:.3f} ms")
    print(f"  Range:  {stats['total_time']['min_ms']:.3f} - {stats['total_time']['max_ms']:.3f} ms")
    print()
    
    # Calculate throughput
    throughput = 1000 / stats['inference_time']['mean_ms']  # samples per second
    print(f"THROUGHPUT: {throughput:.1f} samples/second")
    print("="*60)

def warm_up_model_ad_cnn(model, generator, cuda=True, warmup_samples=10):
    """
    Warm up the AD_CNN model and GPU to get stable timing measurements
    """
    print("Warming up AD_CNN model...")
    model.eval()
    
    generate_func = generator.generate_testing(data_type='testing')
    
    with torch.no_grad():
        for batch_data in generate_func:
            # Extract only mel spectrogram data
            batch_x = batch_data[0]  # Only mel spectrogram
            
            batch_size = batch_x.shape[0]
            
            for i in range(min(warmup_samples, batch_size)):
                single_x = batch_x[i:i+1]
                single_x_gpu = move_data_to_gpu(single_x, cuda)
                
                # Warmup forward pass - AD_CNN only takes mel spectrogram
                _ = model(single_x_gpu)
                
                if i >= warmup_samples - 1:
                    break
            break
    
    if cuda:
        torch.cuda.synchronize()
    
    print("Warmup completed.")

# Example usage function
def test_ad_cnn_timing():
    """
    Example function showing how to use the timing measurement with AD_CNN
    """
    # Import your AD_CNN model
    # from ad_cnn_file import AD_CNN  # Replace with your actual import
    
    # Initialize model
    model = AD_CNN()
    
    # Load trained weights if available
    # model_path = 'path/to/your/trained_model.pth'
    # model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    if config.cuda and torch.cuda.is_available():
        model.cuda()
    
    # Initialize your data generator (without loudness components)
    # Dataset_path = os.path.join(os.getcwd(), 'Dataset')
    # generator = YourDataGenerator(Dataset_path)  # Replace with actual generator
    
    # Warm up the model
    # warm_up_model_ad_cnn(model, generator, cuda=config.cuda)
    
    # Measure timing
    # print("\nMeasuring AD_CNN inference timing...")
    # timing_stats = measure_single_sample_inference_ad_cnn(
    #     model, generator, cuda=config.cuda, num_samples=100
    # )
    
    # Print results
    # print_timing_stats(timing_stats)
    
    print("Replace the commented sections with your actual model and data generator")
    return None

if __name__ == "__main__":
    test_ad_cnn_timing()