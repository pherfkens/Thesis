import time
import torch
import numpy as np
import os
import sys

# Add your framework path here
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

import framework.config as config

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

def measure_single_sample_inference_cnn_models(model, generator, cuda=True, num_samples=100, model_name="CNN_Model"):
    """
    Measure inference time for single samples using CNN models (mel spectrogram only)
    Works with: AD_CNN_decreased_conv_layers, AD_CNN_hop_length, AD_CNN_linear_layer, AD_CNN_harder_max_pooling
    
    Args:
        model: Any of the CNN model variants
        generator: DataGenerator instance 
        cuda: Whether to use GPU
        num_samples: Number of samples to test for averaging
        model_name: Name of the model for display purposes
    
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
                
                # Forward pass - All CNN models only take mel spectrogram input
                # Returns: event, pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous
                outputs = model(single_x_gpu)
                
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
        'model_name': model_name,
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
    print(f"{stats['model_name']} SINGLE SAMPLE INFERENCE TIMING")
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

def warm_up_model_cnn(model, generator, cuda=True, warmup_samples=10):
    """
    Warm up any CNN model and GPU to get stable timing measurements
    """
    print("Warming up CNN model...")
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
                
                # Warmup forward pass - All CNN models only take mel spectrogram
                _ = model(single_x_gpu)
                
                if i >= warmup_samples - 1:
                    break
            break
    
    if cuda:
        torch.cuda.synchronize()
    
    print("Warmup completed.")

def compare_all_models(generator, cuda=True, num_samples=100):
    """
    Compare timing across all model variants
    """
    # Import all models
    from framework.models_pytorch import (
        AD_CNN_decreased_conv_layers,
        AD_CNN_hop_length, 
        AD_CNN_linear_layer,
        AD_CNN_harder_max_pooling
    )
    
    models = [
        (AD_CNN_decreased_conv_layers(), "AD_CNN_decreased_conv_layers"),
        (AD_CNN_hop_length(), "AD_CNN_hop_length"),
        (AD_CNN_linear_layer(), "AD_CNN_linear_layer"),
        (AD_CNN_harder_max_pooling(), "AD_CNN_harder_max_pooling")
    ]
    
    all_stats = []
    
    for model, model_name in models:
        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print('='*60)
        
        if cuda and torch.cuda.is_available():
            model.cuda()
        
        # Warm up model
        warm_up_model_cnn(model, generator, cuda=cuda)
        
        # Measure timing
        stats = measure_single_sample_inference_cnn_models(
            model, generator, cuda=cuda, num_samples=num_samples, model_name=model_name
        )
        
        print_timing_stats(stats)
        all_stats.append(stats)
        
        # Clean up GPU memory
        if cuda:
            model.cpu()
            torch.cuda.empty_cache()
    
    # Summary comparison
    print("\n" + "="*80)
    print("TIMING COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model Name':<30} {'Mean (ms)':<12} {'Throughput (fps)':<15}")
    print("-" * 80)
    
    for stats in all_stats:
        mean_time = stats['inference_time']['mean_ms']
        throughput = 1000 / mean_time
        print(f"{stats['model_name']:<30} {mean_time:<12.3f} {throughput:<15.1f}")
    
    return all_stats

# Example usage function for single model
def test_single_model_timing(model_class, model_name, model_path=None):
    """
    Test timing for a single model
    
    Args:
        model_class: The model class (e.g., AD_CNN_decreased_conv_layers)
        model_name: Name for display
        model_path: Path to trained weights (optional)
    """
    # Initialize model
    model = model_class()
    
    # Load trained weights if available
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Loaded weights from {model_path}")
    
    if config.cuda and torch.cuda.is_available():
        model.cuda()
    
    # You need to provide your data generator here
    # Dataset_path = os.path.join(os.getcwd(), 'Dataset')
    # generator = YourDataGenerator(Dataset_path)  # Replace with actual generator
    
    # Warm up the model
    # warm_up_model_cnn(model, generator, cuda=config.cuda)
    
    # Measure timing
    # print(f"\nMeasuring {model_name} inference timing...")
    # timing_stats = measure_single_sample_inference_cnn_models(
    #     model, generator, cuda=config.cuda, num_samples=100, model_name=model_name
    # )
    
    # Print results
    # print_timing_stats(timing_stats)
    
    print("Replace the commented sections with your actual data generator")
    return None
    
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
    # from your_ad_cnn_file import AD_CNN  # Replace with your actual import
    
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
    # Example usage:
    
    # For testing a single model:
    # from framework.models_pytorch import AD_CNN_decreased_conv_layers
    # test_single_model_timing(AD_CNN_decreased_conv_layers, "AD_CNN_decreased_conv_layers")
    
    # For comparing all models:
    # Dataset_path = os.path.join(os.getcwd(), 'Dataset')
    # generator = YourDataGenerator(Dataset_path)  # Replace with actual generator
    # compare_all_models(generator, cuda=config.cuda, num_samples=100)
    
    print("Available functions:")
    print("1. test_single_model_timing() - Test one specific model")
    print("2. compare_all_models() - Compare all model variants")
    print("3. measure_single_sample_inference_cnn_models() - Measure timing for any CNN model")
    print("\nReplace the commented sections with your actual data generator and model paths")