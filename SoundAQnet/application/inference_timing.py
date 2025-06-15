import time
import torch
import numpy as np
from framework.processing import forward
from framework.models_pytorch import move_data_to_gpu
import framework.config as config

def measure_single_sample_inference(model, generator, cuda=True, num_samples=100):
    """
    Measure inference time for single samples during testing
    
    Args:
        model: The trained SoundAQnet model
        generator: DataGenerator_Mel_loudness_graph instance
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
                
            (batch_x, batch_x_loudness, batch_scene, batch_event, batch_graph, 
             batch_ISOPls, batch_ISOEvs, batch_pleasant, batch_eventful, 
             batch_chaotic, batch_vibrant, batch_uneventful, batch_calm, 
             batch_annoying, batch_monotonous) = batch_data
            
            # Process each sample in the batch individually
            batch_size = batch_x.shape[0]
            
            for i in range(batch_size):
                if sample_count >= num_samples:
                    break
                    
                # Extract single sample
                single_x = batch_x[i:i+1]  # Keep batch dimension
                single_x_loudness = batch_x_loudness[i:i+1]
                single_graph = [batch_graph[i]]
                
                # Measure preprocessing time (data movement to GPU)
                preprocess_start = time.time()
                single_x_gpu = move_data_to_gpu(single_x, cuda)
                single_x_loudness_gpu = move_data_to_gpu(single_x_loudness, cuda)
                preprocess_end = time.time()
                
                # Measure pure inference time
                torch.cuda.synchronize() if cuda else None  # Ensure GPU operations are complete
                inference_start = time.time()
                
                # Forward pass
                scene, event, ISOPls, ISOEvs, pleasant, eventful, chaotic, vibrant, \
                uneventful, calm, annoying, monotonous = model(single_x_gpu, single_x_loudness_gpu, single_graph)
                
                torch.cuda.synchronize() if cuda else None  # Ensure GPU operations are complete
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
    print("SINGLE SAMPLE INFERENCE TIMING STATISTICS")
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

def warm_up_model(model, generator, cuda=True, warmup_samples=10):
    """
    Warm up the model and GPU to get stable timing measurements
    """
    print("Warming up model...")
    model.eval()
    
    generate_func = generator.generate_testing(data_type='testing')
    
    with torch.no_grad():
        for batch_data in generate_func:
            (batch_x, batch_x_loudness, batch_scene, batch_event, batch_graph, 
             *_) = batch_data
            
            batch_size = batch_x.shape[0]
            
            for i in range(min(warmup_samples, batch_size)):
                single_x = batch_x[i:i+1]
                single_x_loudness = batch_x_loudness[i:i+1]
                single_graph = [batch_graph[i]]
                
                single_x_gpu = move_data_to_gpu(single_x, cuda)
                single_x_loudness_gpu = move_data_to_gpu(single_x_loudness, cuda)
                
                # Warmup forward pass
                _ = model(single_x_gpu, single_x_loudness_gpu, single_graph)
                
                if i >= warmup_samples - 1:
                    break
            break
    
    if cuda:
        torch.cuda.synchronize()
    
    print("Warmup completed.")

# Modified main function for your Inference.py
def main_with_timing(argv):
    node_emb_dim = 64
    hidden_dim, out_dim = 32, 64
    batch_size = 32
    number_of_nodes = 8
    
    using_model = SoundAQnet
    
    model = using_model(
        max_node_num=number_of_nodes,
        node_emb_dim=node_emb_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim)
    
    # Load your trained model
    syspath = os.path.join(os.getcwd(), 'system', 'model')
    file = 'SoundAQnet_ASC96_AEC95_PAQ1052.pth'  # Your model file
    event_model_path = os.path.join(syspath, file)
    
    model_event = torch.load(event_model_path, map_location='cpu')
    
    if 'state_dict' in model_event.keys():
        model.load_state_dict(model_event['state_dict'])
    else:
        model.load_state_dict(model_event)
    
    if config.cuda:
        model.cuda()
    
    Dataset_path = os.path.join(os.getcwd(), 'Dataset')
    generator = DataGenerator_Mel_loudness_graph(Dataset_path, node_emb_dim, number_of_nodes)
    
    # Warm up the model for stable timing
    warm_up_model(model, generator, cuda=config.cuda)
    
    # Measure inference timing
    print("\nMeasuring inference timing...")
    timing_stats = measure_single_sample_inference(
        model, generator, cuda=config.cuda, num_samples=100
    )
    
    # Print results
    print_timing_stats(timing_stats)
    
    return timing_stats

if __name__ == "__main__":
    import sys
    try:
        sys.exit(main_with_timing(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)