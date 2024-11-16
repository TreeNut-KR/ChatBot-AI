import torch

def check_cuda_devices():
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_devices}")
    
    for i in range(num_devices):
        device_name = torch.cuda.get_device_name(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
        print(f"Device {i}: {device_name} (Total Memory: {total_memory:.2f} GB)")

if __name__ == "__main__":
    check_cuda_devices()
