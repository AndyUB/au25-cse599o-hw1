import torch

def test_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available. Using device: {device}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    tensor = torch.arange(100000, device=device)
    print(f"Tensor on device: {tensor.device}")

if __name__ == "__main__":
    test_gpu()