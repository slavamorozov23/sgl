import torch
import sys

def check_cuda():
    """
    Checks CUDA availability and provides detailed information.
    """
    print("--- CUDA Check ---")

    if torch.cuda.is_available():
        print("CUDA is available. GPU detected.")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")

        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices available: {device_count}")

        if device_count > 0:
            current_device_index = torch.cuda.current_device()
            current_device_name = torch.cuda.get_device_name(current_device_index)
            print(f"Current CUDA device index: {current_device_index}")
            print(f"Current CUDA device name: {current_device_name}")

            try:
                print("Attempting a simple tensor operation on GPU...")
                a = torch.randn(2, 3).cuda()
                b = torch.randn(2, 3).cuda()
                c = a + b
                print("Simple tensor operation successful.")
                print(f"Result tensor device: {c.device}")
            except Exception as e:
                print(f"Error during simple tensor operation on GPU: {e}")
                print("This might indicate an issue with the CUDA setup or driver.")

        else:
            print("No CUDA devices found despite torch.cuda.is_available() being True.")
            print("This is unexpected and might indicate a configuration issue.")

    else:
        print("CUDA is NOT available. PyTorch is likely using the CPU.")
        print("Possible reasons:")
        print("  - No NVIDIA GPU detected.")
        print("  - NVIDIA driver not installed or not configured correctly.")
        print("  - CUDA Toolkit not installed or not configured correctly.")
        print("  - PyTorch installed without CUDA support.")
        print("  - Running in an environment without GPU access (e.g., some virtual machines).")

    print("--- End of CUDA Check ---")

if __name__ == "__main__":
    check_cuda()