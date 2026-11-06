# Training time and memory

The table summarizes the five View 2 runs of each method. Training time is the sum of the recorded epoch times; held-out scoring is excluded.

The experiments ran on different machines. Siamese used an 8 GB Apple M1, Triplet used another Apple Silicon Mac, and Softmax, Center Loss and SphereFace used an Apple Silicon Mac M1 Pro with 16GB of RAM, ArcFace used a Tesla T4. PyTorch versions also differed. The timings describe these experiments and do not provide a comparison of loss-function cost on common hardware.

The amount of work per epoch differs too: 9,600 images for Siamese, 10,800 for Triplet, roughly 7,000–7,800 eligible images in batches of 64 for the shuffled classifiers, and 9,600 images for ArcFace.

MPS memory is the largest driver-memory reading sampled after a batch, including allocator cache. CUDA memory is the peak tensor allocation reported by PyTorch. These are different measurements, and the sampled MPS value can miss a peak between readings.

| Method | Accelerator | Platform | PyTorch | Epochs per run | Total training (hours) | Hours per run | Minutes per epoch | Memory (MiB) | Memory measurement |
|---|---|---|---|---|---|---|---|---|---|
| Softmax baseline | Apple Silicon GPU (MPS) | macOS-26.3.1-arm64-arm-64bit | 2.13.0 | 30 | 6.57 | 1.31 | 2.63 | 5422 | MPS driver memory, max of per-batch samples (not a true peak) |
| Softmax + Center Loss | Apple Silicon GPU (MPS) | macOS-26.3.1-arm64-arm-64bit | 2.13.0 | 30 | 6.6 | 1.32 | 2.64 | 5422 | MPS driver memory, max of per-batch samples (not a true peak) |
| Siamese (restricted) | Apple Silicon GPU (MPS) | macOS-15.5-arm64-arm-64bit-Mach-O | 2.14.0 | 30 | 25.34 | 5.07 | 10.14 | 3420 | MPS driver memory, max of per-batch samples (not a true peak) |
| Siamese (unrestricted) | Apple Silicon GPU (MPS) | macOS-15.5-arm64-arm-64bit-Mach-O | 2.14.0 | 30 | 24.59 | 4.92 | 9.84 | 3420 | MPS driver memory, max of per-batch samples (not a true peak) |
| Triplet (semi-hard) | Apple Silicon GPU (MPS) | macOS-26.6.2-arm64-arm-64bit-Mach-O | 2.14.0 | 30 | 9.96 | 1.99 | 3.99 | 5448 | MPS driver memory, max of per-batch samples (not a true peak) |
| Triplet (batch-hard) | Apple Silicon GPU (MPS) | macOS-26.6.2-arm64-arm-64bit-Mach-O | 2.14.0 | 30 | 9.51 | 1.9 | 3.81 | 5448 | MPS driver memory, max of per-batch samples (not a true peak) |
| SphereFace | Apple Silicon GPU (MPS) | macOS-26.3.1-arm64-arm-64bit | 2.13.0 | 30 | 6.49 | 1.3 | 2.59 | 5454 | MPS driver memory, max of per-batch samples (not a true peak) |
| ArcFace | Tesla T4 (CUDA) | Linux-6.18.55+-x86_64-with-glibc2.39 | 2.11.0+cu128 | 28 | 4.2 | 0.84 | 1.8 | 2052 | CUDA max_memory_allocated (true peak of tensors) |
