# Training cost (View 2, five runs per method)

Times are wall-clock training time summed over epochs (held-out scoring excluded), as recorded by each run. **They are not directly comparable across methods:** runs used different machines (Siamese: Apple M1, 8 GB, macOS 15.5; Triplet, Center and SphereFace: Apple Silicon Macs on macOS 26.x, whose chip model was not recorded; ArcFace: a CUDA Tesla T4), different PyTorch versions, and different amounts of work per epoch (4,800 pairs = 9,600 images for Siamese; 300 × 36 images for Triplet; one pass over ~7,000–7,800 images in batches of 64 for Softmax / Center / SphereFace; 300 × 32 images for ArcFace). MPS memory is the largest of per-batch samples of driver-allocated memory (includes allocator cache, not a true peak); CUDA memory is PyTorch's peak allocated tensor memory. The two are different quantities.

| method | accelerator | platform | pytorch | epochs_per_run | total_training_hours | mean_hours_per_run | mean_minutes_per_epoch | peak_memory_mib | memory_measurement |
|---|---|---|---|---|---|---|---|---|---|
| Softmax baseline | Apple Silicon GPU (MPS) | macOS-26.3.1-arm64-arm-64bit | 2.13.0 | 30 | 6.57 | 1.31 | 2.63 | 5422 | MPS driver memory, max of per-batch samples (not a true peak) |
| Softmax + Center Loss | Apple Silicon GPU (MPS) | macOS-26.3.1-arm64-arm-64bit | 2.13.0 | 30 | 6.6 | 1.32 | 2.64 | 5422 | MPS driver memory, max of per-batch samples (not a true peak) |
| Siamese (restricted) | Apple Silicon GPU (MPS) | macOS-15.5-arm64-arm-64bit-Mach-O | 2.14.0 | 30 | 25.34 | 5.07 | 10.14 | 3420 | MPS driver memory, max of per-batch samples (not a true peak) |
| Siamese (unrestricted) | Apple Silicon GPU (MPS) | macOS-15.5-arm64-arm-64bit-Mach-O | 2.14.0 | 30 | 24.59 | 4.92 | 9.84 | 3420 | MPS driver memory, max of per-batch samples (not a true peak) |
| Triplet (semi-hard) | Apple Silicon GPU (MPS) | macOS-26.6.2-arm64-arm-64bit-Mach-O | 2.14.0 | 30 | 9.96 | 1.99 | 3.99 | 5448 | MPS driver memory, max of per-batch samples (not a true peak) |
| Triplet (batch-hard) | Apple Silicon GPU (MPS) | macOS-26.6.2-arm64-arm-64bit-Mach-O | 2.14.0 | 30 | 9.51 | 1.9 | 3.81 | 5448 | MPS driver memory, max of per-batch samples (not a true peak) |
| SphereFace | Apple Silicon GPU (MPS) | macOS-26.3.1-arm64-arm-64bit | 2.13.0 | 30 | 6.49 | 1.3 | 2.59 | 5454 | MPS driver memory, max of per-batch samples (not a true peak) |
| ArcFace | Tesla T4 (CUDA) | Linux-6.18.55+-x86_64-with-glibc2.39 | 2.11.0+cu128 | 28 | 4.2 | 0.84 | 1.8 | 2052 | CUDA max_memory_allocated (true peak of tensors) |
