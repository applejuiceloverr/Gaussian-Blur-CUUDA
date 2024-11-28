Gaussian Blur Implementation in CUDA
Overview

This project implements a Gaussian Blur effect on images using CUDA for parallel processing on a GPU. The goal is to efficiently apply the blur effect by computing a weighted average of pixel values in an image, using a Gaussian kernel.

At the core of the Gaussian blur is a Gaussian kernel, which is a square matrix derived from a Gaussian distribution. The kernel is used to calculate the new value of each pixel by considering its neighboring pixels and applying a weighted sum.

For edge pixels where neighbors are missing, we substitute the current pixel value. Additionally, the kernel parameters—size and standard deviation (sigma)—can be customized, enabling flexible blurring effects.
Features
1. Gaussian Blur Application

    floute(image_source, image_destination)
    Blurs the image specified by image_source and saves the result to image_destination.

2. Custom Gaussian Blur

    FlouteGaussienCustom(image_source, image_destination, largeur, sigma)
    Generates a Gaussian kernel with the specified largeur (size) and sigma (standard deviation). The kernel is then applied to the image for custom blurring.

3. Benchmarking

    Bench(image_source)
    Runs a benchmark by applying the blur effect on the specified image using multiple thread-block sizes. Measures the average execution time over 10 runs (excluding the first run) to analyze performance.

How It Works
Gaussian Kernel

A Gaussian kernel is generated based on the size (largeur) and standard deviation (sigma). The kernel values are calculated using the Gaussian formula, ensuring the sum of weights is normalized.
Blurring Process

For each pixel:

    Compute the weighted sum of the pixel and its neighbors using the Gaussian kernel.
    Normalize the sum to maintain brightness consistency.

Example Calculation:
For a pixel with value 13:
New Value=(1⋅1+2⋅4+3⋅6+4⋅4+5⋅1+6⋅4+… )256
New Value=256(1⋅1+2⋅4+3⋅6+4⋅4+5⋅1+6⋅4+…)​
Edge Handling

When a neighbor is missing (e.g., at the edges of the image), the current pixel value is used as a substitute to ensure the calculation proceeds smoothly.
