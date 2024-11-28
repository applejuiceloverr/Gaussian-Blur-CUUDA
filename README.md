# Gaussian Blur Implementation in CUDA  

## Overview  

This project implements a **Gaussian Blur** effect on images using CUDA for GPU-based parallel processing. The Gaussian blur technique smoothens an image by calculating a weighted average of pixel values based on their neighbors, using a **Gaussian kernel**.  

### Features  

- **Blurring images using a Gaussian kernel**  
- **Customizable kernel size and sigma (standard deviation)**  
- **Edge handling for border pixels**  
- **Benchmarking with different CUDA thread-block sizes**  

---

## Project Structure  

1. **Gaussian Blur Functionality**  
   - **`floute(image_source, image_destination)`**:  
     Blurs the input image (`image_source`) using a default Gaussian kernel and saves the result to the specified output path (`image_destination`).  

2. **Custom Gaussian Blur**  
   - **`FlouteGaussienCustom(image_source, image_destination, largeur, sigma)`**:  
     Generates a custom Gaussian kernel with specified size (`largeur`) and sigma, then applies it to blur the input image.  

3. **Benchmarking**  
   - **`Bench(image_source)`**:  
     Benchmarks the Gaussian blur operation on the input image using different thread-block sizes. Computes the average execution time over 10 runs (excluding the first).  

---

## How It Works  

### Gaussian Kernel  
A **Gaussian kernel** is a square matrix calculated from the Gaussian distribution. The kernel values are weights applied to a pixel and its neighbors.  

For each pixel, the new value is calculated as:  
\[
\text{New Value} = \frac{\text{Weighted Sum of Pixel and Neighbors}}{\text{Sum of Kernel Weights}}
\]  

**Edge Handling**: For border pixels where neighbors are missing, the pixel itself is used as a substitute.  

### Example Calculation  
For a pixel with value `13`, using the kernel:  
\[
\text{New Value} = \frac{(1 \cdot 1 + 2 \cdot 4 + 3 \cdot 6 + 4 \cdot 4 + 5 \cdot 1 + 6 \cdot 4 + \dots)}{256}
\]  

---

## Usage  

1. **Blur Image with Default Kernel**  
   ```cpp
   floute("input_image.jpg", "output_image.jpg");
