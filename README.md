
# Vision Transformer for video interolation 

This project implements state-of-the-art model Vision Transformer for video frame interpolation to increase its FPS , and compares its performance with traditional approaches like Deep Voxel flow , Super slomo model.

## Table of Contents

- [Introduction](#introduction)
- [Demo](#demo)
- [Results](#results)




- [License](#license)

## Introduction
  Here I implemented VIFT and Super Slomo model as published in these paper, paper respectively .
- Deep Voxel Flow
    - uses Optical flow & CNN approach
    - unable to handle complex motions
- Super Slomo
    - It replaces Optical flow by flow interpretation Unet like architecture .
    - computationally expensive
- Video Frame Interpolation Transformer
    - It uses Swin Transformer blocks ( Shifted Window transformer ) to reduce time complexity from quadratic to linear .
    - much smaller compared to Super Slomo , while still achieving better performance .
     

## Demo

<div style="display: flex; align-items: center;" >
  <img src='https://github.com/nagarajRPoojari/video-interpolation-AI/assets/116948655/1b31f647-ece1-4400-98dc-03305d0e35d3'>
</div>


## Results

| model / metric  | Parameters (M) |  PSNR ( peek-signal-to-noise-ration ) | SSMI ( structural similarity index ) |
|-----------------|----------------|---------------------------------------|--------------------------------------|
| Deep voxel flow |       -        |                27.6                   |                  0.92                |
| Super Slomo     |       38       |                31.4                   |                  0.94                |
| VIFT            |       7        |                35.1                   |                  0.96                |

## License
This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.


