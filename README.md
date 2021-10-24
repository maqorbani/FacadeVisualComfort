# FacadeVisualComfort
## Annual synthetic Radiance HDR image renderings and analysis

**This repository includes:**

* Python script to run Radiance software on a Linux-based machine to simulate annual (or a subset of) luminance-based images in a given Radiance scene and settings.
* Python script to do a k-means clustering to find the required renderings based on the target's climate conditions, based on an enhanced version of [this paper](https://arxiv.org/abs/2009.09928).
* Python script to create tensor-based data sets out of the simulated images.
* Python script for deep learning to predict annual luminance-based images (the unstimulated images in the annual data set).
* Python script to analyze and evaluate the resulting images on the desirable parameters based on the luminance distribution of the images.
* Python script to plot and save the results.

Please note that this repo is going to be updated, and a unified program to do the mentioned tasks will be published. So, stay tuned!

_Climate Data_ folder includes the lighting and sky conditions of Tehran's climate.

_K-meansClustering_ folder includes the k-means clustering-related script.

_RadianceRendering_ folder includes the scripts for Radiance renderings.

_analysis_ folder includes the scripts for the image analysis.

# Procedure

## Rendering

Rendering starts with designing the required 3D file in a 3D CAD software like Rhinoceros.

The sky descriptions of the climate are extracted from the EPW climate file of the city. Additionally, the climate's sky conditions data is also necessary for the following steps.

![image](https://user-images.githubusercontent.com/47574645/138586916-1216f283-8569-47a6-aa37-ac7aad27480b.png)

The rendering scripts and files are implemented on a Linux-based VPS, and the rendering process is initiated concurrently on each of the VPS's CPU cores. The rendering period could be annually or be a subset of the year. Moreover, using the K-means clustering method, annual representative hours could be selected to render and predict the remaining images using the deep learning model.

![HDRs](https://user-images.githubusercontent.com/47574645/138588301-eca3a80b-8a4e-4d86-a4a8-ae2a046530fb.gif)

After each image is rendered, the script automatically extracts them into a float32 NumPy array, ready for the deep learning step.

## Analysis

The analysis script analyses the images based on the required criteria, e.g., max_luminance, average_luminance, DGP value, etc., and plots the annual heatmap for the desired extracted parameter. 


![Alt2_mean](https://user-images.githubusercontent.com/47574645/138587942-1934cf17-3e50-4097-ab12-3af41075cbf4.png)
