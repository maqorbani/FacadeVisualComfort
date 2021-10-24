# FacadeVisualComfort
## Annual synthetic Radiance HDR renderings and image analysis

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
