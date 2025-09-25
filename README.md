# HairReduct
Hair strand clustering and simplification framework.

![대표사진](results/rep.jpg)

</p>
HairReduct is a novel simplification framework that efficiently groups and reduces hair strands by leveraging position, direction, and curvature information. It dramatically decreases data size while preserving the flow and structural consistency of hair, enabling more efficient processing for graphics applications. With intuitive visualization and adjustable parameters, HairReduct offers a flexible solution for various hair graphics pipelines including simulation, rendering, and animation.

## 1. PipeLine
HairReduct operates in the following steps:

<p align="center">
  <img src="results/pipeline.jpg" width="90%" alt="Pipeline of HairReduct"/>
</p>

1. **Text File Input**  
   - Load hair strand data from a text file format.

2. **Preprocess & Load**  
   - Normalize coordinates and clean unnecessary values before loading into memory.

3. **Short Strand Filtering**  
   - Remove strands that are shorter than a predefined threshold.

4. **1st Grouping (Root-based)**  
   - Classify strands into Front/Back/Left/Right groups based on root position.

5. **2nd Grouping (Tail-based Threshold)**  
   - Further subdivide groups using direction and shape features from the tail section.

6. **Visualization (Full / Rep Only)**  
   - Render either all hair strands or only the representative lines.

## 2. Key Features
- **Preprocessing of raw hair strand data**  
  Normalize positions and segment strands for consistent analysis.

- **Strand grouping into clusters**  
  Classify strands into Front/Back/Left/Right groups with additional sub-grouping.

- **Representative strand extraction**  
  Select medoid strands from each group based on position, direction, and shape similarity.

- **Visualization modes**  
  Switch between Full mode (all strands) and Representative-only mode.

- **Adjustable parameters**  
  Control clustering and visualization with parameters such as `kPerGroup`, `tailRatio`, `S`, and weights (`w_pos`, `w_dir`, `w_shape`), as well as merge rules.

## 3. Requirements
- Windows 10/11, Visual Studio 2022 (C++17)
- OpenGL/GLUT (glew, glut)
- OpenCV (core, highgui, imgproc)

## 4. Build
1. Open `StrandAnalyzer.sln` in Visual Studio
2. Set Configuration = Release, Platform = x86
3. Build solution 

## 5. Run
StrandAnalyzer.exe data\sample_strands.txt

## 6. Data
- Sample data is **not included** in this repository due to file size limits.  
- Please download it from Google Drive and place it into the `data/` folder:  
  [Google Drive Link](https://drive.google.com/drive/folders/1Hmtub4w612y4uSeobGEtwiMfrqKNfr3a?usp=drive_link)

## Video Example
[Watch on YouTube](https://youtu.be/LARh_Wxccl0)
