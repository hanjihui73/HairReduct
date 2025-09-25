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

