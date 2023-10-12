# STI-Pose
## Introduction
  The code implementation for STI-Pose.
  We provide the code with a script `test.py` for test.
## Project Structure
  `PSO` is the module of the OPSO, which is adapted from the code of `sko` library.
  `render` is the module of the silhouette renderer, whici is adapted from the code of `vispy`. We achieve the slihouette rendering by writing shaders.
  `SilhouettePE` is the kernel algorithm implementation.
## Requirements
  Execute `pip install -r requirements.txt` to complete the environment configuration.
## Visualization
<div style="display:flex">
  <img src="ref_silhouette.png" alt="ref_silhouette.png" width="50%" height="50%">
  <img src="result_silhouette.png" alt="result_silhouette.png" width="50%" height="50%">
</div>
  
