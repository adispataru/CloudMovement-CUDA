# CloudMovement-CUDA
PhD project for detecting cloud movement using Optical Flow and forecasting advection using Boids Algorithm.
This is the variant written in C++.

## Dependencies 
The following dependencies need to be installed in order to run the software. To ease things, we recommend using the conda environment manager. 
<pre><code>
conda env create cuda-dev
conda create --name cuda-dev python=3.6
conda activate cuda-dev
conda install cudatoolkit-dev
conda install gxx_linux-ppc64le
conda install -c imb-ai libopencv
or
conda install -c ibmdl/export/pub/software/server/ibm-ai/conda libopencv
conda install -c https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/linux-ppc64le libopencv
conda install cmake
</code></pre>
