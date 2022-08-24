# CloudMovement-CUDA
CUDA implementation of cloud movement using Optical Flow and Boids Algorithm.

## Dependencies 
The following dependencies need to be installed in order to run the software. To ease things, we recommend using the conda environment manager. 
The following environment has been configured for a PowerPC architecture on `linux-ppc64le`. 
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

## Build
The software is built using `cmake`:

<pre><code>
mkdir build & cd build
cmake .. & make
</code></pre>

## Run

### Simulation program
`./cuda_movement wind-folder mask-folder output-folder [num-particles steps]`

 - `wind-folder` is the folder containing the wind maps (if generated by this software they will be in a folder called `flows`)
 - `mask-folder` is the folder containing the cloud masks (if generated by this software they will be in a folder called `masks`)
 - `output-folder` is the name of the output files
 - `num-particles` is the number of cloud particles to use in the simulation
 - `steps` is the number of steps to simulate. 1 step is equal to the time between two cloud masks. 
 
 #### Intermediary data generation functions
 The program main function can be changed to use the cloud masks and cloud flow functions by uncommenting their call in main:
 - `OpticalFlowService::computeFlowForImages(...)`
 - `createCloudMasks()`
 - `createCloudMaskFromFlow()`
