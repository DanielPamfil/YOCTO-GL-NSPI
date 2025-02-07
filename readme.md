<h1>Null-scatering path integral (NSPI)</h1>

<h2>Introduction</h2>
The goal of this project is to introduce the heterogeneous volumetric path tracing into the [Yocto/GL](https://xelatihy.github.io/yocto-gl/) Renderer. allowing accurate simulations of complex volumetric phenomena such as smoke and fog.<br />
In order to achieve this we firsty found a way to convert the OpenVDB volumetric files into a format compatible with Yocto-GL, then implemented the code to upload them using the JSON scenes and extract the data needed to render the volumetric materials.

<h2>Volumes handling</h2>
Once we extracted the data we defined a new struct <code>volume_data</code> to collected them and modified the already existing <code>material_point</code><br />
Since the volumetric object might be descripted by both density values and emission values for each voxel we implemented some function to evaluate these properties at a give point such as: <code>eval_density</code>, <code>eval_emission_nspi</code> and <code>eval_volume</code>

<h2>The algorithm</h2>
The last step was the implementation of the actual null-scatering path integral formulation algorith, we started from one of the path tracers that already implemented the multi importance sampling (MIS) and added the shading of heterogeneus volumes. <br />
Since in null-scattering algorithms, we can see a heterogeneous volume as a mixture of real and fictitious particles, the main step is keeping track of which type of collision occurs, and we do that using the <code>eval_medium_interaction</code> function, as you can see in the <code>trace_path_volume_mis</code> tracer.

<h2>Compilation</h2>
This library requires a C++17 compiler and is know to compiled on OsX (Xcode >= 11), Windows (MSVC >= 2019) and Linux (gcc >= 9, clang >= 9). <br />

You can build the example applications using CMake with <code> mkdir build; cd build; cmake ..; cmake --build . </code> <br />

Yocto/GL required dependencies are included in the distribution and do not need to be installed separately.  <br />


<h2>Run</H2>
After the buildng the program can be runned by command line in the following way: <br />
<code> ./bin/Debug/ytrace --scene tests\_version40\bunny_smoke\smoke.json --output out/lowres/smoke.jpg --samples 4096 --resolution 1280 --sampler volpath --interactive </code>
