<h1>Null-scatering path integral (NSPI)</h1>

<h2>Introduction</h2>
The goal of this project is to introduce the heterogeneous volumetric path tracing into the Yocto-GL Renderer. allowing accurate simulations of complex volumetric phenomena such as smoke and fog.<br />
In order to achieve this we firsty found a way to convert the OpenVDB volumetric files into a format compatible with Yocto-GL, then implemented the code to upload them using the JSON scenes and extract the data needed to render the volumetric materials.

<h2>Volumes handling</h2>
Once we extracted the data we defined a new struct <code>volume_data</code> to collected them and modified the already existing <code>material_point</code><br />
Since the volumetric object might be descripted by both density values and emission values for each voxel we implemented some function to evaluate these properties at a give point such as: <code>eval_density</code>, <code>eval_emission_nspi</code> and <code>eval_volume</code>

<h2>The algorithm</h2>
The last step was the implementation of the actual Null-scatering path integral formulation algorith, we started from one of the path tracer 
