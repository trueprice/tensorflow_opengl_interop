This codebase demonstrates how to pass textures to/from TensorFlow networks,
avoiding CPU<->GPU data transfers. You'll also need to change some hard-coded
paths in the simple\_blend\_network.cc file.

1. Prerequisites (you may already have these fulfilled)
  * See: https://www.tensorflow.org/install/install_sources
  * You’ll need a machine running the display on a TensorFlow-capable GPU.
    Currently, I've only been able to get everything to work on a single-GPU
    configuration.
  a. `sudo apt update`
  b. `sudo apt install autoconf build-essential automake libtool curl make g++ unzip python3-numpy swig python3-dev python3-pip python3-wheel pkg-config zip zlib1g-dev wget libcupti-dev`
  c. Check that the TensorFlow GPU pre-reqs are all installed (see link above)
  d. Install bazel following https://docs.bazel.build/versions/master/install-ubuntu.html

2. Install tensorflow-cmake **(NOTE STEP C)**
  a. git clone https://github.com/cjweeks/tensorflow-cmake
  b. cd tensorflow-cmake
  c. Modify build.sh:108 to `bazel build --config=monolithic tensorflow:libtensorflow_all.so` and comment out lines 54-58
  d. Modify eigen.sh:52 to `ARCHIVE_HEADER="tf_http_archive\(\s*`
  e. Modify protobuf.sh:56 to `HTTP_HEADER="tf_http_archive\(\s"`
  f. `mkdir build ../local # or wherever you want to build/install this version`
  g. `bash build.sh ./build ../local`

3. Set up the project
  * The Eigen and Protobuf versions apparently need to compatible with the
    installed version of TensorFlow, so you can use the following
    tensorflow-cmake commands to set your cmake files.
  a. `./eigen.sh generate installed build/tensorflow-github /path/to/cmake_modules/ /path/to/local/install/folder`
  b. `./protobuf.sh generate installed build/tensorflow-github /path/to/cmake_modules/ /path/to/local/install/folder`
  c. Things to set in your cmake config:
    * `TensorFlow_INCLUDE_DIR = /path/to/local/include/google/tensorflow`
    * `TensorFlow_LIBRARY = /path/to/local/lib/libtensorflow_all.so`
    * `Eigen_INCLUDE_DIR = /path/to/local/include/eigen3`
    * `Protobuf_INCLUDE_DIR: /path/to/local/include`

Network implementation
  * TODO: Add an example of creating a binary protobuf file from existing
    TensorFlow Saver output. This is similar to the freeze_graph.py script that
    ships with TensorFlow, but I found it easier to write my own, given the
    changes that need to be made.
  * Concerning changes to the network structure, I ended up (in Python)
    redefining the existing network inputs as TensorFlow placeholders (HxWx3).
    In the C++ code, these placeholders get changed into custom TextureInput
    ops. This way, you don't have to worry about the custom ops being defined in
    Python.
  * The C++ code introduces a CopyToTexture op that copies whatever network
    output you specify to a texture.
  * Finally, this code includes a new version of the ResizeBilinear Op that I
    hoped would be faster, but it doesn't appear to be. You can just comment out 
    these lines in simple\_blend\_network.cc. (TODO: Show how to replace the
    tf.image.resize\_bilinear op in Python with transpose, tile, and
    un-transpose.)
