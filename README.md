# Before converting TF model of v2.9 for Jetson Nano
1. Install tensorflow v2.9 from source, refering to https://github.com/jkjung-avt/jetson_nano
2. Install tensorflow addons v0.19 before installing tensorflow object detection package for for machine learning, with reference to https://qengineering.eu/install-tensorflow-addons-on-jetson-nano.html
4. There may be error during install addons as claim in https://codeyarns.com/tech/2017-12-22-nvcc-argument-redefinition-error.html#gsc.tab=0, in this case, the script line "nvccopts += std_options" in function "def InvokeNvc" should be comment out, the InvokeNvc function is in file "crosstool_wrapper_driver_is_not_gcc.tpl" of addons file folder "addons-0.19.0/build_deps/toolchains/gpu/crosstool/clang/bin/"
5. Install tensorflow-object-detection-api, ref. https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-object-detection-api-installation. The COCO is the depenence of tensorflow v2.X, if check it is installed, thus it would be no need to install COCO API for tensorflow v2.X. 
   > $ git clone https://github.com/tensorflow/models
