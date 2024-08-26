# Before converting object detection model of tensorflow v2.9 for Jetson Nano
1. Install tensorflow v2.9 from source, referring to https://github.com/jkjung-avt/jetson_nano
2. Install tensorflow addons v0.19 before installing tensorflow object detection package for for machine learning, with reference to https://qengineering.eu/install-tensorflow-addons-on-jetson-nano.html
   > $ wget https://github.com/tensorflow/addons/archive/refs/tags/v0.19.0.tar.gz && tar -xvf addons-0.19.0.tar.gz
4. Important Note:
   a. There may be error during installing the addons as claimed in https://codeyarns.com/tech/2017-12-22-nvcc-argument-redefinition-error.html#gsc.tab=0, in this case, the script line "nvccopts += std_options" in the function "def InvokeNvc" should be comment out, the InvokeNvc function is in file "crosstool_wrapper_driver_is_not_gcc.tpl" of addons file folder "addons-0.19.0/build_deps/toolchains/gpu/crosstool/clang/bin/"
   b. There is also a bug in "crosstool_wrapper_driver_is_not_gcc.tpl". In the line 206 of the tpl file, there are only 2 arguments but 3 "capability" data are provided, one of which should be deleted.
6. Install tensorflow-object-detection-api, ref. https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-object-detection-api-installation. The COCO is the depenence of tensorflow v2.X, if check it is installed, thus it would be no need to install COCO API for tensorflow v2.X. 
   > $ git clone https://github.com/tensorflow/models
7. The installation of tensorflow-object-detection-api would change the dependence of tensorflow v2.9, thus it may need to reinstall tensorflow v2.9 built in step 1 to reestabilish the dependence.
# Dependence of model conversion
1. onnx-1.16.2
2. protobuf-3.20.3
