#!/bin/bash

version=2.9.3
set -e

if [[ ! $(head -1 /etc/nv_tegra_release) =~ R32.*7\.[34] ]] ; then
# if [[ ! $(head -1 /etc/nv_tegra_release) =~ R32.*4 ]] ; then
  echo "ERROR: not JetPack-4.6"
  exit 1
else
  echo "NV Tegra Release : $(head -1 /etc/nv_tegra_release) \n"
fi

case $(cat /sys/module/tegra_fuse/parameters/tegra_chip_id) in
  "33" )  # Nano and TX1
    cuda_compute=5.3
    ;;
  "24" )  # TX2
    cuda_compute=6.2
    ;;
  "25" )  # Xavier NX and AGX Xavier
    cuda_compute=7.2
    ;;
  * )     # default
    cuda_compute=5.3,6.2,7.2
    ;;
esac

# script_path=$(realpath $0)
# patch_path=$(dirname $script_path)/tensorflow/tensorflow-2.3.0.patch
trt_version=$(echo /usr/lib/aarch64-linux-gnu/libnvinfer.so.? | cut -d '.' -f 3)

src_folder=${HOME}/src
mkdir -p $src_folder

if pip3 list | grep tensorflow > /dev/null; then
  echo "ERROR: tensorflow is installed already"
  exit 1
fi

if ! which bazel > /dev/null; then
  echo "ERROR: bazel has not been installled"
  exit 1
fi

echo "** Install requirements"
sudo apt-get install -y libhdf5-serial-dev hdf5-tools
# sudo pip3 install packaging Cython
# sudo pip3 install -U pip 'numpy<=1.19.0'
sudo pip3 install -U "numpy<1.24.0"
sudo pip3 install -U pip six wheel packaging requests opt_einsum setuptools mock 'future>=0.17.1' 'gast==0.3.3' typing_extensions h5py
sudo pip3 install -U keras_applications --no-deps
sudo pip3 install -U keras_preprocessing --no-deps

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

echo "** Download and patch tensorflow-${version}"
pushd $src_folder

DIR_TF_SRC=tensorflow-${version}

if [ ! -d "./${DIR_TF_SRC}" ]; then
  echo "** downloading ${DIR_TF_SRC}"
  wget https://github.com/tensorflow/tensorflow/archive/refs/tags/v${version}.tar.gz -O ${DIR_TF_SRC}.tar.gz
  # git clone https://github.com/tensorflow/tensorflow.git ./tensorflow-${version}
  tar xzvf ${DIR_TF_SRC}.tar.gz
fi

# cd tensorflow-${version}
pushd ${DIR_TF_SRC}

# patch -N -p1 < $patch_path && echo "tensorflow-2.7.0 source tree appears to be patched already.  Continue..."

echo "** Configure and build tensorflow-${version}"
export TMP=/tmp \
CC=/usr/bin/clang \
BAZEL_COMPILER=/usr/bin/clang \
PYTHON_BIN_PATH=$(which python3) \
PYTHON_LIB_PATH=$(python3 -c 'import site; print(site.getsitepackages()[0])') \
TF_CUDA_COMPUTE_CAPABILITIES=${cuda_compute} \
TF_CUDA_VERSION=10.2 \
TF_CUDA_CLANG=0 \
TF_CUDNN_VERSION=8 \
TF_TENSORRT_VERSION=${trt_version} \
CUDA_TOOLKIT_PATH=/usr/local/cuda \
CUDNN_INSTALL_PATH=/usr/lib/aarch64-linux-gnu \
TENSORRT_INSTALL_PATH=/usr/lib/aarch64-linux-gnu \
TF_NEED_IGNITE=0 \
TF_ENABLE_XLA=0 \
TF_NEED_OPENCL_SYCL=0 \
TF_NEED_COMPUTECPP=0 \
TF_NEED_ROCM=0 \
TF_NEED_CUDA=1 \
TF_NEED_TENSORRT=1 \
TF_NEED_OPENCL=0 \
TF_NEED_MPI=0 \
GCC_HOST_COMPILER_PATH=$(which gcc) \
TF_SET_ANDROID_WORKSPACE=0
# CC_OPT_FLAGS="-.*XNNPACK/.*@-mcpu=native" \
# CC_OPT_FLAGS="-march=native" \



# printenv | grep CC
./configure

# https://github.com/google/XNNPACK/issues/5566, using --per_file_copt='-.*XNNPACK/.*@-mcpu=native' instead of --copt='-mcpu=native'
bazel build --config=opt \
            --config=cuda \
            --config=noaws \
            --per_file_copt='-.*XNNPACK/.*@-mcpu=native' \
            --local_cpu_resources=HOST_CPUS*0.25 \
            --local_ram_resources=HOST_RAM*0.67 \
            --verbose_failures \
            //tensorflow/tools/pip_package:build_pip_package
            
bazel-bin/tensorflow/tools/pip_package/build_pip_package wheel/tensorflow_pkg

popd
echo "** Install tensorflow-${version}"
#sudo pip3 install wheel/tensorflow_pkg/tensorflow-2.3.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install wheel/tensorflow_pkg/tensorflow-${version}-cp38-cp38m-linux_aarch64.whl

popd

TF_CPP_MIN_LOG_LEVEL=3 python3 -c "import tensorflow as tf; tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR); print('tensorflow version: %s' % tf.__version__); print('tensorflow.test.is_built_with_cuda(): %s' % tf.test.is_built_with_cuda()); print('tensorflow.test.is_gpu_available(): %s' % tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))"

echo "** Build and install tensorflow-${version} successfully"
