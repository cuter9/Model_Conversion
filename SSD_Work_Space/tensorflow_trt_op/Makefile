CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python3
# DIR_SOURCE = ~/Model_Conversion/tensorflow_trt_op/cc_plugin
DIR_SOURCE = ~/Model_Conversion/tensorflow_trt_op/cc
DIR_OPLIB = ~/Model_Conversion/tensorflow_trt_op/python3/ops/set

GRID_ANCHOR_TRT_SRCS = $(wildcard $(DIR_SOURCE)/grid_ancjor_trt/*.cc)
FLATTERN_CONCAT_TRT_SRCS = $(wildcard $(DIR_SOURCE)/flattern_concat_trt/*.cc)
CONCAT_TRT_SRCS = $(wildcard $(DIR_SOURCE)/concat_trt/*.cc) 
NMS_TRT_SRCS = $(wildcard $(DIR_SOURCE)/nms_trt/*.cc) 
SQUEEZE_TRT_SRCS = $(wildcard $(DIR_SOURCE)/squeeze_trt/*.cc) 

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

# CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
CFLAGS = ${TF_LFLAGS} -fPIC -O2 -std=c++14
LDFLAGS = ${TF_LFLAGS}
# LDFLAGS = -shared ${TF_LFLAGS}

GRID_ANCHOR_TRT_TARGET_LIB = $(DIR_OPLIB)/_grid_anchor_trt_ops.so
FLATTERN_CONCAT_TRT_TARGET_LIB = $(DIR_OPLIB)/_flattern_concat_trt_ops.so
CONCAT_TRT_TARGET_LIB = $(DIR_OPLIB)/_concat_trt_ops.so
NMS_TRT_TARGET_LIB = $(DIR_OPLIB)/_nms_trt_ops.so
SQUEEZE_TRT_TARGET_LIB = $(DIR_OPLIB)/_squeeze_trt_ops.so
TRT_LIB = $(DIR_OPLIB)/_trt_ops.so

all: grid_anchor_trt_op flattern_concat_trt_op concat_trt_op nms_trt_op squeeze_trt_op

trt_ops:  $(TRT_LIB)
$(TRT_LIB): $(GRID_ANCHOR_TRT_SRCS) $(FLATTERN_CONCAT_TRT_SRCS) $(CONCAT_TRT_SRCS) $(NMS_TRT_SRCS) $(SQUEEZE_TRT_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

# grid_anchor_trt op for CPU
# grid_anchor_trt_op: $(GRID_ANCHOR_TRT_TARGET_LIB)
# $(GRID_ANCHOR_TRT_TARGET_LIB): $(GRID_ANCHOR_TRT_SRCS)
#	$(CXX) $(CFLAGS) -o $@ $^ $(CFLAGS) ${LDFLAGS}

grid_anchor_trt_op: $(GRID_ANCHOR_TRT_SRCS)
	$(CXX) -shared $^ -o $@ $(CFLAGS) $(LDFLAGS)

# flattern_concat_trt op for CPU
flattern_concat_trt_op: $(FLATTERN_CONCAT_TRT_TARGET_LIB)
$(FLATTERN_CONCAT_TRT_TARGET_LIB): $(FLATTERN_CONCAT_TRT_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

# concat_trt_op for CPU
concat_trt_op: $(CONCAT_TRT_TARGET_LIB)
$(CONCAT_TRT_TARGET_LIB): $(CONCAT_TRT_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

nms_trt_op: $(NMS_TRT_TARGET_LIB)
$(NMS_TRT_TARGET_LIB): $(NMS_TRT_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

squeeze_trt_op: $(SQUEEZE_TRT_TARGET_LIB)
$(SQUEEZE_TRT_TARGET_LIB): $(SQUEEZE_TRT_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

# grid_anchor_trt_test: model_conversion/python/ops/grid_anchor_trt_ops_test.py model_conversion/python/ops/grid_anchor_trt_ops.py $(GRID_ANCHOR_TRT_TARGET_LIB)
#	$(PYTHON_BIN_PATH) model_conversion/python/ops/grid_anchor_trt_ops_test.py

# grid_anchor_trt_pip_pkg: $(GRID_ANCHOR_TRT_TARGET_LIB)
#	./build_pip_pkg.sh make artifacts


clean:
	rm -f $(GRID_ANCHOR_TRT_TARGET_LIB)
	rm -f $(FLATTERN_CONCAT_TRT_TARGET_LIB)
	rm -f $(CONCAT_TRT_TARGET_LIB)
	rm -f $(NMS_TRT_TARGET_LIB)
	rm -f $(SQUEEZE_TRT_TARGET_LIB)
	rm -f $(TRT_LIB)

.PHONY: all clean