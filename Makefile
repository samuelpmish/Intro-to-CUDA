NVCC ?= nvcc
CXX ?= c++
BUILD_DIR ?= build/

INCLUDES := -Icommon
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo
CXXFLAGS ?= -O3 -std=c++17

CUDA_SOURCES := \
  $(wildcard 01_kernel_basics/*.cu) \
  $(wildcard 02_memory/*.cu) \
  $(wildcard 03_parallel_reduction/*.cu) \
  $(wildcard 04_streams/*.cu) \
  $(wildcard 05_bugs/*.cu) \
  $(wildcard 06_profiling/unstructured_mesh/*.cu) \
  $(wildcard 06_profiling/bytes_in_flight/*.cu) \
  $(wildcard 06_profiling/box_blur/*.cu) \
  $(wildcard 06_profiling/instruction_latencies/*.cu) \
  $(wildcard 06_profiling/load_balancing/*.cu) \
  $(wildcard 06_profiling/csr_matrix/*.cu)

CXX_SOURCES := $(wildcard 01_kernel_basics/*.cpp)
SINGLE_TARGETS := $(basename $(notdir $(CUDA_SOURCES) $(CXX_SOURCES)))
MULTI_TARGETS := poisson_0 poisson_1 poisson_2 poisson_3 poisson_4 poisson_5 laplace_tuning
TARGETS := $(SINGLE_TARGETS) $(MULTI_TARGETS)
BINARIES := $(addprefix $(BUILD_DIR)/,$(TARGETS))

.DEFAULT_GOAL := all
.PHONY: all clean list $(TARGETS)

all: $(BINARIES)

# Each executable can also be built by name, for example: make hello_world
$(TARGETS): %: $(BUILD_DIR)/%

$(BUILD_DIR):
	mkdir -p $@

NVCCFLAGS_host_device_annotations := --expt-relaxed-constexpr --extended-lambda
NVCCFLAGS_instruction_latencies := --extended-lambda
NVCCFLAGS_function_approximation := --extended-lambda

define CUDA_EXECUTABLE
$(BUILD_DIR)/$(basename $(notdir $(1))): $(1) | $(BUILD_DIR)
	$$(NVCC) $$(CPPFLAGS) $$(INCLUDES) $$(NVCCFLAGS) $$(NVCCFLAGS_$(basename $(notdir $(1)))) $$< $$(LDFLAGS) -o $$@ $$(LDLIBS)
endef

define CXX_EXECUTABLE
$(BUILD_DIR)/$(basename $(notdir $(1))): $(1) | $(BUILD_DIR)
	$$(CXX) $$(CPPFLAGS) $$(INCLUDES) $$(CXXFLAGS) $$< $$(LDFLAGS) -o $$@ $$(LDLIBS)
endef

$(foreach source,$(CUDA_SOURCES),$(eval $(call CUDA_EXECUTABLE,$(source))))
$(foreach source,$(CXX_SOURCES),$(eval $(call CXX_EXECUTABLE,$(source))))

define MULTI_SOURCE_CUDA_EXECUTABLE
$(BUILD_DIR)/$(1): $(2) | $(BUILD_DIR)
	$$(NVCC) $$(CPPFLAGS) $$(INCLUDES) $$(NVCCFLAGS) $$^ $$(LDFLAGS) -o $$@ $$(LDLIBS)
endef

$(eval $(call MULTI_SOURCE_CUDA_EXECUTABLE,poisson_0,06_profiling/poisson_0/vector.cu 06_profiling/poisson_0/poisson.cu))
$(eval $(call MULTI_SOURCE_CUDA_EXECUTABLE,poisson_1,06_profiling/poisson_1/vector.cu 06_profiling/poisson_1/poisson.cu))
$(eval $(call MULTI_SOURCE_CUDA_EXECUTABLE,poisson_2,06_profiling/poisson_2/vector.cu 06_profiling/poisson_2/poisson.cu))
$(eval $(call MULTI_SOURCE_CUDA_EXECUTABLE,poisson_3,06_profiling/poisson_3/vector.cu 06_profiling/poisson_3/poisson.cu))
$(eval $(call MULTI_SOURCE_CUDA_EXECUTABLE,poisson_4,06_profiling/poisson_4/vector.cu 06_profiling/poisson_4/poisson.cu))
$(eval $(call MULTI_SOURCE_CUDA_EXECUTABLE,poisson_5,06_profiling/poisson_5/vector.cu 06_profiling/poisson_5/poisson.cu))
$(eval $(call MULTI_SOURCE_CUDA_EXECUTABLE,laplace_tuning,06_profiling/poisson_5/laplace_tuning_sweep.cu))

$(BUILD_DIR)/poisson_0 $(BUILD_DIR)/poisson_4: LDLIBS += -lcusparse

list:
	@printf '%s\n' $(TARGETS)

clean:
	$(RM) -r $(BUILD_DIR)
