.PHONY : build clean format install-python test-cpp test-onnx submodules

TYPE ?= Release
TEST ?= ON

CMAKE_OPT = -DCMAKE_BUILD_TYPE=$(TYPE)
CMAKE_OPT += -DBUILD_TEST=$(TEST)

# 初始化 git submodule（googletest 等第三方依赖，clone 后首次构建必须执行）
submodules:
	git submodule update --init --recursive

build: submodules
	mkdir -p build/$(TYPE)
	cd build/$(TYPE) && cmake $(CMAKE_OPT) ../.. && make -j8

clean:
	rm -rf build

test-cpp:
	@echo
	cd build/$(TYPE) && make test
