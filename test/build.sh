#!/bin/bash

# 自动编译脚本

echo "========================================="
echo "开始编译 ORB-SLAM3 测试程序"
echo "========================================="

# 返回到项目根目录
cd "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"

echo "项目根目录: $(pwd)"

# 检查 build 目录
if [ ! -d "build" ]; then
    echo "创建 build 目录..."
    mkdir build
fi

# 进入 build 目录
cd build

# 配置和编译
echo "执行 CMake 配置..."
cmake .. -DCMAKE_BUILD_TYPE=Release

echo "编译项目..."
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "========================================="
    echo "编译成功!"
    echo "========================================="
    # 检查测试可执行文件
    if [ -f "test_slam" ]; then
        echo "测试可执行文件已生成: ./test_slam"
    fi
else
    echo "========================================="
    echo "编译失败，请检查错误信息"
    echo "========================================="
    exit 1
fi
