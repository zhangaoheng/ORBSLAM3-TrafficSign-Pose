#!/bin/bash

# 自动运行脚本

echo "========================================="
echo "运行 ORB-SLAM3 测试程序"
echo "========================================="

# 返回到项目根目录
PROJECT_ROOT="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"
cd "$PROJECT_ROOT"

echo "项目根目录: $(pwd)"

# 检查 build 目录是否存在
if [ ! -d "build" ]; then
    echo "build 目录不存在，开始编译..."
    bash test/build.sh
    if [ $? -ne 0 ]; then
        echo "编译失败，退出"
        exit 1
    fi
fi

# 检查可执行文件是否存在
if [ ! -f "build/test_slam" ]; then
    echo "测试可执行文件不存在，开始编译..."
    bash test/build.sh
    if [ $? -ne 0 ]; then
        echo "编译失败，退出"
        exit 1
    fi
fi

# 进入 test 目录运行程序
cd test

echo "进入 test 目录: $(pwd)"
echo "执行: ../build/test_slam"
echo "========================================="

# 运行测试
../build/test_slam

TEST_EXIT_CODE=$?

echo "========================================="
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "测试程序执行成功"
else
    echo "测试程序执行失败 (退出码: $TEST_EXIT_CODE)"
fi
echo "========================================="

exit $TEST_EXIT_CODE
