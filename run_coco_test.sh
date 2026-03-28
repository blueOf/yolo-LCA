#!/bin/bash
# COCO测试快速启动脚本 / COCO Test Quick Launch Script
#
# 使用方法 / Usage:
#   ./run_coco_test.sh [quick|full|compare]
#
# 选项 / Options:
#   quick    - 快速验证 (5 epochs, 8 images)
#   full     - 完整测试 (300 epochs, COCO 80-class)
#   compare  - 对比测试 (300 epochs each, Detect vs ContrastDetect)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

check_requirements() {
    print_header "检查环境 / Checking Environment"

    # Check Python
    if ! command -v python &> /dev/null; then
        print_error "Python未找到 / Python not found"
        exit 1
    fi
    print_success "Python: $(python --version)"

    # Check PyTorch
    if python -c "import torch" 2>/dev/null; then
        print_success "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
    else
        print_error "PyTorch未安装 / PyTorch not installed"
        exit 1
    fi

    # Check CUDA
    if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        print_success "CUDA: Available ($(python -c 'import torch; print(torch.cuda.get_device_name(0))'))"
    else
        print_warning "CUDA: Not available, will use CPU (slow!)"
    fi

    # Check Ultralytics
    if python -c "from ultralytics import YOLO" 2>/dev/null; then
        print_success "Ultralytics: Installed"
    else
        print_error "Ultralytics未安装 / Ultralytics not installed"
        exit 1
    fi

    echo ""
}

run_quick_validation() {
    print_header "运行快速验证 / Running Quick Validation"
    print_warning "这将需要约5-10分钟 / This will take about 5-10 minutes"
    echo ""

    python quick_validation.py

    if [ $? -eq 0 ]; then
        print_success "快速验证完成！/ Quick validation completed!"
        echo ""
        echo "现在可以运行完整测试 / You can now run full test:"
        echo "  ./run_coco_test.sh full"
    else
        print_error "验证失败 / Validation failed"
        exit 1
    fi
}

run_full_training() {
    print_header "运行完整COCO训练 / Running Full COCO Training"
    print_warning "这将需要约2-3天 / This will take about 2-3 days"
    echo ""
    echo "配置 / Configuration:"
    echo "  - 数据集 / Dataset: COCO 80-class"
    echo "  - 模型 / Model: YOLOv8n-Contrast"
    echo "  - 轮数 / Epochs: 300"
    echo "  - 批次 / Batch: 16"
    echo ""

    read -p "继续? / Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消 / Cancelled"
        exit 0
    fi

    python train_coco_test.py

    if [ $? -eq 0 ]; then
        print_success "训练完成！/ Training completed!"
        echo ""
        echo "结果保存在 / Results saved in:"
        echo "  runs/coco_test/yolov8n-contrast-300ep/"
    else
        print_error "训练失败 / Training failed"
        exit 1
    fi
}

run_comparison() {
    print_header "运行对比测试 / Running Comparison Test"
    print_warning "这将需要约4-6天 / This will take about 4-6 days"
    echo ""
    echo "将训练两个模型 / Will train two models:"
    echo "  1. 标准YOLOv8 / Standard YOLOv8 (Detect)"
    echo "  2. 对比学习YOLOv8 / Contrastive YOLOv8 (ContrastDetect)"
    echo ""

    read -p "继续? / Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消 / Cancelled"
        exit 0
    fi

    python compare_detect_heads.py

    if [ $? -eq 0 ]; then
        print_success "对比测试完成！/ Comparison test completed!"
        echo ""
        echo "结果保存在 / Results saved in:"
        echo "  runs/coco_comparison/yolov8n-standard-300ep/"
        echo "  runs/coco_comparison/yolov8n-contrast-300ep/"
    else
        print_error "对比测试失败 / Comparison test failed"
        exit 1
    fi
}

show_usage() {
    echo "COCO测试快速启动脚本 / COCO Test Quick Launch Script"
    echo ""
    echo "使用方法 / Usage:"
    echo "  $0 [quick|full|compare]"
    echo ""
    echo "选项 / Options:"
    echo "  quick    - 快速验证 (5 epochs, 8 images, ~5-10 mins)"
    echo "  full     - 完整测试 (300 epochs, COCO 80-class, ~2-3 days)"
    echo "  compare  - 对比测试 (300 epochs each, ~4-6 days)"
    echo ""
    echo "示例 / Examples:"
    echo "  $0 quick      # 先运行快速验证"
    echo "  $0 full       # 运行完整300轮训练"
    echo "  $0 compare    # 对比两种检测头"
    echo ""
}

# Main script
main() {
    if [ $# -eq 0 ]; then
        show_usage
        exit 1
    fi

    check_requirements

    case "$1" in
        quick)
            run_quick_validation
            ;;
        full)
            run_full_training
            ;;
        compare)
            run_comparison
            ;;
        *)
            print_error "未知选项 / Unknown option: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
