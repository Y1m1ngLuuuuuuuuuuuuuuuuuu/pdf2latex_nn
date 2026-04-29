#!/usr/bin/env python3
"""
环境验证脚本
检查所有依赖是否正确安装
"""

import sys

def check_environment():
    """检查环境配置"""
    print("=" * 50)
    print("环境验证检查")
    print("=" * 50)

    errors = []
    warnings = []

    # 1. Python 版本
    print(f"\n[1] Python 版本: {sys.version}")
    if sys.version_info < (3, 10):
        errors.append("Python 版本需要 >= 3.10")

    # 2. PyTorch
    try:
        import torch
        print(f"[2] PyTorch: {torch.__version__}")
        print(f"    CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    CUDA 版本: {torch.version.cuda}")
            print(f"    GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            warnings.append("CUDA 不可用（无卡模式正常）")
    except ImportError as e:
        errors.append(f"PyTorch 未安装: {e}")

    # 3. TorchVision
    try:
        import torchvision
        print(f"[3] TorchVision: {torchvision.__version__}")
    except ImportError as e:
        errors.append(f"TorchVision 未安装: {e}")

    # 4. PyTorch Geometric
    try:
        import torch_geometric
        print(f"[4] PyTorch Geometric: {torch_geometric.__version__}")
    except ImportError as e:
        errors.append(f"PyTorch Geometric 未安装: {e}")

    # 5. Ultralytics (YOLOv8)
    try:
        import ultralytics
        print(f"[5] Ultralytics: {ultralytics.__version__}")
    except ImportError as e:
        errors.append(f"Ultralytics 未安装: {e}")

    # 6. PaddleOCR
    try:
        import paddleocr
        print(f"[6] PaddleOCR: 已安装")
    except ImportError as e:
        errors.append(f"PaddleOCR 未安装: {e}")

    # 7. Transformers
    try:
        import transformers
        print(f"[7] Transformers: {transformers.__version__}")
    except ImportError as e:
        errors.append(f"Transformers 未安装: {e}")

    # 8. 其他关键库
    libs = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('PIL', 'Pillow'),
        ('cv2', 'OpenCV'),
        ('scipy', 'SciPy'),
        ('boto3', 'Boto3'),
        ('pdf2image', 'pdf2image'),
        ('PyPDF2', 'PyPDF2'),
    ]

    print(f"\n[8] 其他依赖:")
    for module, name in libs:
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', '已安装')
            print(f"    {name}: {version}")
        except ImportError:
            warnings.append(f"{name} 未安装")

    # 9. 系统工具检查
    print(f"\n[9] 系统工具:")
    import subprocess
    tools = ['pdftoppm', 'pdfinfo']
    for tool in tools:
        try:
            result = subprocess.run(['which', tool], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"    {tool}: {result.stdout.strip()}")
            else:
                warnings.append(f"{tool} 未找到（需要 poppler-utils）")
        except Exception as e:
            warnings.append(f"检查 {tool} 失败: {e}")

    # 总结
    print("\n" + "=" * 50)
    print("验证结果")
    print("=" * 50)

    if errors:
        print("\n❌ 错误:")
        for err in errors:
            print(f"  - {err}")

    if warnings:
        print("\n⚠️  警告:")
        for warn in warnings:
            print(f"  - {warn}")

    if not errors and not warnings:
        print("\n✅ 所有检查通过！环境配置正确。")
        return 0
    elif not errors:
        print("\n✅ 核心依赖已安装，有一些可选警告。")
        return 0
    else:
        print("\n❌ 环境配置有错误，请修复后重试。")
        return 1

if __name__ == "__main__":
    sys.exit(check_environment())
