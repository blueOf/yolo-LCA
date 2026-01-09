"""
合并对比两个模型的预测结果
将两个文件夹中相同名字的预测图片并排放在一起，方便对比

使用方法:
    python compare_predictions.py <folder1> <folder2> [output_folder]
    例如: python compare_predictions.py runs/detect/predict3 runs/detect/predict4 runs/detect/comparison
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def get_image_files(folder):
    """
    获取文件夹中的所有图片文件
    
    Args:
        folder: 文件夹路径
    
    Returns:
        图片文件路径字典 {文件名: 完整路径}
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = {}
    
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"错误: 文件夹不存在 {folder}")
        return image_files
    
    for file in folder_path.iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:
            image_files[file.name] = str(file)
    
    return image_files


def add_label(image, text, position='top'):
    """
    在图片上添加标签文字
    
    Args:
        image: PIL Image对象
        text: 标签文字
        position: 标签位置 ('top' 或 'bottom')
    
    Returns:
        添加标签后的图片
    """
    # 创建新图片，增加标签区域
    label_height = 40
    new_height = image.height + label_height
    new_image = Image.new('RGB', (image.width, new_height), color='white')
    
    # 粘贴原图片
    if position == 'top':
        new_image.paste(image, (0, label_height))
        text_y = 10
    else:
        new_image.paste(image, (0, 0))
        text_y = image.height + 10
    
    # 绘制文字
    draw = ImageDraw.Draw(new_image)
    
    # 尝试使用更好的字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
    
    # 计算文字位置（居中）
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (image.width - text_width) // 2
    
    # 绘制文字（带阴影效果）
    draw.text((text_x + 2, text_y + 2), text, fill='gray', font=font)
    draw.text((text_x, text_y), text, fill='black', font=font)
    
    return new_image


def merge_images_horizontal(img1_path, img2_path, label1="Model 1", label2="Model 2"):
    """
    将两张图片横向合并
    
    Args:
        img1_path: 第一张图片路径
        img2_path: 第二张图片路径
        label1: 第一张图片的标签
        label2: 第二张图片的标签
    
    Returns:
        合并后的PIL Image对象
    """
    # 加载图片
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    
    # 添加标签
    img1_labeled = add_label(img1, label1, position='top')
    img2_labeled = add_label(img2, label2, position='top')
    
    # 调整尺寸（如果两张图片高度不同）
    max_height = max(img1_labeled.height, img2_labeled.height)
    
    if img1_labeled.height != max_height:
        ratio = max_height / img1_labeled.height
        new_width = int(img1_labeled.width * ratio)
        img1_labeled = img1_labeled.resize((new_width, max_height), Image.LANCZOS)
    
    if img2_labeled.height != max_height:
        ratio = max_height / img2_labeled.height
        new_width = int(img2_labeled.width * ratio)
        img2_labeled = img2_labeled.resize((new_width, max_height), Image.LANCZOS)
    
    # 创建合并后的图片
    total_width = img1_labeled.width + img2_labeled.width + 10  # 10像素间隔
    merged_image = Image.new('RGB', (total_width, max_height), color='white')
    
    # 粘贴两张图片
    merged_image.paste(img1_labeled, (0, 0))
    merged_image.paste(img2_labeled, (img1_labeled.width + 10, 0))
    
    return merged_image


def compare_predictions(folder1, folder2, output_folder, label1=None, label2=None):
    """
    对比两个文件夹的预测结果
    
    Args:
        folder1: 第一个模型的预测结果文件夹
        folder2: 第二个模型的预测结果文件夹
        output_folder: 输出文件夹
        label1: 第一个模型的标签名
        label2: 第二个模型的标签名
    """
    # 获取两个文件夹的图片
    images1 = get_image_files(folder1)
    images2 = get_image_files(folder2)
    
    # 找到共同的图片文件名
    common_names = set(images1.keys()) & set(images2.keys())
    
    if not common_names:
        print("错误: 两个文件夹中没有相同名字的图片!")
        return
    
    print(f"\n找到 {len(common_names)} 张相同名字的图片")
    
    # 创建输出文件夹
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 自动生成标签
    if label1 is None:
        label1 = Path(folder1).name
    if label2 is None:
        label2 = Path(folder2).name
    
    print(f"\n开始合并对比:")
    print(f"  模型1 ({label1}): {folder1}")
    print(f"  模型2 ({label2}): {folder2}")
    print(f"  输出目录: {output_folder}")
    print("=" * 70)
    
    # 处理每一对图片
    success_count = 0
    for i, filename in enumerate(sorted(common_names), 1):
        try:
            img1_path = images1[filename]
            img2_path = images2[filename]
            
            # 合并图片
            merged_img = merge_images_horizontal(img1_path, img2_path, label1, label2)
            
            # 保存
            output_path_file = output_path / f"comparison_{filename}"
            merged_img.save(output_path_file, quality=95)
            
            success_count += 1
            print(f"[{i}/{len(common_names)}] ✓ {filename}")
            
        except Exception as e:
            print(f"[{i}/{len(common_names)}] ✗ {filename} - 错误: {e}")
    
    print("=" * 70)
    print(f"\n完成! 成功合并 {success_count}/{len(common_names)} 张图片")
    print(f"结果保存在: {output_folder}/\n")
    
    # 统计信息
    only_in_1 = set(images1.keys()) - set(images2.keys())
    only_in_2 = set(images2.keys()) - set(images1.keys())
    
    if only_in_1:
        print(f"仅在 {label1} 中的图片 ({len(only_in_1)} 张):")
        for name in sorted(list(only_in_1)[:5]):
            print(f"  - {name}")
        if len(only_in_1) > 5:
            print(f"  ... 还有 {len(only_in_1) - 5} 张")
    
    if only_in_2:
        print(f"\n仅在 {label2} 中的图片 ({len(only_in_2)} 张):")
        for name in sorted(list(only_in_2)[:5]):
            print(f"  - {name}")
        if len(only_in_2) > 5:
            print(f"  ... 还有 {len(only_in_2) - 5} 张")


def main():
    parser = argparse.ArgumentParser(description='合并对比两个模型的预测结果')
    parser.add_argument('folder1', type=str, help='第一个模型的预测结果文件夹')
    parser.add_argument('folder2', type=str, help='第二个模型的预测结果文件夹')
    parser.add_argument('output_folder', type=str, nargs='?', 
                       default='runs/detect/comparison',
                       help='输出文件夹（默认：runs/detect/comparison）')
    parser.add_argument('--label1', type=str, default=None,
                       help='第一个模型的标签名（默认使用文件夹名）')
    parser.add_argument('--label2', type=str, default=None,
                       help='第二个模型的标签名（默认使用文件夹名）')
    
    args = parser.parse_args()
    
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  预测结果对比工具".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    compare_predictions(
        args.folder1, 
        args.folder2, 
        args.output_folder,
        args.label1,
        args.label2
    )


if __name__ == "__main__":
    main()
