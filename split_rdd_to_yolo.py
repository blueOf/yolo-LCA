import os
import glob
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# RDD2022 核心四类道路损伤
CLASSES = ["D00", "D10", "D20", "D40"]


def convert_to_yolo_bbox(size, box):
    """将 VOC 的 [xmin, xmax, ymin, ymax] 转换为 YOLO 的 [x_center, y_center, width, height]"""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)


def parse_xml(xml_path):
    """解析 XML 文件返回类别和边界框"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    classes_present = []
    yolo_labels = []

    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if cls_name not in CLASSES:
            continue

        cls_id = CLASSES.index(cls_name)
        xmlbox = obj.find("bndbox")
        b = (
            float(xmlbox.find("xmin").text),
            float(xmlbox.find("xmax").text),
            float(xmlbox.find("ymin").text),
            float(xmlbox.find("ymax").text),
        )

        yolo_box = convert_to_yolo_bbox((w, h), b)
        yolo_labels.append(f"{cls_id} {' '.join(map(str, yolo_box))}")
        classes_present.append(cls_id)

    return classes_present, yolo_labels


def process_region(region_name, base_dir, output_dir):
    """处理单个地区的数据集"""
    print(f"\n--- 开始处理地区: {region_name} ---")

    # 根据解压后的 RDD 目录结构获取所有的 xmls
    xml_dir = os.path.join(base_dir, region_name, "train", "annotations", "xmls")
    img_dir = os.path.join(base_dir, region_name, "train", "images")

    if not os.path.exists(xml_dir) or not os.path.exists(img_dir):
        print(f"找不到 {region_name} 的数据，检查路径 {xml_dir}")
        return

    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    print(f"找到 {len(xml_files)} 个 XML 标签文件")

    image_paths = []
    labels_data = []
    multi_labels = []  # 用于分层切分的特征矩阵 (N, 4)

    for xml_file in tqdm(xml_files, desc="解析 XML"):
        img_name = os.path.basename(xml_file).replace(".xml", ".jpg")
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            continue

        classes_present, yolo_labels = parse_xml(xml_file)

        # 即使是空背景图也可能要保留，但 RDD 主要集中在有标注的数据，这里跳过没有所含四类的情况
        if not classes_present:
            continue

        # 生成 One-hot 编码用于分层 (只要该图存在该类，该位置为1)
        one_hot = np.zeros(len(CLASSES), dtype=int)
        for c in classes_present:
            one_hot[c] = 1

        image_paths.append(img_path)
        labels_data.append(yolo_labels)
        multi_labels.append(one_hot)

    X = np.array(image_paths)
    y = np.array(multi_labels)
    Y_labels = np.array(labels_data, dtype=object)

    # 第一次切分：分离出 80% Train 和 20% (包含 Val + Test)
    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, temp_idx = next(msss1.split(X, y))

    X_train = X[train_idx]
    y_train_labels = Y_labels[train_idx]

    X_temp = X[temp_idx]
    y_temp_labels = Y_labels[temp_idx]
    y_temp_stratify = y[temp_idx]

    # 第二次切分：将剩余的 20% 均等划分为 10% Val 和 10% Test
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(msss2.split(X_temp, y_temp_stratify))

    X_val = X_temp[val_idx]
    y_val_labels = y_temp_labels[val_idx]

    X_test = X_temp[test_idx]
    y_test_labels = y_temp_labels[test_idx]

    # 生成文件
    splits = {
        "train": (X_train, y_train_labels),
        "val": (X_val, y_val_labels),
        "test": (X_test, y_test_labels),
    }

    for split_name, (imgs, lbls) in splits.items():
        print(f"[{split_name.upper()}] 生成中: {len(imgs)} 张图片")
        split_img_dir = os.path.join(output_dir, split_name, "images")
        split_lbl_dir = os.path.join(output_dir, split_name, "labels")

        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_lbl_dir, exist_ok=True)

        for img_path, label_content in zip(imgs, lbls):
            base_name = os.path.basename(img_path)
            # 拷贝图片
            shutil.copy(img_path, os.path.join(split_img_dir, base_name))
            # 写入 yolo txt
            txt_path = os.path.join(split_lbl_dir, base_name.replace(".jpg", ".txt"))
            with open(txt_path, "w") as f:
                f.write("\n".join(label_content) + "\n")


def create_yaml(output_dir):
    """自动生成 YOLO 训练用 YAML 配置文件"""
    yaml_content = f"""path: {os.path.abspath(output_dir)}
train: train/images
val: val/images
test: test/images

# Classes
names:
  0: D00  # Longitudinal Crack
  1: D10  # Transverse Crack
  2: D20  # Alligator Crack
  3: D40  # Pothole
"""
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\n✅ YOLO 配置文件已生成：{yaml_path}")


if __name__ == "__main__":
    # 请先确保以下压缩包已经进行了解压，例如解压到了 /home/nkq/yolo11 下
    base_raw_dir = (
        "."  # 原始数据集所在的根目录 (例如包含 Japan, India 等解压出的文件夹)
    )

    regions = [
        "Japan",
        "India",
        "Czech",
        "United_States",
    ]  # 需确保这几个文件夹名与解压出来的匹配

    for r in regions:
        output_yolo_dir = f"./datasets/RDD2022_YOLO_{r}"  # 每个地区单独生成一个数据集
        process_region(r, base_raw_dir, output_yolo_dir)
        create_yaml(output_yolo_dir)

    print("ALL DONE. 数据集转换、分层切分完成。")
