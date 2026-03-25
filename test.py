# COCO数据集下载，断点可以按照现有文件继续下载后续内容。

import os
import requests


def download_file(url, local_filename):
    # 如果文件已经存在，确定其大小
    if os.path.exists(local_filename):
        # 获取已下载文件的大小
        downloaded_size = os.path.getsize(local_filename)
    else:
        downloaded_size = 0

    # 如果文件存在且大小不为0，则设置Range头进行断点续传
    headers = {}
    if downloaded_size:
        headers = {"Range": f"bytes={downloaded_size}-"}

    with requests.get(url, stream=True, headers=headers) as r:
        # 检查服务器是否支持范围请求
        if downloaded_size and r.status_code == 416:
            print(f"{local_filename} already fully downloaded.")
            return local_filename

        r.raise_for_status()

        # 以追加模式打开文件
        mode = "ab" if downloaded_size else "wb"
        with open(local_filename, mode) as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # 过滤掉保持连接的空块
                    f.write(chunk)
                    # 更新已下载的文件大小
                    downloaded_size += len(chunk)
                    print(f"Downloaded {downloaded_size} bytes of {local_filename}")

    return local_filename


# 下载 COCO 数据集
urls = [
    #'http://images.cocodataset.org/zips/train2017.zip',
    #'http://images.cocodataset.org/zips/val2017.zip',
    #'http://images.cocodataset.org/zips/test2017.zip',
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
]

for url in urls:
    filename = os.path.join(os.getcwd(), os.path.basename(url))
    print(f"Downloading {filename}...")
    download_file(url, filename)
    print(f"{filename} downloaded successfully.")
