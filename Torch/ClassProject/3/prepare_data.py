import os
import shutil
import numpy as np
from PIL import Image
from collections import Counter

def extract_dominant_color(img_array, top_k=5):
    """提取图片中的主导颜色"""
    pixels = img_array.reshape(-1, 3)
    pixel_counts = Counter(map(tuple, pixels))
    most_common = pixel_counts.most_common(top_k)
    dominant = np.mean([np.array(c[0]) for c in most_common], axis=0)
    return dominant

def classify_hair_color(r, g, b):
    """根据RGB值判断头发颜色"""
    brightness = (r + g + b) / 3
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    saturation = (max_val - min_val) if max_val > 0 else 0
    # 灰色判断：低饱和度
    if saturation < 25 and brightness < 180:
        return 'grey'
    # 黄色判断：红色和绿色较高，蓝色较低
    if r > 150 and g > 130 and b < 160 and r > b + 20:
        return 'yellow'
    # 蓝色判断：蓝色最高或较高
    if b > r and b > g:
        return 'blue'
    # 红色判断：红色为主
    if r > 180 and r > g + 10 and r > b + 10:
        return 'red'
    
    # 根据亮度判断
    if brightness < 100:
        return 'grey'
    elif brightness > 180:
        return 'yellow'
    else:
        return 'blue'

def prepare_dataset():
    """准备数据集：清理旧目录并创建新的目录结构"""
    # 删除旧目录
    for d in ['animefaces', 'data']:
        if os.path.exists(d):
            shutil.rmtree(d)
    # 创建目录结构
    for split in ['train', 'test']:
        for color in ['grey', 'yellow', 'blue', 'red']:
            os.makedirs(f'./data/{split}/{color}', exist_ok=True)
    
    print("目录结构已创建")

def split_dataset():
    """划分数据集：按80%训练、20%测试划分"""
    faces_dir = './faces'
    color_stats = {'grey': 0, 'yellow': 0, 'blue': 0, 'red': 0}
    
    print("正在分析并分类图片...")
    files = sorted([f for f in os.listdir(faces_dir) if f.endswith('.jpg')], 
                   key=lambda x: int(x.replace('.jpg', '')))
    
    for i, f in enumerate(files):
        if i % 10000 == 0:
            print(f"处理 {i}/{len(files)}...")
        img = Image.open(os.path.join(faces_dir, f))
        img_array = np.array(img)
        # 提取头发区域（顶部1/3）
        h, w = img_array.shape[:2]
        hair_region = img_array[:h//3, :, :]
        # 获取主导颜色
        dominant = extract_dominant_color(hair_region, top_k=3)
        r, g, b = dominant
        # 分类
        color = classify_hair_color(r, g, b)
        color_stats[color] += 1
        # 80%训练，20%测试
        split = 'train' if i % 5 != 0 else 'test'
        dst = f'./data/{split}/{color}/{f}'
        shutil.copy(os.path.join(faces_dir, f), dst)
    
    print(f"\n数据集统计:")
    print(f"总分类数: {sum(color_stats.values())}")
    print(f"\n各类别数量:")
    for color, count in sorted(color_stats.items()):
        print(f"  {color}: {count}")

def print_dataset_info():
    """打印数据集信息"""
    print("\n训练集分布:")
    for color in ['grey', 'yellow', 'blue', 'red']:
        count = len(os.listdir(f'./data/train/{color}'))
        print(f"  {color}: {count}")
    
    print("\n测试集分布:")
    for color in ['grey', 'yellow', 'blue', 'red']:
        count = len(os.listdir(f'./data/test/{color}'))
        print(f"  {color}: {count}")

if __name__ == '__main__':
    print("=" * 50)
    print("准备目录结构")
    prepare_dataset()
    
    print("\n" + "=" * 50)
    print("划分数据集")
    split_dataset()
    
    print("\n" + "=" * 50)
    print("数据集准备完成!")
    print_dataset_info()
