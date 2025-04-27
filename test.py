import sys  # 导入系统模块，用于处理系统相关的操作
import traceback  # 导入traceback模块，用于处理异常信息
import serial  # 导入串口通信库
import threading  # 导入线程模块，用于创建串口读取线程
from collections import defaultdict  # 导入defaultdict用于存储食物重量数据
from llama_cpp import Llama  # 导入phi-4模型加载库

try:
    import os  # 导入操作系统模块，用于处理文件和目录操作
    import time  # 导入时间模块，用于处理时间相关的操作
    import torch  # 导入PyTorch库，用于深度学习任务
    import torch.nn as nn  # 导入神经网络模块，用于构建神经网络
    import torch.optim as optim  # 导入优化器模块，用于优化模型参数
    import numpy as np  # 导入NumPy库，用于数值计算
    from PIL import Image  # 导入PIL库，用于图像处理
    
    # 如果上述库成功导入，继续导入其他库
    import torchvision  # 导入torchvision库，用于计算机视觉任务
    import torchvision.transforms as transforms  # 导入图像预处理模块
    import torchvision.models as models  # 导入预训练模型
    from torch.utils.data import DataLoader, Dataset  # 导入数据加载器模块
    import cv2  # 导入OpenCV库，用于图像处理
    import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘图
    import pickle  # 导入pickle模块，用于序列化和反序列化Python对象
    import math  # 导入数学模块，用于数学运算
    from torchvision.models.resnet import ResNet50_Weights
    import logging  # 导入日志模块，用于记录日志信息
except ImportError as e:
    logging.error(f"导入错误: {e}")  # 记录导入错误信息
    print("请安装所需的库: pip install torch==2.2.0 torchvision==0.17.0 opencv-python pillow numpy matplotlib")  # 提示用户安装缺失的库
    sys.exit(1)  # 退出程序，返回状态码1

print(f"Python路径: {sys.path}")  # 打印当前Python路径，用于调试和检查模块导入问题

try:
    # 尝试导入torch
    print(f"PyTorch版本: {torch.__version__}")
except ImportError:
    print("导入PyTorch失败，请确保安装了兼容版本的PyTorch")
    print("请尝试运行: pip install torch==2.2.0")
    sys.exit(1)

try:
    # 尝试导入torch的子模块
    print("PyTorch子模块导入成功")
except ImportError as e:
    print(f"导入PyTorch子模块失败: {e}")
    print("请尝试重新安装PyTorch: pip install torch==2.2.0")
    sys.exit(1)

print("所有PyTorch基本模块导入成功!")

# 简单测试PyTorch功能
x = torch.randn(3, 3)
print("创建张量成功:")
print(x)

# 如果成功，可以继续导入其他模块
try:
    print("基本库导入成功")
    
    # 可选：继续导入更复杂的库
    print("torchvision导入成功")
    
    print("OpenCV导入成功")
    
    print("matplotlib导入成功")
    
    print("所有库导入成功!")
except ImportError as e:
    print(f"导入错误: {e}")
    print("部分库导入失败，但PyTorch基本功能正常")
    
print("基本测试完成，程序运行正常!")

# 检查是否有可用的CUDA
# 检查GPU可用性并设置设备
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("未检测到GPU，使用CPU")
print(f"使用设备: {device}")

# 串口数据处理类
class SerialDataProcessor:
    def __init__(self, port='COM3', baudrate=9600, max_retries=3):
        self.port = port
        self.baudrate = baudrate
        self.max_retries = max_retries
        self.current_weight = 0.0
        self.lock = threading.Lock()
        self.serial_connection = None
        self._initialize_serial()
        
    def _initialize_serial(self):
        """初始化串口连接，带有错误处理和重试机制"""
        retries = 0
        while retries < self.max_retries:
            try:
                self.serial_connection = serial.Serial(self.port, self.baudrate)
                print(f"成功连接到串口 {self.port}")
                return
            except Exception as e:
                retries += 1
                print(f"尝试 {retries}/{self.max_retries}: 无法连接到串口 {self.port}, 错误: {e}")
                if retries < self.max_retries:
                    time.sleep(1)
        
        print(f"警告: 无法连接到串口 {self.port}, 将使用模拟数据模式")
        self.serial_connection = None
        
    def start_reading(self):
        """启动串口数据读取线程"""
        threading.Thread(target=self._read_serial_data, daemon=True).start()
        
    def _read_serial_data(self):
        """持续读取串口数据"""
        while True:
            try:
                if self.serial_connection is None:
                    # 模拟模式，返回固定重量
                    with self.lock:
                        self.current_weight = 100.0
                    time.sleep(1)
                    continue
                    
                if self.serial_connection.in_waiting > 0:
                    data = self.serial_connection.readline().decode('utf-8').strip()
                    try:
                        weight = float(data)
                        with self.lock:
                            self.current_weight = weight
                    except ValueError:
                        print(f"无效的串口数据: {data}")
            except Exception as e:
                print(f"串口读取错误: {e}")
                if self.serial_connection:
                    self.serial_connection.close()
                self._initialize_serial()
                time.sleep(1)
                
    def get_current_weight(self):
        """获取当前重量数据"""
        with self.lock:
            return self.current_weight

# 食物重量计算函数
def calculate_food_weights(food_proportions, total_weight):
    """
    根据食物比例和总重量计算各类食物重量
    :param food_proportions: 字典 {食物名称: 比例(0-1)}
    :param total_weight: 从串口获取的总重量
    :return: 字典 {食物名称: 重量}
    """
    return {food: proportion * total_weight for food, proportion in food_proportions.items()}

# 图像识别结果处理函数
def process_detection_results(detections, class_names, confidence_threshold=0.5):
    """
    处理图像识别结果，计算各类食物比例
    :param detections: 模型输出的检测结果
    :param class_names: 类别名称列表
    :param confidence_threshold: 置信度阈值
    :return: 字典 {食物名称: 比例}
    """
    # 统计各类食物的像素面积
    food_areas = defaultdict(float)
    total_area = 0.0
    
    for detection in detections:
        if detection['confidence'] > confidence_threshold:
            class_id = detection['class_id']
            food_name = class_names[class_id]
            area = (detection['bbox'][2] - detection['bbox'][0]) * (detection['bbox'][3] - detection['bbox'][1])
            food_areas[food_name] += area
            total_area += area
    
    # 计算各类食物比例
    if total_area > 0:
        return {food: area/total_area for food, area in food_areas.items()}
    return {}

# 计算食物营养成分函数
def calculate_nutrition(food_weights, nutrition_db=None):
    """
    根据食物重量和营养成分数据库计算总营养成分
    :param food_weights: 食物重量字典 {食物名称: 重量}
    :param nutrition_db: 营养成分数据库字典 {食物名称: {营养成分: 每100克含量}}
    :return: 字典 {营养成分: 总量}
    """
    if nutrition_db is None:
        nutrition_db = DEFAULT_NUTRITION_DB
    
    total_nutrition = defaultdict(float)
    
    for food, weight in food_weights.items():
        if food in nutrition_db:
            for nutrient, value in nutrition_db[food].items():
                total_nutrition[nutrient] += value * weight / 100
    
    return dict(total_nutrition)

# 加载phi-4模型
def load_phi4_model(model_path="D:\\module\\phi-4\\phi-4-Q4_K_M.gguf"):
    """
    加载本地部署的phi-4模型
    :param model_path: 模型文件路径
    :return: 加载的模型对象
    """
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4
        )
        return llm
    except Exception as e:
        print(f"加载phi-4模型失败: {e}")
        return None

# 生成营养建议
def generate_nutrition_advice(llm, food_weights):
    """
    使用phi-4模型生成营养建议
    :param llm: 加载的phi-4模型
    :param food_weights: 食物重量字典 {食物名称: 重量}
    :return: 生成的营养建议文本
    """
    if llm is None:
        return "无法生成建议: 模型未加载"
    
    prompt = f"根据以下食物摄入量生成营养分析和建议:\n{food_weights}"
    output = llm.create_completion(
        prompt,
        max_tokens=512,
        temperature=0.7,
        top_p=0.9
    )
    
    return output['choices'][0]['text']

# 显示结果函数
def display_results(image, food_weights, position=(10, 30), font_size=20):
    """
    在图像上显示食物重量和营养成分计算结果
    :param image: 原始图像
    :param food_weights: 食物重量字典 {食物名称: 重量}
    :param position: 文字起始位置 (x, y)
    :param font_size: 字体大小
    :return: 添加了文字信息的图像
    """
    result_img = image.copy()
    x, y = position
    
    # 显示标题
    result_img = cv2_put_chinese_text(result_img, "食物重量计算结果:", (x, y), font_size, (0, 0, 255))
    y += font_size + 10
    
    # 显示各类食物重量
    for food, weight in food_weights.items():
        text = f"{food}: {weight:.2f}g"
        result_img = cv2_put_chinese_text(result_img, text, (x, y), font_size, (0, 0, 0))
        y += font_size + 5
    
    # 计算并显示营养成分
    nutrition = calculate_nutrition(food_weights)
    if nutrition:
        y += font_size + 10
        result_img = cv2_put_chinese_text(result_img, "营养成分计算结果:", (x, y), font_size, (0, 0, 255))
        y += font_size + 10
        
        for nutrient, value in nutrition.items():
            text = f"{nutrient}: {value:.2f}"
            result_img = cv2_put_chinese_text(result_img, text, (x, y), font_size, (0, 0, 0))
            y += font_size + 5
        
        # 加载phi-4模型并生成营养建议
        llm = load_phi4_model()
        if llm:
            y += font_size + 10
            result_img = cv2_put_chinese_text(result_img, "营养建议:", (x, y), font_size, (0, 0, 255))
            y += font_size + 10
            
            advice = generate_nutrition_advice(llm, food_weights)
            # 分割建议文本以适应显示
            for line in advice.split('\n')[:3]:  # 只显示前3行建议
                result_img = cv2_put_chinese_text(result_img, line, (x, y), font_size-2, (0, 0, 0))
                y += font_size + 5
    
    return result_img

# 优化数据加载器以支持GPU加速
def get_dataloaders(dataset, batch_size=32, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

# 解决OpenCV中文显示问题的函数
def cv2_put_chinese_text(img, text, position, font_size, text_color, bg_color=None):
    """
    在图像上绘制中文文字，支持背景色
    :param img: 图像
    :param text: 文本
    :param position: 位置，元组(x, y)
    :param font_size: 字体大小
    :param text_color: 文字颜色，元组(B, G, R)
    :param bg_color: 背景颜色，元组(B, G, R)，默认为None表示无背景
    :return: 添加文字后的图像
    """
    # 创建文字图像
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    import cv2
    
    # 创建一个与text大小相同的临时图像
    # 获取文本大小
    try:
        font = ImageFont.truetype("simhei.ttf", font_size)
    except IOError:
        try:
            # 尝试使用另一种常见字体
            font = ImageFont.truetype("simkai.ttf", font_size)
        except IOError:
            try:
                # 尝试使用系统字体
                font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
            except IOError:
                # 如果找不到任何中文字体，使用默认字体
                font = ImageFont.load_default()
                print("警告: 未找到中文字体，使用默认字体")
    
    # 创建临时图像
    temp_img = np.zeros((font_size*3, len(text)*font_size, 3), dtype=np.uint8)
    temp_img.fill(255)  # 白色背景
    
    # 将OpenCV格式转换为PIL格式
    temp_pil_img = Image.fromarray(cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))
    
    # 创建绘图对象
    draw = ImageDraw.Draw(temp_pil_img)
    
    # 绘制文字
    draw.text((5, 5), text, font=font, fill=(text_color[2], text_color[1], text_color[0]))
    
    # 转回OpenCV格式
    temp_img = cv2.cvtColor(np.array(temp_pil_img), cv2.COLOR_RGB2BGR)
    
    # 获取文字区域
    gray = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 找到所有轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(contours[0])
        for cnt in contours:
            x_temp, y_temp, w_temp, h_temp = cv2.boundingRect(cnt)
            if x_temp < x:
                x = x_temp
            if y_temp < y:
                y = y_temp
            if x_temp + w_temp > x + w:
                w = x_temp + w_temp - x
            if y_temp + h_temp > y + h:
                h = y_temp + h_temp - y
        
        # 裁剪文字区域
        text_region = temp_img[y:y+h, x:x+w]
        
        # 在原图上绘制背景
        x0, y0 = position
        if bg_color is not None:
            # 创建背景图像
            bg = np.ones((h, w, 3), dtype=np.uint8)
            bg[:] = bg_color
            
            # 在原图上绘制背景
            if y0 >= 0 and x0 >= 0 and y0+h <= img.shape[0] and x0+w <= img.shape[1]:
                # 扩展背景，带有透明度
                alpha = 0.7
                img[y0:y0+h, x0:x0+w] = cv2.addWeighted(img[y0:y0+h, x0:x0+w], 1-alpha, bg, alpha, 0)
        
        # 将文字区域放到原图上
        for i in range(h):
            for j in range(w):
                if thresh[y+i, x+j] > 0:  # 只复制文字部分
                    if y0+i < img.shape[0] and x0+j < img.shape[1]:  # 防止越界
                        img[y0+i, x0+j] = temp_img[y+i, x+j]
    
    return img

# 数据集路径
DATA_DIR = r"D:\TEST3"
MODEL_PATH = os.path.join(DATA_DIR, "food_model.pth")
CLASSES_PATH = os.path.join(DATA_DIR, "classes.pkl")
STATS_PATH = os.path.join(DATA_DIR, "stats.pkl")
NUTRITION_DB_PATH = os.path.join(DATA_DIR, "nutrition_db.pkl")

# 默认食物营养成分数据库(每100克含量)
DEFAULT_NUTRITION_DB = {
    "米饭": {"热量": 116, "蛋白质": 2.6, "脂肪": 0.3, "碳水化合物": 25.6},
    "八宝粥": {"热量": 85, "蛋白质": 2.5, "脂肪": 0.5, "碳水化合物": 17.0},
    "白粥": {"热量": 30, "蛋白质": 1.0, "脂肪": 0.1, "碳水化合物": 6.0},
    "炒面": {"热量": 180, "蛋白质": 5.0, "脂肪": 7.0, "碳水化合物": 25.0},
    "葱油饼": {"热量": 220, "蛋白质": 4.0, "脂肪": 10.0, "碳水化合物": 30.0},
    "醋溜土豆丝": {"热量": 90, "蛋白质": 2.0, "脂肪": 4.0, "碳水化合物": 12.0},
    "地三鲜": {"热量": 120, "蛋白质": 3.0, "脂肪": 6.0, "碳水化合物": 15.0},
    "干煸豆角": {"热量": 110, "蛋白质": 4.0, "脂肪": 5.0, "碳水化合物": 12.0},
    "宫保鸡丁": {"热量": 200, "蛋白质": 15.0, "脂肪": 10.0, "碳水化合物": 10.0},
    "蚝油生菜": {"热量": 50, "蛋白质": 2.0, "脂肪": 3.0, "碳水化合物": 5.0},
    "红烧肉": {"热量": 300, "蛋白质": 12.0, "脂肪": 25.0, "碳水化合物": 5.0},
    "溜肉段": {"热量": 250, "蛋白质": 18.0, "脂肪": 15.0, "碳水化合物": 10.0},
    "平菇炒肉": {"热量": 150, "蛋白质": 10.0, "脂肪": 8.0, "碳水化合物": 8.0},
    "青菜炒面": {"热量": 160, "蛋白质": 5.0, "脂肪": 6.0, "碳水化合物": 22.0},
    "肉夹馍": {"热量": 280, "蛋白质": 12.0, "脂肪": 12.0, "碳水化合物": 30.0},
    "上汤菠菜": {"热量": 60, "蛋白质": 3.0, "脂肪": 4.0, "碳水化合物": 5.0},
    "什锦炒饭": {"热量": 190, "蛋白质": 6.0, "脂肪": 7.0, "碳水化合物": 25.0},
    "西红柿炒鸡蛋": {"热量": 120, "蛋白质": 8.0, "脂肪": 7.0, "碳水化合物": 5.0},
    "西兰花": {"热量": 35, "蛋白质": 3.0, "脂肪": 0.5, "碳水化合物": 5.0},
    "炸鸡腿": {"热量": 350, "蛋白质": 20.0, "脂肪": 25.0, "碳水化合物": 10.0}
}

# 数据预处理 - 修改为500×500并增强数据增强功能
def get_transforms():
    # 定义图像尺寸
    IMG_SIZE = 500
    
    return {
        'train': transforms.Compose([
            # 保持原始比例的resize方法
            transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            # 中心裁剪为正方形
            transforms.CenterCrop(IMG_SIZE),
            # 强化数据增强
            transforms.RandomApply([
                transforms.RandomHorizontalFlip(p=0.7),  # 水平翻转
                transforms.RandomVerticalFlip(p=0.3),    # 垂直翻转
                transforms.RandomRotation(25),           # 随机旋转
                transforms.RandomAffine(
                    degrees=15,                         # 随机旋转角度
                    translate=(0.15, 0.15),             # 随机平移
                    scale=(0.85, 1.15),                 # 随机缩放
                    shear=10                            # 随机错切
                ),
                transforms.ColorJitter(
                    brightness=0.15,                    # 亮度调整
                    contrast=0.15,                      # 对比度调整 
                    saturation=0.15,                    # 饱和度调整
                    hue=0.07                            # 色调调整
                ),
            ], p=0.9),  # 90%的概率应用上述增强
            
            # 保持图片质量的额外增强
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3),  # 高斯模糊
                transforms.RandomGrayscale(p=0.1),       # 随机灰度化
                transforms.RandomPerspective(distortion_scale=0.3, p=0.3),  # 透视变换
                transforms.RandomAutocontrast(p=0.2),    # 自动对比度
            ], p=0.4),  # 40%的概率应用额外增强
            
            # 转换为张量并标准化
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            
            # 移动RandomErasing到ToTensor后，因为它需要处理张量而不是PIL图像
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random'),
        ]),
        'val': transforms.Compose([
            transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(IMG_SIZE),
            # 测试时轻微数据增强以进行测试时增强(TTA)
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

# 自定义数据集类，用于保持图片原始比例
class AspectRatioPreservingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for target_class in self.classes:
            target_dir = os.path.join(root_dir, target_class)
            class_idx = self.class_to_idx[target_class]
            
            for root, _, fnames in os.walk(target_dir):
                for fname in sorted(fnames):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        path = os.path.join(root, fname)
                        self.samples.append((path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        
        try:
            # 使用PIL保持图像原始比例
            img = Image.open(path).convert('RGB')
            
            # 确保图像尺寸一致
            img = img.resize((500, 500))
            
            # 应用转换
            if self.transform is not None:
                img_tensor = self.transform(img)
                
            return img_tensor, target
        except (OSError, IOError) as e:
            logging.error(f"无法加载图像文件 {path}: {e}")
            logging.error(traceback.format_exc())
            # 返回一个占位符张量以避免类型错误
            logging.warning(f"无法加载图像文件 {path}，使用占位符张量")
            return torch.zeros(3, 500, 500), 0  # 将无效标签改为0以避免CUDA错误

# 加载数据集 - 修改为使用自定义数据集类
def load_data():
    print("加载数据集...")
    data_transforms = get_transforms()
    
    # 检查是否已有处理好的数据
    if os.path.exists(CLASSES_PATH) and os.path.exists(STATS_PATH):
        print("发现已处理的数据，正在加载...")
        with open(CLASSES_PATH, 'rb') as f:
            class_names = pickle.load(f)
        with open(STATS_PATH, 'rb') as f:
            dataset_sizes = pickle.load(f)
        
        image_datasets = {
            'train': AspectRatioPreservingDataset(os.path.join(DATA_DIR, 'train'), 
                                                 data_transforms['train']),
            'val': AspectRatioPreservingDataset(os.path.join(DATA_DIR, 'val'), 
                                               data_transforms['val'])
        }
        
        dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=16, shuffle=True, num_workers=0),
            'val': DataLoader(image_datasets['val'], batch_size=16, shuffle=False, num_workers=0)
        }
        
        # 显示数据集统计信息
        print(f"已加载数据集，类别: {class_names}")
        print("数据集统计信息:")
        print(f"训练集大小: {dataset_sizes['train']} 张图片")
        print(f"验证集大小: {dataset_sizes['val']} 张图片")
        
        return dataloaders, dataset_sizes, class_names
    
    # 如果没有处理好的数据，则进行处理
    print("未发现处理好的数据，正在处理...")
    
    # 创建训练和验证目录
    os.makedirs(os.path.join(DATA_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'val'), exist_ok=True)
    
    # 获取所有类别文件夹
    class_names = [d for d in os.listdir(DATA_DIR) 
                   if os.path.isdir(os.path.join(DATA_DIR, d)) and d not in ['train', 'val']]
    
    # 初始化统计信息
    total_images = 0
    class_stats = {}
    processed_images = 0
    
    # 预先计算总图片数以显示进度
    for class_name in class_names:
        class_dir = os.path.join(DATA_DIR, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_images += len(images)
        class_stats[class_name] = {'total': len(images), 'train': 0, 'val': 0}
    
    print(f"总共发现 {total_images} 张图片在 {len(class_names)} 个类别中")
    
    # 分割数据集为训练集和验证集 (80%/20%)
    for class_index, class_name in enumerate(class_names):
        print(f"\n处理类别 [{class_index+1}/{len(class_names)}]: {class_name}")
        class_dir = os.path.join(DATA_DIR, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # 创建训练和验证目录的类别子目录
        os.makedirs(os.path.join(DATA_DIR, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, 'val', class_name), exist_ok=True)
        
        # 随机打乱图像列表
        np.random.shuffle(images)
        
        # 确定分割点 (80% 训练, 20% 验证)
        split_idx = int(len(images) * 0.8)
        
        # 复制图像到训练和验证目录
        for i, img in enumerate(images):
            src = os.path.join(class_dir, img)
            
            # 确定目标位置
            if i < split_idx:  # 训练集
                dst = os.path.join(DATA_DIR, 'train', class_name, img)
                class_stats[class_name]['train'] += 1
                destination = 'train'
            else:  # 验证集
                dst = os.path.join(DATA_DIR, 'val', class_name, img)
                class_stats[class_name]['val'] += 1
                destination = 'val'
            
            # 如果目标文件不存在，则复制
            if not os.path.exists(dst):
                with open(src, 'rb') as fsrc:
                    with open(dst, 'wb') as fdst:
                        fdst.write(fsrc.read())
            
            # 更新进度
            processed_images += 1
            progress = processed_images / total_images * 100
            print(f"\r进度: [{processed_images}/{total_images}] {progress:.1f}% - 将 {img} 复制到 {destination} 集", end="")
        
        # 打印当前类别的统计信息
        print(f"\n类别 {class_name} 处理完成: 训练集 {class_stats[class_name]['train']} 张, 验证集 {class_stats[class_name]['val']} 张")
    
    print("\n数据分割完成！")
    
    # 显示统计信息
    print("\n数据集统计信息:")
    total_train = sum(stats['train'] for stats in class_stats.values())
    total_val = sum(stats['val'] for stats in class_stats.values())
    print(f"总计: 训练集 {total_train} 张, 验证集 {total_val} 张")
    
    # 加载处理后的数据集
    image_datasets = {
        'train': AspectRatioPreservingDataset(os.path.join(DATA_DIR, 'train'), 
                                             data_transforms['train']),
        'val': AspectRatioPreservingDataset(os.path.join(DATA_DIR, 'val'), 
                                           data_transforms['val'])
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=16, shuffle=True, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=16, shuffle=False, num_workers=0)
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    # 保存类别和数据集大小信息
    with open(CLASSES_PATH, 'wb') as f:
        pickle.dump(class_names, f)
    with open(STATS_PATH, 'wb') as f:
        pickle.dump(dataset_sizes, f)
    
    print(f"\n数据处理完成，共有 {len(class_names)} 个类别")
    return dataloaders, dataset_sizes, class_names

# 构建模型
def build_model(num_classes):
    # 使用预训练的ResNet50
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    
    # 微调策略：只冻结前面几层，让更多的层可训练
    for param in list(model.parameters())[:100]:  # 只冻结前100层而非之前的"除了最后20层"
        param.requires_grad = False
    
    # 修改最后的全连接层以匹配我们的类别数
    num_ftrs = model.fc.in_features
    
    # 使用更复杂的分类器头部，包含批归一化、Dropout和非线性激活
    model.fc = nn.Sequential(
        nn.BatchNorm1d(num_ftrs),
        nn.Dropout(0.3),  # 减小dropout比率提高稳定性
        nn.Linear(num_ftrs, 1024),  # 更大的隐藏层
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.4),
        nn.Linear(1024, 512),  # 添加一层，更深的网络
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)  # 使用传入的类别数
    )
    
    # 初始化新添加的层
    for m in model.fc.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model = model.to(device)
    return model

# 训练模型
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=16):
    # 确保模型和数据在正确设备上
    model = model.to(device)
    for phase in ['train', 'val']:
        dataloaders[phase] = DataLoader(dataloaders[phase].dataset, batch_size=dataloaders[phase].batch_size, shuffle=True, num_workers=0, pin_memory=True)
    since = time.time()
    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 添加早停机制
    patience = 8  # 减少早停耐心值，适应较短的训练周期
    no_improve_epochs = 0
    last_val_loss = float('inf')
    
    print(f"\n开始训练模型，共 {num_epochs} 个周期")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f'\n周期 {epoch+1}/{num_epochs}')
        print('-' * 30)
        
        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                print("训练阶段:")
            else:
                model.eval()
                print("验证阶段:")
                
            running_loss = 0.0
            running_corrects = 0
            
            # 用于计算每个类别的准确率
            class_correct = {cls: 0 for cls in dataloaders[phase].dataset.classes}
            class_total = {cls: 0 for cls in dataloaders[phase].dataset.classes}
            
            # 迭代数据
            total_batches = len(dataloaders[phase])
            batch_count = 0
            
            for inputs, labels in dataloaders[phase]:
                batch_count += 1
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 梯度置零
                optimizer.zero_grad()
                
                # 前向传播
                # 只在训练阶段跟踪历史
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # 如果是训练阶段，则反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        # 梯度裁剪防止梯度爆炸
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # 更新每个类别的准确率统计
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_name = dataloaders[phase].dataset.classes[label]
                    class_total[class_name] += 1
                    if preds[i] == labels[i]:
                        class_correct[class_name] += 1
                
                # 显示批次进度
                progress = batch_count / total_batches * 100
                batch_acc = torch.sum(preds == labels.data).double() / inputs.size(0)
                print(f"\r批次: [{batch_count}/{total_batches}] {progress:.1f}% - 损失: {loss.item():.4f}, 准确率: {batch_acc:.4f}", end="")
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            # 保存历史
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            print(f"\n{phase} 损失: {epoch_loss:.4f} 准确率: {epoch_acc:.4f}")
            
            # 显示每个类别的准确率
            print(f"\n{phase} 各类别准确率:")
            for cls in sorted(class_correct.keys()):
                if class_total[cls] > 0:
                    cls_acc = class_correct[cls] / class_total[cls]
                    print(f"  {cls}: {cls_acc:.4f} ({class_correct[cls]}/{class_total[cls]})")
            
            # 如果模型表现更好，则复制权重
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict().copy()
                print(f"保存新的最佳模型，准确率: {best_acc:.4f}")
        
        # 计算当前周期用时
        epoch_time = time.time() - epoch_start_time
        print(f"\n周期 {epoch+1} 用时: {epoch_time//60:.0f}分 {epoch_time%60:.0f}秒")
        
        # 估计剩余时间
        elapsed_time = time.time() - since
        estimated_total_time = elapsed_time / (epoch + 1) * num_epochs
        estimated_remaining_time = estimated_total_time - elapsed_time
        
        print(f"已用时: {elapsed_time//60:.0f}分 {elapsed_time%60:.0f}秒")
        print(f"估计剩余时间: {estimated_remaining_time//60:.0f}分 {estimated_remaining_time%60:.0f}秒")
        print(f"估计总时间: {estimated_total_time//60:.0f}分 {estimated_total_time%60:.0f}秒")
        
        # 检查早停机制
        if epoch_loss < last_val_loss:
            last_val_loss = epoch_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        if no_improve_epochs >= patience:
            print(f"\n早停机制触发，停止训练")
            break
    
    time_elapsed = time.time() - since
    print(f'\n训练完成，用时 {time_elapsed//60:.0f}分 {time_elapsed%60:.0f}秒')
    print(f'最佳验证准确率: {best_acc:.4f}')
    
    # 绘制训练历史图表
    try:
        plt.figure(figsize=(12, 4))
        
        # 绘制损失图
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='训练损失')
        plt.plot(history['val_loss'], label='验证损失')
        plt.xlabel('周期')
        plt.ylabel('损失')
        plt.title('训练和验证损失')
        plt.legend()
        
        # 绘制准确率图
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='训练准确率')
        plt.plot(history['val_acc'], label='验证准确率')
        plt.xlabel('周期')
        plt.ylabel('准确率')
        plt.title('训练和验证准确率')
        plt.legend()
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, 'training_history.png'))
        print(f"训练历史图表已保存到 {os.path.join(DATA_DIR, 'training_history.png')}")
    except Exception as e:
        print(f"绘制训练历史图表时出错: {e}")
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

# 加载或训练模型
def get_model(class_names):
    num_classes = len(class_names)
    
    # 检查是否有保存的模型
    if os.path.exists(MODEL_PATH):
        print("发现已保存的模型，正在加载...")
        model = build_model(num_classes)
        try:
            # 加载模型权重，忽略不匹配的键
            state_dict = torch.load(MODEL_PATH, weights_only=True)
            model_state_dict = model.state_dict()
            # 过滤掉不匹配的键
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
            # 加载过滤后的权重
            model_state_dict.update(filtered_state_dict)
            model.load_state_dict(model_state_dict)
            model.eval()
            print("模型加载完成")
            return model
        except (OSError, IOError) as e:
            print(f"加载模型时出错: {e}")
            print("将构建新模型...")
    
    print("未发现保存的模型，创建新模型...")
    # 构建一个新模型
    model = build_model(num_classes)
    model.eval()  # 设置为评估模式
    
    # 保存新模型
    if not os.path.exists(MODEL_PATH):
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"新模型已保存到 {MODEL_PATH}")
    
    return model

# 用于实时识别的图像预处理
def preprocess_image(image, augment=False):
    # 确保图像是RGB格式
    if len(image.shape) == 2:  # 灰度图
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA图像
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 3:  # BGR图像（OpenCV默认格式）
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 转换为PIL图像以应用变换
    image_pil = Image.fromarray(image)
    
    # 如果需要测试时增强，创建多个变换版本
    if augment:
        transforms_list = [
            # 原始图像
            transforms.Compose([
                transforms.Resize((500, 500), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            # 水平翻转
            transforms.Compose([
                transforms.Resize((500, 500), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            # 轻微调整亮度
            transforms.Compose([
                transforms.Resize((500, 500), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ColorJitter(brightness=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            # 轻微旋转
            transforms.Compose([
                transforms.Resize((500, 500), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            # 中心裁剪
            transforms.Compose([
                transforms.Resize((550, 550), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop((500, 500)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            # 添加对比度变化
            transforms.Compose([
                transforms.Resize((500, 500), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ColorJitter(contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            # 添加饱和度变化
            transforms.Compose([
                transforms.Resize((500, 500), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ColorJitter(saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        ]
        
        # 应用多个变换并堆叠结果
        batch_tensor = torch.stack([transform(image_pil) for transform in transforms_list])
        return batch_tensor
    else:
        # 不进行增强，只进行标准预处理
        transform = transforms.Compose([
            transforms.Resize((500, 500), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 转换为PyTorch张量
        image = transform(image_pil)
        
        # 添加批次维度
        image = image.unsqueeze(0)
        
        return image

# 检测食物并绘制轮廓
def detect_food(frame, model, class_names, confidence_threshold=0.7, show_progress=False, use_tta=True):
    # 记录开始时间用于性能测量
    if show_progress:
        start_time = time.time()
    
    # 原始帧的副本用于显示
    display_frame = frame.copy()
    height, width = frame.shape[:2]
    
    # 处理整个画面
    # 设置模型输入大小
    IMG_SIZE = 500
    
    if show_progress:
        print("\r预处理: 创建正方形画布...", end="")
    
    # 创建一个正方形画布
    max_dim = max(height, width)
    square_frame = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    # 将原始帧放在中心位置
    start_y = (max_dim - height) // 2
    start_x = (max_dim - width) // 2
    square_frame[start_y:start_y+height, start_x:start_x+width] = frame
    
    if show_progress:
        print("\r预处理: 调整大小并处理图像...", end="")
    
    # 预处理图像，根据设置决定是否使用测试时增强
    input_tensor = preprocess_image(square_frame, augment=use_tta)
    input_tensor = input_tensor.to(device)
    
    if show_progress:
        print("\r预测: 使用模型分析图像...", end="")
    
    # 预测
    model.eval()  # 确保模型处于评估模式
    with torch.no_grad():
        if use_tta:
            # 使用测试时增强 - 平均多个预测结果
            outputs = model(input_tensor)
            # 对所有增强版本的输出取平均
            outputs_mean = torch.mean(outputs, dim=0, keepdim=True)
            probs = torch.nn.functional.softmax(outputs_mean, dim=1)
            
            # 计算各个增强版本的一致性分数
            consistency_scores = []
            preds = []
            for i in range(outputs.size(0)):
                prob_i = torch.nn.functional.softmax(outputs[i].unsqueeze(0), dim=1)
                pred_i = torch.argmax(prob_i, dim=1).item()
                preds.append(pred_i)
                consistency_scores.append(prob_i[0, pred_i].item())
            
            # 计算预测的一致性 - 同一类别预测的比例
            most_common_pred = max(set(preds), key=preds.count)
            consistency = preds.count(most_common_pred) / len(preds)
            
            # 如果一致性低于阈值，降低整体置信度
            if consistency < 0.6:  # 如果少于60%的增强版本给出相同预测
                probs *= consistency  # 降低置信度
        else:
            # 单一预测
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        
        # 获取最高概率及其类别
        confidence, class_id = torch.max(probs, 1)
        confidence = confidence.item()
        class_id = class_id.item()
    
    if show_progress:
        print(f"\r预测完成: 类别 = {class_names[class_id]}, 置信度 = {confidence:.4f}", end="")
    
    # 获取前3个可能的类别及其概率，用于更丰富的显示
    topk_values, topk_indices = torch.topk(probs, min(3, len(class_names)), dim=1)
    top_predictions = [(class_names[idx.item()], val.item()) for val, idx in zip(topk_values[0], topk_indices[0])]
    
    # 只有当置信度高于阈值时才处理
    if confidence > confidence_threshold:
        label = class_names[class_id]
        color = get_color_for_class(class_id)
        
        # 对图像进行分割以找到物体轮廓
        try:
            if show_progress:
                print("\r后处理: 提取物体轮廓...", end="")
            
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 应用高斯模糊
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 使用Canny边缘检测器，调整参数以获取更好的边缘
            edges = cv2.Canny(blurred, 50, 150)
            
            # 应用形态学操作以连接断开的边缘
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # 寻找轮廓
            contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if show_progress:
                print(f"\r后处理: 找到 {len(contours)} 个轮廓", end="")
            
            # 如果找到轮廓
            if contours:
                # 按面积排序并保留最大的几个轮廓
                sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                max_contours = sorted_contours[:3]  # 保留最大的3个轮廓
                
                # 创建一个蒙版用于存储所有有效轮廓
                valid_contours = []
                
                # 面积阈值根据图像尺寸动态调整
                min_area_threshold = (height * width) * 0.005  # 图像面积的0.5%
                
                for contour in max_contours:
                    # 计算轮廓面积
                    area = cv2.contourArea(contour)
                    
                    # 忽略太小的轮廓
                    if area < min_area_threshold:
                        continue
                    
                    # 简化轮廓 - 使用更优的参数
                    epsilon = 0.005 * cv2.arcLength(contour, True)
                    approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                    
                    valid_contours.append(approx_contour)
                
                # 绘制有效轮廓
                cv2.drawContours(display_frame, valid_contours, -1, color, 2)
                
                # 在轮廓附近显示标签
                if valid_contours:
                    # 计算所有轮廓的最上方点
                    min_y = height
                    for contour in valid_contours:
                        for point in contour:
                            if point[0][1] < min_y:
                                min_y = point[0][1]
                    
                    # 计算一个适合放置标签的中心位置
                    center_x = width // 2
                    label_pos = (center_x - 100, max(min_y - 30, 30))  # 确保标签不超出画面顶部
                    
                    # 使用中文绘制标签
                    text = f"{label}: {confidence:.2f}"
                    display_frame = cv2_put_chinese_text(display_frame, text, label_pos, 36, (255, 255, 255), color)
                    
                    # 添加到类别面积计算
                    total_area = sum(cv2.contourArea(contour) for contour in valid_contours)
                    
                    # 在左上角显示检测到的物体
                    display_frame = cv2_put_chinese_text(display_frame, "检测到的物体:", (10, 30), 28, (0, 255, 0))
                    
                    # 显示前3个最可能的类别
                    for i, (cls, prob) in enumerate(top_predictions):
                        text = f"{i+1}. {cls}: {prob:.2f}"
                        text_color = color if i == 0 else (200, 200, 200)
                        display_frame = cv2_put_chinese_text(display_frame, text, (10, 70 + i*35), 24, text_color)
            else:
                # 没有找到有效轮廓，显示全框
                cv2.rectangle(display_frame, (0, 0), (width-1, height-1), color, 2)
                
                # 使用中文绘制标签
                text = f"{label}: {confidence:.2f}"
                display_frame = cv2_put_chinese_text(display_frame, text, (width//2 - 100, 40), 36, (255, 255, 255), color)
                
                # 在左上角显示检测到的物体
                display_frame = cv2_put_chinese_text(display_frame, "检测到的物体:", (10, 30), 28, (0, 255, 0))
                
                # 显示前3个最可能的类别
                for i, (cls, prob) in enumerate(top_predictions):
                    text = f"{i+1}. {cls}: {prob:.2f}"
                    text_color = color if i == 0 else (200, 200, 200)
                    display_frame = cv2_put_chinese_text(display_frame, text, (10, 70 + i*35), 24, text_color)
        except (OSError, IOError) as e:
            if show_progress:
                print(f"\r轮廓提取错误: {e}", end="")
            # 出错时回退到全框
            cv2.rectangle(display_frame, (0, 0), (width-1, height-1), color, 2)
            
            # 使用中文绘制标签
            text = f"{label}: {confidence:.2f}"
            display_frame = cv2_put_chinese_text(display_frame, text, (width//2 - 100, 40), 36, (255, 255, 255), color)
            
            # 在左上角显示检测到的物体
            display_frame = cv2_put_chinese_text(display_frame, "检测到的物体:", (10, 30), 28, (0, 255, 0))
            
            # 显示前3个最可能的类别
            for i, (cls, prob) in enumerate(top_predictions):
                text = f"{i+1}. {cls}: {prob:.2f}"
                text_color = color if i == 0 else (200, 200, 200)
                display_frame = cv2_put_chinese_text(display_frame, text, (10, 70 + i*35), 24, text_color)
    else:
        # 没有检测到物体，显示提示信息
        display_frame = cv2_put_chinese_text(display_frame, "未检测到食物", (width//2 - 100, height//2), 36, (0, 0, 255))
        
        # 显示前3个最可能的类别，即使置信度低
        display_frame = cv2_put_chinese_text(display_frame, "最可能的类别:", (10, 30), 28, (0, 255, 0))
        for i, (cls, prob) in enumerate(top_predictions):
            text = f"{i+1}. {cls}: {prob:.2f}"
            display_frame = cv2_put_chinese_text(display_frame, text, (10, 70 + i*35), 24, (200, 200, 200))
    
    # 显示处理时间（如果启用了进度显示）
    if show_progress:
        processing_time = time.time() - start_time
        fps = 1.0 / processing_time
        print(f"\r完成: 处理时间 {processing_time:.3f}秒, FPS: {fps:.1f} ")
        
        # 在图像上显示处理时间和FPS
        fps_text = f"FPS: {fps:.1f}"
        display_frame = cv2_put_chinese_text(display_frame, fps_text, (width - 150, height - 40), 24, (0, 255, 255))
        
        # 显示TTA状态
        tta_text = f"测试时增强: {'开启' if use_tta else '关闭'}"
        display_frame = cv2_put_chinese_text(display_frame, tta_text, (width - 250, height - 80), 24, (0, 255, 255))
    
    return display_frame

# 根据类别ID获取颜色
def get_color_for_class(class_id):
    # 定义一些明显不同的颜色
    colors = [
        (0, 0, 255),     # 红色
        (0, 255, 0),     # 绿色
        (255, 0, 0),     # 蓝色
        (255, 255, 0),   # 青色
        (255, 0, 255),   # 洋红色
        (0, 255, 255),   # 黄色
        (128, 0, 0),     # 深蓝色
        (0, 128, 0),     # 深绿色
        (0, 0, 128),     # 深红色
        (128, 128, 0),   # 深青色
        (128, 0, 128),   # 深洋红色
        (0, 128, 128),   # 深黄色
        (192, 192, 192), # 浅灰色
    ]
    return colors[class_id % len(colors)]

# 合并重叠的检测框
def merge_boxes(boxes, class_names, overlap_threshold=0.3, same_class_count=2):
    if not boxes:
        return []
    
    # 按类别分组
    class_boxes = {}
    for box in boxes:
        class_id = box[4]
        if class_id not in class_boxes:
            class_boxes[class_id] = []
        class_boxes[class_id].append(box)
    
    merged_boxes = []
    
    # 处理每个类别
    for class_id, boxes_of_class in class_boxes.items():
        # 如果同一类别的框数量少于阈值，则不合并
        if len(boxes_of_class) < same_class_count:
            # 只保留置信度最高的框
            if boxes_of_class:
                best_box = max(boxes_of_class, key=lambda box: box[5])
                merged_boxes.append(best_box)
            continue
        
        # 创建一个合并跟踪列表
        merged = [False] * len(boxes_of_class)
        
        # 对于每个框
        for i in range(len(boxes_of_class)):
            if merged[i]:
                continue
                
            # 创建一个新的合并组
            current_group = [boxes_of_class[i]]
            merged[i] = True
            
            # 检查其他框
            for j in range(i + 1, len(boxes_of_class)):
                if merged[j]:
                    continue
                    
                # 计算重叠
                box1 = boxes_of_class[i][:4]  # x, y, w, h
                box2 = boxes_of_class[j][:4]  # x, y, w, h
                
                # 计算两个框的IoU
                intersection = calc_intersection(box1, box2)
                union = box1[2] * box1[3] + box2[2] * box2[3] - intersection
                iou = intersection / union if union > 0 else 0
                
                # 如果重叠足够大，则合并
                if iou > overlap_threshold:
                    current_group.append(boxes_of_class[j])
                    merged[j] = True
            
            # 合并组中的框
            if len(current_group) >= 1:
                # 计算合并框
                x_min = min(box[0] for box in current_group)
                y_min = min(box[1] for box in current_group)
                x_max = max(box[0] + box[2] for box in current_group)
                y_max = max(box[1] + box[3] for box in current_group)
                
                w = x_max - x_min
                h = y_max - y_min
                
                # 使用最高置信度
                confidence = max(box[5] for box in current_group)
                
                merged_boxes.append([x_min, y_min, w, h, class_id, confidence])
    
    return merged_boxes

# 计算两个框的交集面积
def calc_intersection(box1, box2):
    # box格式: [x, y, w, h]
    x1_min, y1_min, w1, h1 = box1
    x2_min, y2_min, w2, h2 = box2
    
    x1_max = x1_min + w1
    y1_max = y1_min + h1
    x2_max = x2_min + w2
    y2_max = y2_min + h2
    
    # 计算交集的坐标
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)
    
    # 计算交集面积
    if x_min >= x_max or y_min >= y_max:
        return 0
    return (x_max - x_min) * (y_max - y_min)

# 强制重新处理数据和训练模型
def retrain_model():
    print("开始重新处理数据和训练模型...")
    
    # 显示警告
    print("\n" + "!"*50)
    print("警告: 这将删除已有的模型和处理好的数据！")
    print("!" * 50)
    
    confirm = input("\n确定要重新训练模型吗？ (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消重新训练操作")
        return None, None
    
    print("\n开始清理旧数据...")
    
    # 删除已有的模型和数据文件
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        print(f"已删除旧模型: {MODEL_PATH}")
    
    if os.path.exists(CLASSES_PATH):
        os.remove(CLASSES_PATH)
        print(f"已删除类别数据: {CLASSES_PATH}")
    
    if os.path.exists(STATS_PATH):
        os.remove(STATS_PATH)
        print(f"已删除统计数据: {STATS_PATH}")
    
    # 删除训练/验证目录
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')
    
    if os.path.exists(train_dir):
        import shutil
        print(f"正在删除训练目录: {train_dir}")
        shutil.rmtree(train_dir)
        print(f"已删除训练目录")
    
    if os.path.exists(val_dir):
        import shutil
        print(f"正在删除验证目录: {val_dir}")
        shutil.rmtree(val_dir)
        print(f"已删除验证目录")
    
    # 删除历史图表
    history_chart = os.path.join(DATA_DIR, 'training_history.png')
    if os.path.exists(history_chart):
        os.remove(history_chart)
        print(f"已删除历史图表: {history_chart}")
    
    print("\n清理完成，开始重新处理数据...")
    
    # 加载数据并训练模型
    dataloaders, dataset_sizes, class_names = load_data()
    
    print("\n数据处理完成，开始训练模型...")
    # 构建模型
    model = build_model(len(class_names))
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 使用标签平滑增强泛化能力
    
    # 分层学习率设置 - 主干网络使用较小的学习率，全连接层使用较大的学习率
    # 为不同层设置不同的学习率
    param_groups = []
    # 主干部分使用较小的学习率
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'fc' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
    
    # 添加参数组
    param_groups.append({"params": backbone_params, "lr": 0.0005})
    param_groups.append({"params": head_params, "lr": 0.001})
    
    # 使用AdamW优化器，更好的权重衰减，更好的泛化性能
    optimizer = optim.AdamW(param_groups, weight_decay=0.01)
    
    # 学习率调度器 - 使用余弦退火学习率
    num_epochs = 16  # 设置训练周期数为16
    
    # 创建余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # 训练模型
    print("\n开始训练模型...")
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=num_epochs)
    
    # 保存模型
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型已保存到 {MODEL_PATH}")
    
    # 验证模型
    print("\n在验证集上评估模型...")
    model.eval()
    running_corrects = 0
    class_correct = {cls: 0 for cls in class_names}
    class_total = {cls: 0 for cls in class_names}
    
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            running_corrects += torch.sum(preds == labels.data)
            
            # 更新每个类别的准确率统计
            for i in range(len(labels)):
                label = labels[i].item()
                class_name = class_names[label]
                class_total[class_name] += 1
                if preds[i] == labels[i]:
                    class_correct[class_name] += 1
    
    final_acc = running_corrects.double() / dataset_sizes['val']
    print(f"最终验证准确率: {final_acc:.4f}")
    
    # 显示每个类别的准确率
    print("\n各类别准确率:")
    for cls in sorted(class_correct.keys()):
        if class_total[cls] > 0:
            cls_acc = class_correct[cls] / class_total[cls]
            print(f"  {cls}: {cls_acc:.4f} ({class_correct[cls]}/{class_total[cls]})")
    
    print("\n模型重新训练完成!")
    
    return model, class_names

# 主函数
def main():
    # 检查数据文件是否存在
    if not os.path.exists(CLASSES_PATH) or not os.path.exists(STATS_PATH):
        print('未发现处理好的数据，开始处理数据...')
        dataloaders, dataset_sizes, class_names = load_data()
    else:
        # 加载类别信息
        with open(CLASSES_PATH, 'rb') as f:
            class_names = pickle.load(f)
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print('未发现训练好的模型，开始训练模型...')
        if 'class_names' not in locals():
            print("需要先加载或处理数据以获取类别信息")
            dataloaders, dataset_sizes, class_names = load_data()
        
        model = build_model(len(class_names))
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 使用标签平滑
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # 使用AdamW
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16, eta_min=1e-6)  # 使用余弦退火
        model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=16)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"模型已保存到 {MODEL_PATH}")

    print("菜品识别系统启动...")
    
    # 加载数据
    try:
        print("正在加载类别信息...")
        if 'class_names' not in locals():
            with open(CLASSES_PATH, 'rb') as f:
                class_names = pickle.load(f)
        print(f"共加载 {len(class_names)} 个类别")
        
        # 直接创建模型，跳过训练阶段
        print("正在加载模型...")
        model = get_model(class_names)
        model.eval()  # 设置为评估模式
    except Exception as e:
        print(f"加载模型或类别信息失败: {str(e)}")
        import traceback
        traceback.print_exc()
        print("请确保模型文件存在或准备好训练数据")
        return
    
    # 获取可用摄像头
    available_cameras = []
    max_cameras = 3  # 尝试最多3个摄像头
    current_camera_index = 0
    
    # 检测可用的摄像头
    print("检测可用摄像头...")
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
            print(f"发现摄像头 {i}")
    
    if not available_cameras:
        print("未找到可用摄像头")
        return
    
    # 打开第一个摄像头
    cap = cv2.VideoCapture(available_cameras[current_camera_index])
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("摄像头已打开，开始实时识别...")
    print(f"识别的类别: {class_names}")
    print("按'C'键切换摄像头，按'R'键重新训练模型，按'P'键切换显示处理进度，按'T'键切换测试时增强，按'S'键截图保存，按'Q'键退出")
    
    # 标志变量，表示是否正在训练
    is_training = False
    training_frame = None
    
    # 是否显示处理进度
    show_progress = False
    
    # 是否使用测试时增强
    use_tta = True
    
    # 帧计数和FPS计算
    frame_count = 0
    fps = 0
    fps_time = time.time()
    
    # 模型预热 - 减少第一次推理的延迟
    print("模型预热中...")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _ = detect_food(dummy_frame, model, class_names, show_progress=False, use_tta=False)
    print("预热完成，开始实时识别")
    
    while True:
        # 如果正在训练，显示训练信息
        if is_training:
            if training_frame is None:
                # 创建一个训练进行中的画面
                training_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                training_frame = cv2_put_chinese_text(training_frame, "模型训练中...", (180, 200), 36, (0, 255, 0))
                training_frame = cv2_put_chinese_text(training_frame, "请稍候，训练完成后会自动恢复识别", (100, 250), 28, (255, 255, 255))
            
            # 显示训练画面
            cv2.imshow("菜品识别", training_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            continue
        
        # 正常处理流程
        # 读取一帧
        ret, frame = cap.read()
        
        if not ret:
            print("无法获取视频帧")
            break
        
        # 调整帧大小以加快处理速度
        frame = cv2.resize(frame, (640, 480))
        
        # 检测食物
        result_frame = detect_food(frame, model, class_names, show_progress=show_progress, use_tta=use_tta)
        
        # 计算FPS
        frame_count += 1
        if frame_count >= 10:  # 每10帧更新一次FPS
            current_time = time.time()
            fps = frame_count / (current_time - fps_time)
            fps_time = current_time
            frame_count = 0
        
        # 在右下角显示FPS
        fps_text = f"FPS: {fps:.1f}"
        result_frame = cv2_put_chinese_text(result_frame, fps_text, (result_frame.shape[1] - 150, result_frame.shape[0] - 40), 24, (0, 255, 255))
        
        # 显示TTA状态
        tta_text = f"测试时增强: {'开启' if use_tta else '关闭'}"
        result_frame = cv2_put_chinese_text(result_frame, tta_text, (result_frame.shape[1] - 250, result_frame.shape[0] - 80), 24, (0, 255, 255))
        
        # 显示摄像头信息和操作提示
        camera_info = f"当前摄像头: {available_cameras[current_camera_index]} (共{len(available_cameras)}个)"
        result_frame = cv2_put_chinese_text(result_frame, camera_info, (10, result_frame.shape[0] - 80), 24, (0, 255, 255))
        
        # 添加操作提示
        operation_hint = "按'C'键切换摄像头，按'R'键重新训练，按'P'键切换进度显示，按'T'键切换增强，按'S'键截图，按'Q'键退出"
        result_frame = cv2_put_chinese_text(result_frame, operation_hint, (10, result_frame.shape[0] - 20), 24, (255, 255, 0))
        
        # 显示结果
        cv2.imshow("菜品识别", result_frame)
        
        # 检测按键
        key = cv2.waitKey(1) & 0xFF
        
        # 按'q'退出
        if key == ord('q'):
            break
        
        # 按'p'切换进度显示
        if key == ord('p'):
            show_progress = not show_progress
            print(f"进度显示: {'开启' if show_progress else '关闭'}")
        
        # 按't'切换测试时增强
        if key == ord('t'):
            use_tta = not use_tta
            print(f"测试时增强: {'开启' if use_tta else '关闭'}")
        
        # 按's'保存截图
        if key == ord('s'):
            # 创建截图目录
            screenshots_dir = os.path.join(DATA_DIR, 'screenshots')
            os.makedirs(screenshots_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(screenshots_dir, f"screenshot_{timestamp}.jpg")
            
            # 保存截图
            cv2.imwrite(filename, result_frame)
            print(f"截图已保存: {filename}")
        
        # 按'c'切换摄像头
        if key == ord('c'):
            # 关闭当前摄像头
            cap.release()
            
            # 切换到下一个摄像头
            current_camera_index = (current_camera_index + 1) % len(available_cameras)
            print(f"切换到摄像头 {available_cameras[current_camera_index]}")
            
            # 打开新的摄像头
            cap = cv2.VideoCapture(available_cameras[current_camera_index])
            
            # 检查是否成功打开
            if not cap.isOpened():
                print(f"无法打开摄像头 {available_cameras[current_camera_index]}")
                # 尝试重新打开之前的摄像头
                current_camera_index = (current_camera_index - 1) % len(available_cameras)
                cap = cv2.VideoCapture(available_cameras[current_camera_index])
        
        # 按'r'重新训练模型
        if key == ord('r'):
            # 显示确认对话框
            confirm_frame = frame.copy()
            confirm_frame = cv2_put_chinese_text(confirm_frame, "确定要重新训练模型吗？", (120, 200), 36, (0, 0, 255))
            confirm_frame = cv2_put_chinese_text(confirm_frame, "这将删除现有模型和处理好的数据！", (100, 250), 28, (0, 0, 255))
            confirm_frame = cv2_put_chinese_text(confirm_frame, "按'Y'确认，按其他键取消", (150, 300), 28, (255, 255, 255))
            
            cv2.imshow("菜品识别", confirm_frame)
            confirm_key = cv2.waitKey(0) & 0xFF
            
            if confirm_key == ord('y'):
                print("确认重新训练模型")
                is_training = True
                
                # 在新线程中进行重新训练
                import threading
                
                def train_thread():
                    nonlocal model, class_names, is_training
                    try:
                        # 重新训练模型
                        model, class_names = retrain_model()
                        
                        # 如果用户取消了训练
                        if model is None or class_names is None:
                            print("用户取消了训练操作")
                            is_training = False 
                            return
                    finally:
                        # 无论成功与否，都要重置训练状态
                        is_training = False
                
                # 启动训练线程
                threading.Thread(target=train_thread).start()
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

class SerialReader(threading.Thread):
    """串口数据读取线程"""
    def __init__(self, port, baudrate=9600):
        super().__init__()
        self.serial_port = serial.Serial(port, baudrate)
        self.running = True
        self.data = ""
        
    def run(self):
        """持续读取串口数据"""
        while self.running:
            if self.serial_port.in_waiting > 0:
                self.data = self.serial_port.readline().decode('utf-8').strip()
                print(f"串口数据: {self.data}")
                
    def stop(self):
        """停止线程"""
        self.running = False
        self.serial_port.close()

if __name__ == "__main__":
    # 初始化串口
    try:
        serial_reader = SerialReader('COM3')  # 固定使用COM3串口
        serial_reader.start()
        print("串口读取线程已启动")
    except Exception as e:
        print(f"串口初始化失败: {e}")
    
    main()
    
    # 程序结束时停止串口线程
    if 'serial_reader' in locals():
        serial_reader.stop()
