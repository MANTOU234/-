#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"脂"能少年焕新系统 - 前端界面
基于计算机视觉的食品识别和营养分析系统
"""

import os
import sys
import cv2
import time
import re
import numpy as np
import threading
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, 
                           QSplitter, QTextEdit, QMessageBox, QDialog, QFileDialog, QProgressBar,
                           QGroupBox, QComboBox, QLineEdit, QCheckBox)
from PyQt5.QtGui import QFont, QPixmap, QImage, QColor, QPalette, QBrush, QIcon, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize, QRect, QPoint
import torch
import serial.tools.list_ports

# 尝试导入test.py中的函数
try:
    from test import (
        SerialDataProcessor, calculate_food_weights,
        process_detection_results, detect_food, 
        get_model, preprocess_image, load_phi4_model
    )
except ImportError as e:
    print(f"错误：导入test.py模块失败: {e}")
    sys.exit(1)

# 无文字版的食物检测函数
def detect_food_no_text(frame, model, class_names, confidence_threshold=0.6):
    """
    不在图像上显示任何文字的食物检测函数
    能够检测食物并绘制轮廓，但不添加任何文字标注
    """
    try:
        # 原始帧的副本用于显示
        display_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # 处理整个画面
        # 创建一个正方形画布
        max_dim = max(height, width)
        square_frame = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        
        # 将原始帧放在中心位置
        start_y = (max_dim - height) // 2
        start_x = (max_dim - width) // 2
        square_frame[start_y:start_y+height, start_x:start_x+width] = frame
        
        # 预处理图像
        input_tensor = preprocess_image(square_frame, augment=False)
        if torch.cuda.is_available():
            input_tensor = input_tensor.to("cuda")
        
        # 预测
        model.eval()
        with torch.no_grad():
            # 单一预测
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # 获取最高概率及其类别
            confidence, class_id = torch.max(probs, 1)
            confidence = confidence.item()
            class_id = class_id.item()
        
        # 获取前3个可能的类别及其概率
        topk_values, topk_indices = torch.topk(probs, min(3, len(class_names)), dim=1)
        top_predictions = [(class_names[idx.item()], val.item()) for val, idx in zip(topk_values[0], topk_indices[0])]
        
        # 只有当置信度高于阈值时才处理
        if confidence > confidence_threshold:
            color = get_color_for_class(class_id)
            
            # 对图像进行分割以找到物体轮廓
            try:
                # 转换为灰度图
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 应用高斯模糊
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                
                # 使用Canny边缘检测器
                edges = cv2.Canny(blurred, 50, 150)
                
                # 应用形态学操作以连接断开的边缘
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
                
                # 寻找轮廓
                contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 如果找到轮廓
                if contours:
                    # 按面积排序并保留最大的几个轮廓
                    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    max_contours = sorted_contours[:5]  # 增加至最多5个轮廓用于多菜品
                    
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
                        
                        # 简化轮廓
                        epsilon = 0.005 * cv2.arcLength(contour, True)
                        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                        
                        valid_contours.append(approx_contour)
                    
                    # 为每个预测类别分配不同的颜色
                    for i, (cls, prob) in enumerate(top_predictions[:min(len(valid_contours), 3)]):
                        if i < len(valid_contours) and prob > confidence_threshold:
                            color = get_color_for_class(topk_indices[0][i].item())
                            # 绘制轮廓但不添加文字
                            cv2.drawContours(display_frame, [valid_contours[i]], -1, color, 2)
                    
                    # 如果有效轮廓数量小于预测数量，则绘制全框
                    if len(valid_contours) < len(top_predictions):
                        for i, (cls, prob) in enumerate(top_predictions):
                            if i >= len(valid_contours) and prob > confidence_threshold and i < 3:
                                color = get_color_for_class(topk_indices[0][i].item())
                                
                                # 根据索引位置划分画面区域
                                if i == 0:
                                    # 第一个区域 - 左上
                                    cv2.rectangle(display_frame, (0, 0), (width//2, height//2), color, 2)
                                elif i == 1:
                                    # 第二个区域 - 右上
                                    cv2.rectangle(display_frame, (width//2, 0), (width, height//2), color, 2)
                                elif i == 2:
                                    # 第三个区域 - 下方
                                    cv2.rectangle(display_frame, (0, height//2), (width, height), color, 2)
                else:
                    # 没有找到有效轮廓，但仍然需要显示多个食物类别
                    for i, (cls, prob) in enumerate(top_predictions[:3]):
                        if prob > confidence_threshold:
                            color = get_color_for_class(topk_indices[0][i].item())
                            
                            # 根据索引位置划分画面区域
                            if i == 0:
                                # 第一个区域 - 左上
                                cv2.rectangle(display_frame, (0, 0), (width//2, height//2), color, 2)
                            elif i == 1:
                                # 第二个区域 - 右上
                                cv2.rectangle(display_frame, (width//2, 0), (width, height//2), color, 2)
                            elif i == 2:
                                # 第三个区域 - 下方
                                cv2.rectangle(display_frame, (0, height//2), (width, height), color, 2)
            except Exception as e:
                print(f"轮廓检测错误: {e}")
                # 出错时直接使用多区域方案
                for i, (cls, prob) in enumerate(top_predictions[:3]):
                    if prob > confidence_threshold * 0.8:  # 稍微降低阈值
                        color = get_color_for_class(topk_indices[0][i].item())
                        
                        # 根据索引位置划分画面区域
                        if i == 0:
                            # 第一个区域 - 左上
                            cv2.rectangle(display_frame, (0, 0), (width//2, height//2), color, 2)
                        elif i == 1:
                            # 第二个区域 - 右上
                            cv2.rectangle(display_frame, (width//2, 0), (width, height//2), color, 2)
                        elif i == 2:
                            # 第三个区域 - 下方
                            cv2.rectangle(display_frame, (0, height//2), (width, height), color, 2)
        else:
            # 置信度不够高时，尝试显示可能性较高的类别
            for i, (cls, prob) in enumerate(top_predictions[:3]):
                if prob > confidence_threshold * 0.7:  # 降低阈值以增加检测机会
                    color = get_color_for_class(topk_indices[0][i].item())
                    
                    # 根据索引位置划分画面区域
                    if i == 0:
                        # 第一个区域 - 左上
                        cv2.rectangle(display_frame, (0, 0), (width//2, height//2), color, 2)
                    elif i == 1:
                        # 第二个区域 - 右上
                        cv2.rectangle(display_frame, (width//2, 0), (width, height//2), color, 2)
                    elif i == 2:
                        # 第三个区域 - 下方
                        cv2.rectangle(display_frame, (0, height//2), (width, height), color, 2)
        
        # 手动释放GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return display_frame
    except Exception as e:
        print(f"食物检测异常: {e}")
        import traceback
        traceback.print_exc()
        # 发生异常时返回原始帧，确保程序不会崩溃
        return frame.copy()

# 重写SerialDataProcessor的_read_serial_data方法，以适应JSON格式的重量数据
class CustomSerialDataProcessor(SerialDataProcessor):
    def __init__(self, port='COM3', baudrate=9600, max_retries=3):
        super().__init__(port, baudrate, max_retries)
        self.weight_history = []  # 存储最近的重量数据
        self.max_history = 5      # 最多保存5个历史记录
        self.last_update_time = time.time()
        self.manual_mode = False  # 手动模式标志
        self.manual_weight = 100.0  # 默认手动重量
        self.available_ports = []  # 可用串口列表
        self.update_available_ports()
        self.running = True  # 控制读取线程的运行
        self.read_thread = None  # 存储读取线程
        # 自动连接到COM3
        self.connect_to_port('COM3')
    
    def update_available_ports(self):
        """更新可用串口列表"""
        try:
            ports = list(serial.tools.list_ports.comports())
            self.available_ports = [port.device for port in ports]
            print(f"检测到可用串口: {self.available_ports}")
        except Exception as e:
            print(f"更新串口列表失败: {e}")
            self.available_ports = []
    
    def set_manual_weight(self, weight):
        """设置手动重量值"""
        try:
            weight_val = float(weight)
            with self.lock:
                self.manual_weight = weight_val
                self._update_weight_history(weight_val)
            print(f"手动设置重量: {weight_val}g")
            return True
        except (ValueError, TypeError):
            print(f"无效的重量值: {weight}")
            return False
        
    def toggle_manual_mode(self, enabled=None):
        """切换手动/自动模式"""
        if enabled is not None:
            self.manual_mode = enabled
        else:
            self.manual_mode = not self.manual_mode
        
        print(f"切换到{'手动' if self.manual_mode else '自动'}设置重量模式")
        return self.manual_mode
    
    def connect_to_port(self, port):
        """连接到指定串口"""
        if port not in self.available_ports and port != "":
            self.update_available_ports()
            if port not in self.available_ports:
                print(f"串口 {port} 不可用")
                return False
        
        # 关闭当前连接
        if self.serial_connection:
            try:
                self.serial_connection.close()
            except Exception as e:
                print(f"关闭当前串口连接时出错: {e}")
            self.serial_connection = None
        
        self.port = port
        retries = 0
        while retries < self.max_retries:
            try:
                if port == "":
                    print("未指定串口，将使用模拟模式")
                    self.serial_connection = None
                    self.toggle_manual_mode(True)
                    return True
                    
                self.serial_connection = serial.Serial(port, self.baudrate, timeout=1)
                print(f"成功连接到串口 {port}")
                return True
            except Exception as e:
                retries += 1
                print(f"尝试 {retries}/{self.max_retries}: 无法连接到串口 {port}, 错误: {e}")
                if retries < self.max_retries:
                    time.sleep(0.5)
        
        print(f"警告: 无法连接到串口 {port}, 将使用模拟数据模式")
        self.serial_connection = None
        self.toggle_manual_mode(True)
        return False
    
    def start_reading(self):
        """启动串口数据读取线程"""
        if self.read_thread is not None and self.read_thread.is_alive():
            print("串口读取线程已经在运行")
            return
            
        self.running = True
        self.read_thread = threading.Thread(target=self._read_serial_data, daemon=True)
        self.read_thread.start()
        print("串口数据读取线程已启动")
    
    def stop_reading(self):
        """停止串口数据读取线程"""
        self.running = False
        if self.read_thread and self.read_thread.is_alive():
            try:
                self.read_thread.join(timeout=1.0)  # 等待线程结束，最多1秒
                print("串口数据读取线程已停止")
            except Exception as e:
                print(f"停止串口读取线程时出错: {e}")
        
        # 关闭串口连接
        if self.serial_connection:
            try:
                self.serial_connection.close()
                print("串口连接已关闭")
            except Exception as e:
                print(f"关闭串口连接时出错: {e}")
            self.serial_connection = None
    
    def _read_serial_data(self):
        """持续读取串口数据，支持JSON格式的重量数据解析"""
        import json
        import re
        
        print("串口数据读取线程开始运行")
        error_count = 0  # 错误计数器
        last_error_time = 0  # 上次错误时间
        
        while self.running:
            try:
                if self.manual_mode or self.serial_connection is None:
                    # 手动模式，使用手动设置的重量
                    with self.lock:
                        self.current_weight = self.manual_weight
                        self._update_weight_history(self.manual_weight)
                    time.sleep(0.1)  # 减少睡眠时间至0.1秒
                    continue
                
                # 检查串口是否开启
                if not self.serial_connection.is_open:
                    try:
                        self.serial_connection.open()
                    except Exception as e:
                        print(f"重新打开串口失败: {e}")
                        # 切换到手动模式
                        self.toggle_manual_mode(True)
                        time.sleep(1)
                        continue
                    
                if self.serial_connection.in_waiting > 0:
                    try:
                        data = self.serial_connection.readline().decode('utf-8', errors='ignore').strip()
                        if data:  # 只处理非空数据
                            print(f"原始串口数据: {data}")  # 调试输出原始数据
                            
                            try:
                                # 尝试解析JSON格式：{weight:123}
                                if '{weight:' in data:
                                    # 使用正则表达式提取数字
                                    match = re.search(r'{weight:(-?\d+(\.\d+)?)}', data)
                                    if match:
                                        weight = float(match.group(1))
                                        with self.lock:
                                            self.current_weight = weight
                                            self._update_weight_history(weight)
                                        print(f"串口数据解析成功: {weight}g")
                                    else:
                                        print(f"无法解析串口数据: {data}")
                                # 尝试解析正常JSON
                                elif '{' in data and '}' in data:
                                    try:
                                        # 尝试修复格式不规范的JSON字符串
                                        if not (data.startswith('{') and data.endswith('}')):
                                            data = re.search(r'({.*?})', data).group(1)
                                        
                                        # 将单引号替换为双引号
                                        data = data.replace("'", '"')
                                        
                                        # 确保键名有双引号
                                        data = re.sub(r'([{,])\s*([a-zA-Z0-9_]+):', r'\1"\2":', data)
                                        
                                        json_data = json.loads(data)
                                        if 'weight' in json_data:
                                            weight = float(json_data['weight'])
                                            with self.lock:
                                                self.current_weight = weight
                                                self._update_weight_history(weight)
                                            print(f"JSON解析成功: {weight}g")
                                    except (json.JSONDecodeError, AttributeError) as e:
                                        print(f"JSON解析失败: {data}, 错误: {e}")
                                # 尝试直接解析数字 - 单位是克
                                else:
                                    # 清理非数字字符
                                    clean_data = re.sub(r'[^\d.-]', '', data)
                                    if clean_data:
                                        weight = float(clean_data)
                                        with self.lock:
                                            self.current_weight = weight
                                            self._update_weight_history(weight)
                                        print(f"直接解析数字: {weight}g")
                                    else:
                                        print(f"无法从数据中提取数字: {data}")
                            except ValueError as e:
                                print(f"无效的串口数据: {data}, 错误: {e}")
                    except Exception as e:
                        print(f"读取串口数据行时出错: {e}")
                        # 重置错误计数
                        current_time = time.time()
                        if current_time - last_error_time > 60:  # 如果距离上次错误超过60秒，重置计数
                            error_count = 0
                            last_error_time = current_time
                        
                        error_count += 1
                        if error_count > 5:  # 如果短时间内发生多次错误，切换到手动模式
                            print(f"检测到连续错误，切换到手动模式")
                            self.toggle_manual_mode(True)
                            error_count = 0
                else:
                    # 如果没有数据，短暂等待
                    time.sleep(0.05)
            except Exception as e:
                print(f"串口读取错误: {e}")
                
                # 记录错误
                error_count += 1
                current_time = time.time()
                if current_time - last_error_time > 60:
                    error_count = 0
                last_error_time = current_time
                
                if self.serial_connection:
                    try:
                        self.serial_connection.close()
                    except Exception as close_err:
                        print(f"关闭串口连接时出错: {close_err}")
                    self.serial_connection = None
                
                # 如果短时间内出现多次错误，切换到手动模式
                if error_count > 3:
                    # 切换到手动模式以避免持续错误
                    if not self.manual_mode:
                        self.toggle_manual_mode(True)
                    error_count = 0
                
                # 每5秒尝试重新连接一次
                time.sleep(5)
                self.update_available_ports()
                if self.port in self.available_ports:
                    self.connect_to_port(self.port)
        
        print("串口数据读取线程已退出")
        # 确保串口连接已关闭
        if self.serial_connection:
            try:
                self.serial_connection.close()
            except:
                pass
    
    def _update_weight_history(self, weight):
        """更新重量历史记录"""
        try:
            self.weight_history.append(weight)
            if len(self.weight_history) > self.max_history:
                self.weight_history.pop(0)
            self.last_update_time = time.time()
        except Exception as e:
            print(f"更新重量历史记录错误: {e}")
    
    def get_current_weight(self):
        """获取当前重量数据，如果有历史记录则取平均值减少波动"""
        try:
            with self.lock:
                # 检查是否有历史数据
                if not self.weight_history:
                    return self.current_weight
                
                # 如果自上次更新已超过1秒，且列表长度够，则使用平均值平滑处理
                if len(self.weight_history) >= 3 and time.time() - self.last_update_time < 1.0:
                    # 去掉最大和最小值后取平均，减少异常波动影响
                    sorted_weights = sorted(self.weight_history)
                    if len(sorted_weights) > 3:
                        sorted_weights = sorted_weights[1:-1]  # 去掉最大最小值
                    return sum(sorted_weights) / len(sorted_weights)
            
                return self.current_weight
        except Exception as e:
            print(f"获取当前重量时出错: {e}")
            return self.manual_weight  # 出错时返回手动设置的重量

# 视频处理线程
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    detection_signal = pyqtSignal(list)
    weight_update_signal = pyqtSignal(float)  # 新增重量更新信号
    
    def __init__(self, model, class_names, camera_id=0, parent=None):
        super().__init__(parent)
        self.model = model
        self.class_names = class_names
        self.camera_id = camera_id
        self.running = True
        self.paused = False
        self.last_detection_time = 0  # 上次检测时间
        self.detection_interval = 0.3  # 检测间隔时间（秒）
        self.last_weight = 0.0  # 上次重量
        self.weight_update_interval = 0.1  # 重量更新间隔（秒）
        self.last_weight_update = 0  # 上次重量更新时间
        self.serial_processor = None  # 串口处理器，将在run()中初始化
    
    def run(self):
        # 创建视频捕获对象
        cap = None
        try:
            cap = cv2.VideoCapture(self.camera_id)
            if not cap.isOpened():
                print(f"错误：无法打开摄像头 {self.camera_id}")
                return
                
            # 设置分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # 创建串口数据处理器 - 从主窗口获取
            try:
                # 尝试获取主窗口的串口处理器
                if hasattr(self.parent(), 'serial_processor'):
                    self.serial_processor = self.parent().serial_processor
                else:
                    # 如果主窗口没有，创建一个新的
                    self.serial_processor = CustomSerialDataProcessor()
                    self.serial_processor.start_reading()
            except Exception as e:
                print(f"警告：串口初始化失败: {e}")
                self.serial_processor = None
                
            # 上次检测的结果
            last_detections = []
            
            while self.running:
                if not self.paused:
                    try:
                        current_time = time.time()
                        
                        # 读取视频帧
                        ret, frame = cap.read()
                        if not ret:
                            time.sleep(0.01)
                            continue
                        
                        # 调整帧大小以加快处理速度
                        frame = cv2.resize(frame, (640, 480))
                        
                        # 单独处理重量更新，提高频率
                        if self.serial_processor and current_time - self.last_weight_update >= self.weight_update_interval:
                            try:
                                self.last_weight_update = current_time
                                weight = self.serial_processor.get_current_weight()
                                
                                # 任何情况下都更新UI显示
                                self.weight_update_signal.emit(weight)
                                
                                # 重量有明显变化时，更新检测结果中的重量信息
                                if abs(weight - self.last_weight) > 0.5:
                                    self.last_weight = weight
                                    # 如果有检测结果，更新重量信息
                                    if last_detections:
                                        # 使用改进的重量分配算法
                                        updated_detections = self._update_detections_weight(last_detections, weight)
                                        last_detections = updated_detections
                                        # 发送更新后的检测结果
                                        self.detection_signal.emit(last_detections)
                            except Exception as e:
                                print(f"重量更新错误: {e}")
                                
                        # 食物检测 - 降低检测频率，但保持视频流畅
                        run_detection = current_time - self.last_detection_time >= self.detection_interval
                        
                        if run_detection:
                            self.last_detection_time = current_time
                            
                            try:
                                # 使用无文字版的食物检测函数
                                original_frame = frame.copy()
                                processed_frame = detect_food_no_text(frame, self.model, self.class_names, 
                                                           confidence_threshold=0.6)
                                
                                # 发送处理后的帧用于显示
                                self.change_pixmap_signal.emit(processed_frame)
                                
                                # 从模型直接进行预测获取食物类别
                                input_tensor = preprocess_image(original_frame, augment=False)
                                if torch.cuda.is_available():
                                    input_tensor = input_tensor.cuda()
                                
                                self.model.eval()
                                with torch.no_grad():
                                    outputs = self.model(input_tensor)
                                    probs = torch.nn.functional.softmax(outputs, dim=1)
                                    
                                    # 获取前5个最可能的类别
                                    topk_values, topk_indices = torch.topk(probs, min(5, len(self.class_names)), dim=1)
                                    
                                    # 初始化检测结果列表
                                    detections = []
                                    
                                    # 检查是否多种食物并分配不同的边界框
                                    valid_classes = []
                                    frame_height, frame_width = frame.shape[:2]
                                    
                                    for i in range(topk_values.size(1)):
                                        confidence = topk_values[0][i].item()
                                        class_id = topk_indices[0][i].item()
                                        
                                        if confidence > 0.2:  # 低置信度阈值以识别更多菜品
                                            valid_classes.append({
                                                'class_id': class_id,
                                                'class_name': self.class_names[class_id],
                                                'confidence': confidence
                                            })
                                    
                                    # 根据检测到的类别数量划分画面
                                    num_detections = len(valid_classes)
                                    print(f"检测到 {num_detections} 种食物: {[c['class_name'] for c in valid_classes]}")
                                    
                                    # 生成边界框
                                    bboxes = []
                                    if num_detections == 1:
                                        # 单个食物使用整个画面
                                        bboxes = [[0, 0, frame_width, frame_height]]
                                    elif num_detections == 2:
                                        # 两种食物 - 水平分割
                                        bboxes = [
                                            [0, 0, frame_width//2, frame_height],  # 左半部分
                                            [frame_width//2, 0, frame_width, frame_height]  # 右半部分
                                        ]
                                    elif num_detections == 3:
                                        # 三种食物 - 网格分割
                                        bboxes = [
                                            [0, 0, frame_width//2, frame_height//2],  # 左上
                                            [frame_width//2, 0, frame_width, frame_height//2],  # 右上
                                            [0, frame_height//2, frame_width, frame_height]  # 下部
                                        ]
                                    elif num_detections == 4:
                                        # 四种食物 - 四等分
                                        bboxes = [
                                            [0, 0, frame_width//2, frame_height//2],  # 左上
                                            [frame_width//2, 0, frame_width, frame_height//2],  # 右上
                                            [0, frame_height//2, frame_width//2, frame_height],  # 左下
                                            [frame_width//2, frame_height//2, frame_width, frame_height]  # 右下
                                        ]
                                    elif num_detections >= 5:
                                        # 五种或更多食物 - 自适应网格
                                        grid_size = int(np.ceil(np.sqrt(num_detections)))
                                        cell_width = frame_width // grid_size
                                        cell_height = frame_height // grid_size
                                        
                                        for i in range(num_detections):  # 支持任意数量的区域
                                            row = i // grid_size
                                            col = i % grid_size
                                            
                                            x1 = col * cell_width
                                            y1 = row * cell_height
                                            x2 = min((col + 1) * cell_width, frame_width)
                                            y2 = min((row + 1) * cell_height, frame_height)
                                            
                                            bboxes.append([x1, y1, x2, y2])
                                    
                                    # 为每个检测到的类创建完整的检测记录
                                    for i, food_class in enumerate(valid_classes):
                                        if i < len(bboxes):
                                            x1, y1, x2, y2 = bboxes[i]
                                            bbox_width = x2 - x1
                                            bbox_height = y2 - y1
                                            area = bbox_width * bbox_height
                                            
                                            # 创建检测信息字典
                                            detection = {
                                                'class_id': food_class['class_id'],
                                                'class_name': food_class['class_name'],
                                                'confidence': food_class['confidence'],
                                                'bbox': bboxes[i],
                                                'area': area,
                                                'weight': 0  # 默认重量为0，将在后续更新
                                            }
                                            detections.append(detection)
                                    
                                    # 更新上次检测的结果
                                    if detections:
                                        last_detections = detections
                                        
                                        # 如果有当前重量，立即应用重量分配
                                        if self.serial_processor:
                                            current_weight = self.serial_processor.get_current_weight()
                                            if current_weight > 0:
                                                self.last_weight = current_weight
                                                last_detections = self._update_detections_weight(last_detections, current_weight)
                                        
                                        # 发送检测结果
                                        self.detection_signal.emit(last_detections)
                                
                                # 手动清理GPU内存
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    
                            except Exception as e:
                                print(f"提取检测结果时出错: {e}")
                                import traceback
                                traceback.print_exc()
                                # 出错时仍然更新图像，避免界面冻结
                                self.change_pixmap_signal.emit(frame)
                        else:
                            # 非检测帧，仅更新图像显示
                            self.change_pixmap_signal.emit(frame)
                    
                    except Exception as e:
                        print(f"视频处理错误: {e}")
                        import traceback
                        traceback.print_exc()
                        time.sleep(0.1)  # 错误后短暂暂停，避免CPU过载
                
                else:  # 已暂停
                    time.sleep(0.1)  # 暂停时减少CPU使用
        
        except Exception as e:
            print(f"视频线程异常: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 循环结束后安全释放资源
            if cap is not None:
                cap.release()
            if self.serial_processor and not hasattr(self.parent(), 'serial_processor'):
                try:
                    self.serial_processor.stop_reading()
                except:
                    pass
    
    def _update_detections_weight(self, detections, total_weight):
        """更新检测结果与实际重量的关联
        
        Args:
            detections: 检测结果列表，每项包含食物信息字典
            total_weight: 当前测量到的总重量
        
        Returns:
            带有重量信息的检测结果列表
        """
        # 如果没有检测结果或总重量为0，直接返回原检测结果
        if not detections or total_weight <= 0:
            return detections
        
        # 统计所有检测对象的总面积和置信度总和
        total_area = 0
        total_confidence = 0
        
        for detection in detections:
            area = detection.get('area', 0)
            confidence = detection.get('confidence', 0)
            
            # 累加面积和置信度
            total_area += area
            total_confidence += confidence
        
        # 防止除以0错误
        if total_area == 0 or total_confidence == 0:
            # 均分重量
            equal_weight = round(total_weight / len(detections), 1) if len(detections) > 0 else 0
            for detection in detections:
                detection['weight'] = equal_weight
            return detections
        
        # 根据对象的面积和置信度综合计算权重，并分配重量
        for i, detection in enumerate(detections):
            area = detection.get('area', 0)
            confidence = detection.get('confidence', 0)
            
            # 使用面积占比和置信度综合加权计算
            # 面积权重：占比80%，置信度权重：占比20%
            area_ratio = area / total_area
            confidence_ratio = confidence / total_confidence
            
            # 综合权重
            combined_weight = 0.8 * area_ratio + 0.2 * confidence_ratio
            
            # 根据综合权重分配实际重量
            weight = round(total_weight * combined_weight, 1)
            
            # 打印调试信息
            print(f"食物 {i+1} - {detection.get('class_name', '')}: 面积={area}px², 面积比={area_ratio:.3f}, "
                  f"置信度={confidence:.3f}, 分配重量={weight}g")
            
            # 更新重量
            detection['weight'] = weight
        
        # 检查分配的总重量是否与测量重量接近
        allocated_weight = sum([det.get('weight', 0) for det in detections])
        if abs(allocated_weight - total_weight) > 1.0:  # 允许1克的误差
            print(f"警告: 分配的总重量 ({allocated_weight}g) 与测量重量 ({total_weight}g) 不匹配")
            
            # 调整重量确保总和等于测量值
            correction = total_weight / allocated_weight if allocated_weight > 0 else 0
            for detection in detections:
                detection['weight'] = round(detection.get('weight', 0) * correction, 1)
            
            print(f"调整后: 总重量 = {sum([det.get('weight', 0) for det in detections])}g")
        
        return detections
    
    def stop(self):
        self.running = False
        self.wait()
    
    def pause(self):
        self.paused = True
    
    def resume(self):
        self.paused = False
        
    def switch_camera(self, camera_id):
        self.camera_id = camera_id
        # 需要重启线程来切换相机
        self.running = False
        self.wait()
        self.running = True
        self.start()

# 模型重训练线程
class TrainingThread(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(object, list)
    
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path
    
    def run(self):
        try:
            self.progress_signal.emit(10, "正在加载数据...")
            
            # 加载数据
            from test import AspectRatioPreservingDataset, get_transforms
            transform = get_transforms()
            dataset = AspectRatioPreservingDataset(self.dataset_path, transform=transform)
            
            self.progress_signal.emit(30, "正在准备数据加载器...")
            
            # 创建数据加载器
            dataloaders, dataset_sizes = get_dataloaders(dataset)
            
            self.progress_signal.emit(40, "正在构建模型...")
            
            # 构建模型
            num_classes = len(dataset.classes)
            model = build_model(num_classes)
            
            self.progress_signal.emit(50, "开始训练...")
            
            # 训练模型
            import torch.nn as nn
            import torch.optim as optim
            from torch.optim import lr_scheduler
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            
            # 训练过程
            trained_model = train_model(model, dataloaders, dataset_sizes, criterion, 
                                        optimizer, exp_lr_scheduler, num_epochs=10)
            
            self.progress_signal.emit(90, "保存模型...")
            
            # 保存模型
            torch.save(trained_model.state_dict(), 'best_model.pth')
            
            self.progress_signal.emit(100, "训练完成!")
            
            # 发送完成信号
            self.finished_signal.emit(trained_model, dataset.classes)
            
        except Exception as e:
            import traceback
            print(f"训练错误: {e}")
            traceback.print_exc()
            self.progress_signal.emit(0, f"训练失败: {str(e)}")

# 大模型分析线程
class AnalysisThread(QThread):
    result_signal = pyqtSignal(str)
    
    def __init__(self, model_path, detections=None):
        super().__init__()
        self.model_path = model_path
        self.detections = detections or []
        
    def run(self):
        try:
            # 收集食物重量数据
            food_weights = {}
            total_foods = len(self.detections)
            
            print(f"正在分析{total_foods}种食物的营养摄入情况...")
            for detection in self.detections:
                class_name = detection.get('class_name', '')
                weight = detection.get('weight', 0)
                confidence = detection.get('confidence', 0)
                
                if class_name and confidence > 0.2:  # 与VideoThread使用相同阈值
                    if class_name in food_weights:
                        food_weights[class_name] += weight
                    else:
                        food_weights[class_name] = weight
            
            if not food_weights:
                self.result_signal.emit("未检测到有效食物或重量数据无效。")
                return
            
            # 打印检测到的食物信息用于调试    
            print(f"大模型将分析以下 {len(food_weights)} 种食物: {list(food_weights.keys())}")
            
            # 确保重量为正值
            food_weights = {k: max(0.1, v) for k, v in food_weights.items()}
            
            # 计算总营养成分
            total_calories = 0
            total_protein = 0
            total_fat = 0
            total_carbs = 0
            
            try:
                from nutrition_data import get_nutrition_info
                
                nutrition_details = []  # 用于收集详细的营养信息
                
                for food_name, weight in food_weights.items():
                    weight_ratio = weight / 100.0  # 转换为营养数据库的基准单位
                    nutrition = get_nutrition_info(food_name, weight_ratio)
                    
                    if nutrition:
                        total_calories += nutrition['热量(kcal)']
                        total_protein += nutrition['蛋白质(g)']
                        total_fat += nutrition['脂肪(g)']
                        total_carbs += nutrition['碳水化合物(g)']
                        
                        # 收集详细的营养信息用于大模型分析
                        nutrition_details.append({
                            "食物名称": food_name,
                            "重量(g)": weight,
                            "热量(kcal)": nutrition['热量(kcal)'],
                            "蛋白质(g)": nutrition['蛋白质(g)'],
                            "脂肪(g)": nutrition['脂肪(g)'],
                            "碳水化合物(g)": nutrition['碳水化合物(g)']
                        })
            except ImportError:
                self.result_signal.emit("无法加载营养数据模块，请检查nutrition_data.py文件是否存在。")
                return
            
            # 尝试加载大模型进行分析
            try:
                # 使用本地简单版本生成营养建议
                advice = self._generate_simple_advice(food_weights, total_calories, total_protein, total_fat, total_carbs)
                
                # 尝试使用大模型升级分析
                if load_phi4_model:
                    try:
                        print("正在使用大模型分析多菜品营养情况...")
                        
                        # 构建提示，包括每种食物的详细营养信息
                        foods_str = "\n".join([
                            f"- {detail['食物名称']}({detail['重量(g)']:.1f}g): 热量={detail['热量(kcal)']:.1f}千卡, 蛋白质={detail['蛋白质(g)']:.1f}g, 脂肪={detail['脂肪(g)']:.1f}g, 碳水={detail['碳水化合物(g)']:.1f}g"
                            for detail in nutrition_details
                        ])
                        
                        # 为大模型构建多菜品分析提示
                        prompt = f"""
我的当前餐食包含以下{len(nutrition_details)}种食物：
{foods_str}

总营养成分:
总热量: {total_calories:.1f}千卡
总蛋白质: {total_protein:.1f}克
总脂肪: {total_fat:.1f}克
总碳水化合物: {total_carbs:.1f}克

请分析这些食物的营养搭配是否合理，从健康角度给出评估和改进建议。
分析要点:
1. 这一餐的热量、蛋白质、脂肪和碳水化合物摄入是否均衡?
2. 这一餐可能缺乏哪些营养素?需要搭配什么食物补充?
3. 给出2-3条具体的改进建议，让这一餐更健康。
"""
                        # 打印提示用于调试
                        print(f"大模型提示：\n{prompt}")
                        
                        # 请求大模型分析
                        expert_advice = generate_nutrition_advice(self.model_path, prompt)
                        if expert_advice:
                            # 组合简单建议和大模型建议
                            advice = f"{advice}\n\n专家营养分析:\n{expert_advice}"
                            print("大模型分析完成")
                        else:
                            print("大模型分析未返回结果，使用简单建议")
                    except Exception as e:
                        print(f"大模型分析失败: {e}")
                
                self.result_signal.emit(advice)
            except Exception as e:
                import traceback
                error_msg = f"分析过程中出现错误: {e}\n{traceback.format_exc()}"
                print(error_msg)
                self.result_signal.emit(f"营养分析生成失败: {e}")
            
        except Exception as e:
            import traceback
            error_msg = f"分析过程中出现错误: {e}\n{traceback.format_exc()}"
            print(error_msg)
            self.result_signal.emit(f"营养分析生成失败: {e}")
    
    def _generate_simple_advice(self, food_weights, total_calories, total_protein, total_fat, total_carbs):
        """生成简单的营养建议"""
        # 计算总重量
        total_weight = sum(food_weights.values())
        
        # 基于中国营养学会推荐的膳食营养素参考摄入量
        # 这里使用成人推荐值的30%作为单餐参考
        meal_calories_ref = 2200 * 0.3  # 成人日均热量的30%
        meal_protein_ref = 60 * 0.3     # 成人日均蛋白质的30%
        meal_fat_ref = 60 * 0.3         # 成人日均脂肪的30%
        meal_carbs_ref = 300 * 0.3      # 成人日均碳水化合物的30%
        
        # 计算各营养素在餐食中的占比
        protein_ratio = total_protein * 4 / total_calories if total_calories > 0 else 0
        fat_ratio = total_fat * 9 / total_calories if total_calories > 0 else 0
        carbs_ratio = total_carbs * 4 / total_calories if total_calories > 0 else 0
        
        # 生成建议文本
        advice = f"当前餐食总重量: {total_weight:.1f}g\n"
        advice += f"营养成分分析:\n"
        advice += f"• 总热量: {total_calories:.1f}千卡（推荐单餐{meal_calories_ref:.0f}千卡）\n"
        advice += f"• 蛋白质: {total_protein:.1f}克（占总热量的{protein_ratio*100:.1f}%）\n"
        advice += f"• 脂肪: {total_fat:.1f}克（占总热量的{fat_ratio*100:.1f}%）\n"
        advice += f"• 碳水化合物: {total_carbs:.1f}克（占总热量的{carbs_ratio*100:.1f}%）\n\n"
        
        # 添加营养均衡评估
        advice += "营养均衡评估:\n"
        
        # 评估热量
        if total_calories < meal_calories_ref * 0.7:
            advice += "• 热量摄入偏低，可适当增加食物量。\n"
        elif total_calories > meal_calories_ref * 1.3:
            advice += "• 热量摄入偏高，建议减少高热量食物。\n"
        else:
            advice += "• 热量摄入适中。\n"
        
        # 评估蛋白质
        if protein_ratio < 0.1:  # 推荐蛋白质占总热量10-15%
            advice += "• 蛋白质摄入偏低，可增加优质蛋白如鱼肉、豆制品等。\n"
        elif protein_ratio > 0.2:
            advice += "• 蛋白质摄入较高，注意保持饮食多样性。\n"
        else:
            advice += "• 蛋白质摄入合理。\n"
        
        # 评估脂肪
        if fat_ratio < 0.2:  # 推荐脂肪占总热量20-30%
            advice += "• 脂肪摄入偏低，可适量添加健康油脂。\n"
        elif fat_ratio > 0.35:
            advice += "• 脂肪摄入偏高，建议减少油炸、高脂食物。\n"
        else:
            advice += "• 脂肪摄入合理。\n"
        
        # 评估碳水化合物
        if carbs_ratio < 0.5:  # 推荐碳水占总热量50-65%
            advice += "• 碳水化合物摄入偏低，可适当增加全谷物、薯类等。\n"
        elif carbs_ratio > 0.65:
            advice += "• 碳水化合物摄入偏高，建议适量控制主食摄入。\n"
        else:
            advice += "• 碳水化合物摄入合理。\n"
        
        # 针对具体食物的建议
        advice += "\n食物构成分析:\n"
        for food, weight in food_weights.items():
            percent = weight / total_weight * 100
            advice += f"• {food}: {weight:.1f}g（占总重量的{percent:.1f}%）\n"
        
        return advice

# 主窗口
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 设置异常处理器，避免程序崩溃
        import sys
        sys.excepthook = self._handle_exception
        
        # 设置窗口属性
        self.setWindowTitle('"脂"能少年焕新系统')
        self.showFullScreen()
        
        # 模型和类别初始化为None
        self.model = None
        self.class_names = None
        self.detections = []
        self.camera_id = 0
        
        # 初始化串口处理器
        try:
            self.serial_processor = CustomSerialDataProcessor(port='COM3')
            self.serial_processor.start_reading()
            self.serial_processor.toggle_manual_mode(False)  # 确保默认为自动模式
        except Exception as e:
            print(f"串口初始化错误: {e}")
            import traceback
            traceback.print_exc()
            # 创建一个后备的处理器并使用手动模式
            self.serial_processor = CustomSerialDataProcessor(port='')
            self.serial_processor.toggle_manual_mode(True)
        
        # 初始化UI
        self.init_ui()
        
        # 加载模型
        self.load_model()
        
        # 初始化分析线程
        self.analysis_thread = None
        
        # 设置定时器定期更新分析
        self.analysis_timer = QTimer(self)
        self.analysis_timer.timeout.connect(self.update_analysis)
        self.analysis_timer.start(30000)  # 每30秒更新一次
        
        # 设置自动GC定时器，避免内存泄漏
        self.gc_timer = QTimer(self)
        self.gc_timer.timeout.connect(self._perform_gc)
        self.gc_timer.start(60000)  # 每60秒执行一次垃圾回收
    
    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        """全局异常处理函数，避免程序崩溃"""
        import traceback
        # 打印异常信息到控制台
        print("发生未处理异常:")
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        # 显示错误对话框
        error_msg = f"程序发生错误: {exc_value}"
        self.show_error(error_msg)
        # 仍然传递给默认处理器
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    def _perform_gc(self):
        """执行垃圾回收和内存清理"""
        try:
            import gc
            # 强制执行垃圾回收
            collected = gc.collect()
            print(f"垃圾回收: 释放了 {collected} 个对象")
            
            # 清理PyTorch缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("清理了PyTorch CUDA缓存")
        except Exception as e:
            print(f"垃圾回收时发生错误: {e}")
    
    def init_ui(self):
        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f5fa;
                font-family: "Microsoft YaHei", "SimHei", sans-serif;
            }
            QLabel {
                color: #2c3e50;
                font-weight: 500;
            }
            QPushButton {
                background-color: #2980b9;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                min-height: 36px;
                box-shadow: 0 3px 5px rgba(0, 0, 0, 0.2);
            }
            QPushButton:hover {
                background-color: #3498db;
                box-shadow: 0 5px 10px rgba(0, 0, 0, 0.3);
            }
            QPushButton:pressed {
                background-color: #1c6ea4;
                box-shadow: 0 2px 3px rgba(0, 0, 0, 0.2);
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                box-shadow: none;
            }
            QTableWidget {
                background-color: white;
                alternate-background-color: #f5faff;
                border: 1px solid #d0e3f0;
                border-radius: 8px;
                font-size: 14px;
                gridline-color: #e1ebf2;
                padding: 5px;
                selection-background-color: #3498db;
            }
            QTableWidget::item {
                padding: 8px 5px;
                border-bottom: 1px solid #e1ebf2;
                color: #34495e;
            }
            QTableWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
            QHeaderView::section {
                background-color: #2980b9;
                color: white;
                padding: 10px 8px;
                border: none;
                font-weight: bold;
                border-right: 1px solid #3498db;
                border-bottom: 1px solid #2980b9;
            }
            QHeaderView::section:first {
                border-top-left-radius: 8px;
            }
            QHeaderView::section:last {
                border-top-right-radius: 8px;
                border-right: none;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #d0e3f0;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                color: #34495e;
                line-height: 1.5;
            }
            QComboBox, QLineEdit {
                border: 1px solid #d0e3f0;
                border-radius: 8px;
                padding: 8px 10px;
                background-color: white;
                font-size: 14px;
                min-height: 30px;
                color: #34495e;
            }
            QLineEdit:read-only {
                background-color: #f8f9fa;
                color: #2c3e50;
            }
            QGroupBox {
                border: 1px solid #d0e3f0;
                border-radius: 8px;
                margin-top: 16px;
                padding-top: 16px;
                font-weight: bold;
                color: #2980b9;
                background-color: rgba(255, 255, 255, 0.7);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 10px;
                background-color: #f0f5fa;
            }
            QCheckBox {
                spacing: 8px;
                font-size: 14px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid #d0e3f0;
            }
            QCheckBox::indicator:checked {
                background-color: #2980b9;
                border: 1px solid #2980b9;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f5fa;
                width: 12px;
                margin: 12px 0 12px 0;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #bdc3c7;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: #3498db;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
                height: 12px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)
        
        # 创建中央部件
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # 标题标签
        title_layout = QHBoxLayout()
        title_label = QLabel('"脂"能少年焕新系统', self)
        title_font = QFont("Microsoft YaHei", 26, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            color: #2980b9; 
            margin-bottom: 20px;
            padding: 10px;
            background-color: rgba(255, 255, 255, 0.7);
            border-radius: 15px;
            border-bottom: 3px solid #3498db;
        """)

        # 添加图标标签
        icon_label = QLabel()
        icon_pixmap = QPixmap(40, 40)
        icon_pixmap.fill(Qt.transparent)
        painter = QPainter(icon_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor("#3498db")))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, 40, 40)
        painter.setPen(QPen(QColor("white"), 2))
        painter.drawText(QRect(0, 0, 40, 40), Qt.AlignCenter, "脂")
        painter.end()
        icon_label.setPixmap(icon_pixmap)
        icon_label.setFixedSize(40, 40)

        # 添加版本和作者信息
        version_label = QLabel("V1.0")
        version_label.setStyleSheet("color: #7f8c8d; font-size: 12px;")
        version_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # 组合标题布局
        title_layout.addStretch(1)
        title_layout.addWidget(icon_label)
        title_layout.addWidget(title_label, 4)  # 占比更大
        title_layout.addWidget(version_label, 1)
        title_layout.addStretch(1)

        # 内容区域（视频+信息）
        content_layout = QHBoxLayout()
        
        # 视频区域
        self.video_label = QLabel(self)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: black;
            border: none;
            border-radius: 12px;
            padding: 2px;
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #2980b9, stop:1 #3498db
            );
        """)
        
        # 创建视频内部容器，应用边距效果
        video_container = QWidget()
        video_container_layout = QVBoxLayout(video_container)
        video_container_layout.setContentsMargins(6, 6, 6, 6)
        video_container_layout.addWidget(self.video_label)
        video_container.setStyleSheet("""
            background-color: transparent;
            border-radius: 12px;
        """)
        
        # 右侧信息区域
        right_layout = QVBoxLayout()
        
        # 表格区域
        table_header = QWidget()
        table_header_layout = QHBoxLayout(table_header)
        table_header_layout.setContentsMargins(0, 0, 0, 0)
        
        table_label = QLabel("食物营养信息", self)
        table_label.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        table_label.setStyleSheet("color: #2980b9;")
        table_label.setAlignment(Qt.AlignCenter)
        
        info_icon = QLabel()
        info_pixmap = QPixmap(24, 24)
        info_pixmap.fill(Qt.transparent)
        painter = QPainter(info_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor("#3498db")))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, 24, 24)
        painter.setPen(QPen(QColor("white"), 2))
        painter.drawText(QRect(0, 0, 24, 24), Qt.AlignCenter, "i")
        painter.end()
        info_icon.setPixmap(info_pixmap)
        info_icon.setFixedSize(24, 24)
        
        table_header_layout.addStretch(1)
        table_header_layout.addWidget(info_icon)
        table_header_layout.addWidget(table_label)
        table_header_layout.addStretch(1)
        
        # 初始化表格内容
        # 初始化为1行，后续会根据识别食物的数量动态调整
        self.nutrition_table = QTableWidget(3, 6)
        self.nutrition_table.setHorizontalHeaderLabels(["名称", "重量(g)", "热量(kcal)", "蛋白质(g)", "脂肪(g)", "碳水化合物(g)"])
        self.nutrition_table.verticalHeader().setVisible(False)
        self.nutrition_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.nutrition_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.nutrition_table.setAlternatingRowColors(True)
        self.nutrition_table.setMinimumHeight(250)  # 设置最小高度确保表格可见
        self.nutrition_table.setMaximumHeight(400)  # 设置最大高度避免表格过大
        self.nutrition_table.setShowGrid(True)  # 显示网格线
        self.nutrition_table.verticalHeader().setDefaultSectionSize(40)  # 设置默认行高
        self.nutrition_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)  # 固定行高
        self.nutrition_table.setStyleSheet(self.nutrition_table.styleSheet() + """
            QTableWidget {
                border: none;
                background-color: white;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
        """)
        
        # 创建表格容器，设置样式
        table_container = QWidget()
        table_container_layout = QVBoxLayout(table_container)
        table_container_layout.setContentsMargins(10, 10, 10, 10)
        table_container_layout.addWidget(self.nutrition_table)
        table_container.setStyleSheet("""
            background-color: white;
            border-radius: 10px;
            border: 1px solid #d0e3f0;
        """)
        
        # 初始化表格内容 - 一个空行
        for col in range(6):
            item = QTableWidgetItem("")
            self.nutrition_table.setItem(0, col, item)
        
        # 串口设置区域
        serial_group = QGroupBox("串口连接状态")
        serial_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                margin-top: 15px;
            }
        """)
        serial_layout = QHBoxLayout(serial_group)
        serial_layout.setSpacing(15)
        serial_layout.setContentsMargins(15, 25, 15, 15)
        
        # 添加串口状态图标
        self.port_status_icon = QLabel()
        self.updatePortStatusIcon(True)  # 初始为已连接状态
        self.port_status_icon.setFixedSize(24, 24)
        
        # 添加串口状态标签
        self.port_status_label = QLabel("已连接到串口: COM3")
        self.port_status_label.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 14px;")
        
        # 刷新串口按钮
        refresh_port_btn = QPushButton(" 刷新")
        refresh_port_btn.setIcon(self.getRefreshIcon())
        refresh_port_btn.setIconSize(QSize(16, 16))
        refresh_port_btn.setMaximumWidth(100)
        refresh_port_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 15px;
                font-size: 13px;
            }
        """)
        refresh_port_btn.clicked.connect(self.refresh_ports)
        
        # 添加当前重量显示
        weight_container = QWidget()
        weight_layout = QHBoxLayout(weight_container)
        weight_layout.setContentsMargins(0, 0, 0, 0)
        weight_layout.setSpacing(5)
        
        weight_icon = QLabel()
        weight_pixmap = QPixmap(20, 20)
        weight_pixmap.fill(Qt.transparent)
        painter = QPainter(weight_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor("#f39c12")))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, 20, 20)
        painter.setPen(QPen(QColor("white"), 1))
        painter.drawText(QRect(0, 0, 20, 20), Qt.AlignCenter, "g")
        painter.end()
        weight_icon.setPixmap(weight_pixmap)
        weight_icon.setFixedSize(20, 20)
        
        weight_label = QLabel("当前重量:")
        weight_label.setStyleSheet("font-weight: bold; color: #34495e;")
        
        self.manual_weight_input = QLineEdit()
        self.manual_weight_input.setReadOnly(True)  # 设为只读
        self.manual_weight_input.setMaximumWidth(80)
        self.manual_weight_input.setText("0.0")
        self.manual_weight_input.setAlignment(Qt.AlignCenter)
        self.manual_weight_input.setStyleSheet("""
            QLineEdit {
                font-weight: bold;
                font-size: 14px;
                color: #e67e22;
                background-color: #f9f9f9;
                border: 1px solid #f0f0f0;
            }
        """)
        
        weight_unit = QLabel("克")
        weight_unit.setStyleSheet("color: #7f8c8d;")
        
        weight_layout.addWidget(weight_icon)
        weight_layout.addWidget(weight_label)
        weight_layout.addWidget(self.manual_weight_input)
        weight_layout.addWidget(weight_unit)
        
        # 添加到串口设置布局
        serial_layout.addWidget(self.port_status_icon)
        serial_layout.addWidget(self.port_status_label, 1)
        serial_layout.addWidget(weight_container, 1)
        serial_layout.addWidget(refresh_port_btn)
        
        # 按钮区域
        buttons_container = QWidget()
        buttons_container.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.7);
            border-radius: 10px;
            border: 1px solid #d0e3f0;
            padding: 5px;
        """)
        buttons_layout = QVBoxLayout(buttons_container)
        buttons_layout.setContentsMargins(15, 15, 15, 15)
        buttons_layout.setSpacing(15)
        
        # 创建按钮标题
        buttons_title = QLabel("操作控制")
        buttons_title.setStyleSheet("color: #2980b9; font-weight: bold; font-size: 14px;")
        buttons_title.setAlignment(Qt.AlignCenter)
        buttons_layout.addWidget(buttons_title)
        
        # 第一行按钮
        buttons_row1 = QHBoxLayout()
        buttons_row1.setSpacing(15)
        
        self.start_button = QPushButton(" 开始识别")
        self.start_button.setIcon(self.getStartIcon())
        self.start_button.setIconSize(QSize(20, 20))
        self.start_button.setStyleSheet("""
            QPushButton {
                padding: 8px 15px;
                font-size: 15px;
                text-align: center;
                color: black;
                font-weight: bold;
            }
        """)
        
        self.stop_button = QPushButton(" 停止识别")
        self.stop_button.setIcon(self.getStopIcon())
        self.stop_button.setIconSize(QSize(20, 20))
        self.stop_button.setStyleSheet("""
            QPushButton {
                padding: 8px 15px;
                font-size: 15px;
                text-align: center;
                color: black;
                font-weight: bold;
            }
        """)
        
        self.switch_camera_button = QPushButton(" 切换摄像头")
        self.switch_camera_button.setIcon(self.getCameraIcon())
        self.switch_camera_button.setIconSize(QSize(20, 20))
        self.switch_camera_button.setStyleSheet("""
            QPushButton {
                padding: 8px 15px;
                font-size: 15px;
                text-align: center;
                color: black;
                font-weight: bold;
            }
        """)
        
        self.start_button.clicked.connect(self.start_recognition)
        self.stop_button.clicked.connect(self.stop_recognition)
        self.switch_camera_button.clicked.connect(self.switch_camera)
        
        self.stop_button.setEnabled(False)
        
        buttons_row1.addWidget(self.start_button)
        buttons_row1.addWidget(self.stop_button)
        buttons_row1.addWidget(self.switch_camera_button)
        
        # 第二行按钮
        buttons_row2 = QHBoxLayout()
        buttons_row2.setSpacing(15)
        
        self.retrain_button = QPushButton(" 重新训练模型")
        self.retrain_button.setIcon(self.getRetrainIcon())
        self.retrain_button.setIconSize(QSize(20, 20))
        self.retrain_button.setStyleSheet("""
            QPushButton {
                padding: 8px 15px;
                font-size: 15px;
                text-align: center;
                color: black;
                font-weight: bold;
            }
        """)
        
        self.exit_button = QPushButton(" 退出")
        self.exit_button.setIcon(self.getExitIcon())
        self.exit_button.setIconSize(QSize(20, 20))
        self.exit_button.setStyleSheet("""
            QPushButton {
                padding: 8px 15px;
                font-size: 15px;
                text-align: center;
                color: white;
                background-color: #e74c3c;
                font-weight: bold;
            }
        """)
        
        self.retrain_button.clicked.connect(self.retrain_model)
        self.exit_button.clicked.connect(self.close)
        
        buttons_row2.addWidget(self.retrain_button)
        buttons_row2.addWidget(self.exit_button)
        
        # 添加按钮行到按钮布局
        buttons_layout.addLayout(buttons_row1)
        buttons_layout.addLayout(buttons_row2)
        
        # 添加到右侧布局
        right_layout.addWidget(table_header)
        right_layout.addWidget(table_container)
        right_layout.addWidget(serial_group)
        right_layout.addSpacing(5)
        right_layout.addWidget(buttons_container)
        
        # 添加左右内容到水平布局
        content_layout.addWidget(video_container, 1)
        content_layout.addLayout(right_layout, 1)
        
        # 分析区域
        analysis_container = QWidget()
        analysis_container.setStyleSheet("""
            background-color: white;
            border-radius: 10px;
            border: 1px solid #d0e3f0;
        """)
        analysis_container_layout = QVBoxLayout(analysis_container)
        analysis_container_layout.setContentsMargins(15, 15, 15, 15)
        
        analysis_header = QWidget()
        analysis_header_layout = QHBoxLayout(analysis_header)
        analysis_header_layout.setContentsMargins(0, 0, 0, 5)
        
        analysis_icon = QLabel()
        analysis_pixmap = QPixmap(24, 24)
        analysis_pixmap.fill(Qt.transparent)
        painter = QPainter(analysis_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor("#8e44ad")))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, 24, 24)
        painter.setPen(QPen(QColor("white"), 2))
        painter.drawText(QRect(0, 0, 24, 24), Qt.AlignCenter, "营")
        painter.end()
        analysis_icon.setPixmap(analysis_pixmap)
        analysis_icon.setFixedSize(24, 24)
        
        analysis_label = QLabel("营养分析与建议", self)
        analysis_label.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        analysis_label.setStyleSheet("color: #8e44ad;")
        analysis_label.setAlignment(Qt.AlignCenter)
        
        analysis_header_layout.addStretch(1)
        analysis_header_layout.addWidget(analysis_icon)
        analysis_header_layout.addWidget(analysis_label)
        analysis_header_layout.addStretch(1)
        
        self.analysis_text = QTextEdit(self)
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setMinimumHeight(150)
        self.analysis_text.setPlaceholderText("等待食物识别结果进行营养分析...")
        self.analysis_text.setStyleSheet("""
            QTextEdit {
                background-color: #f9f9f9;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                line-height: 1.6;
            }
        """)
        
        analysis_container_layout.addWidget(analysis_header)
        analysis_container_layout.addWidget(self.analysis_text)
        
        # 将分析区域添加到主布局
        main_layout.addWidget(analysis_container, 1)  # 分配1份空间
        
        # 将所有组件添加到主布局
        main_layout.addLayout(title_layout)
        main_layout.addLayout(content_layout, 3)  # 分配3份空间
    
    def load_model(self):
        """加载预训练模型"""
        try:
            import pickle
            
            # 加载类别信息 - 检查多种可能的路径
            possible_paths = [
                'classes.pkl',  # 当前目录
                r"D:\TEST3\classes.pkl",  # 指定TEST3目录
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "classes.pkl")  # 打包后的资源目录
            ]
            
            classes_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    classes_path = path
                    break
            
            if not classes_path:
                self.show_error("无法找到类别文件: classes.pkl\n请确保文件位于当前目录、resources目录或D:\\TEST3目录下")
                return
            
            print(f"从路径加载类别文件: {classes_path}")
            with open(classes_path, 'rb') as f:
                self.class_names = pickle.load(f)
            print(f"加载类别成功，共{len(self.class_names)}个类别")
            
            # 加载模型
            self.model = get_model(self.class_names)
            print("模型加载成功!")
            
            # 初始化视频线程
            self.video_thread = VideoThread(self.model, self.class_names, self.camera_id, self)
            self.video_thread.change_pixmap_signal.connect(self.update_image)
            self.video_thread.detection_signal.connect(self.update_detection)
            
        except Exception as e:
            self.show_error(f"加载模型失败: {str(e)}")
    
    def update_image(self, cv_img):
        """更新视频画面"""
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """将OpenCV图像转换为Qt图像"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_format)
        return pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
    
    def update_detection(self, detections):
        """更新检测结果和表格数据"""
        self.detections = detections
        
        # 清空表格并强制刷新UI
        self.nutrition_table.clearContents()
        
        # 创建检测到的食物字典，合并相同类别
        food_dict = {}
        for detection in detections:
            class_name = detection.get('class_name', '')
            weight = detection.get('weight', 0)
            confidence = detection.get('confidence', 0)
            
            if class_name and confidence > 0.2:  # 降低置信度阈值与VideoThread保持一致
                if class_name in food_dict:
                    food_dict[class_name]['count'] += 1
                    food_dict[class_name]['weight'] += weight
                else:
                    food_dict[class_name] = {
                        'count': 1,
                        'weight': weight,
                        'confidence': confidence
                    }
        
        # 打印检测到的食物信息
        if food_dict:
            print(f"表格将显示 {len(food_dict)} 种食物: {list(food_dict.keys())}")
        
        # 先按照重量降序排序，让重量较大的食物显示在前面
        sorted_foods = sorted(food_dict.items(), key=lambda x: x[1]['weight'], reverse=True)
        
        # 根据检测到的食物数量动态调整表格行数
        food_count = len(sorted_foods)
        total_rows_needed = food_count + 1  # 食物行数加上总计行
        
        # 确保表格有足够的行数
        current_rows = self.nutrition_table.rowCount()
        if total_rows_needed > current_rows:
            self.nutrition_table.setRowCount(total_rows_needed)
        
        # 更新表格
        row = 0
        total_calories = 0
        total_protein = 0
        total_fat = 0
        total_carbs = 0
        total_weight = 0  # 添加总重量统计
        
        # 定义不同行的背景色
        row_colors = [
            QColor(240, 248, 255, 150),  # 淡蓝色
            QColor(240, 255, 240, 150)   # 淡绿色
        ]
        
        # 定义列的颜色和样式
        col_styles = [
            {"color": QColor("#2c3e50"), "align": Qt.AlignLeft | Qt.AlignVCenter, "font_weight": QFont.Bold},    # 名称列
            {"color": QColor("#27ae60"), "align": Qt.AlignCenter, "font_weight": QFont.Normal},  # 重量列
            {"color": QColor("#e74c3c"), "align": Qt.AlignCenter, "font_weight": QFont.Bold},    # 热量列
            {"color": QColor("#3498db"), "align": Qt.AlignCenter, "font_weight": QFont.Normal},  # 蛋白质列
            {"color": QColor("#e67e22"), "align": Qt.AlignCenter, "font_weight": QFont.Normal},  # 脂肪列
            {"color": QColor("#9b59b6"), "align": Qt.AlignCenter, "font_weight": QFont.Normal}   # 碳水列
        ]
        
        for food_name, info in sorted_foods:
            # 名称列
            count = info['count']
            name_text = f"{food_name} x{count}" if count > 1 else food_name
            
            # 重量列 - 显示克数
            weight = info['weight']
            if weight <= 0:
                print(f"警告：食物 {food_name} 的重量为零或负值: {weight}g，设置为最小值0.1g")
                weight = 0.1  # 设置一个默认最小值，防止营养计算出错
            
            # 累计总重量
            total_weight += weight
                
            # 计算营养数据 - 单位已经是克，只需计算相对于100克的比例
            try:
                from nutrition_data import NUTRITION_DATA, get_nutrition_info
                
                # weight是克数，除以100计算比例
                weight_ratio = weight / 100.0
                
                # 确保比例大于0
                if weight_ratio <= 0:
                    weight_ratio = 0.001  # 非常小的数
                
                print(f"计算 {food_name} 的营养成分，重量: {weight}g, 比例: {weight_ratio:.4f}")
                nutrition = get_nutrition_info(food_name, weight_ratio)
                
                if nutrition:
                    # 准备表格数据
                    calories = nutrition['热量(kcal)']
                    protein = nutrition['蛋白质(g)']
                    fat = nutrition['脂肪(g)']
                    carbs = nutrition['碳水化合物(g)']
                    
                    print(f"食物 {food_name} ({weight}g) 的营养成分: 热量={calories:.1f}kcal, 蛋白质={protein:.1f}g, 脂肪={fat:.1f}g, 碳水={carbs:.1f}g")
                    
                    # 累计总营养成分
                    total_calories += calories
                    total_protein += protein
                    total_fat += fat
                    total_carbs += carbs
                    
                    # 获取当前行背景色
                    row_bg_color = row_colors[row % 2]
                    
                    # 创建数据数组
                    data = [
                        name_text,
                        f"{weight:.1f}",
                        f"{calories:.1f}",
                        f"{protein:.1f}",
                        f"{fat:.1f}",
                        f"{carbs:.1f}"
                    ]
                    
                    # 应用样式到所有单元格
                    for col, value in enumerate(data):
                        item = QTableWidgetItem(value)
                        style = col_styles[col]
                        
                        # 设置文本颜色
                        item.setForeground(style["color"])
                        
                        # 设置对齐方式
                        item.setTextAlignment(style["align"])
                        
                        # 设置字体
                        font = QFont()
                        font.setPointSize(10)
                        font.setWeight(style["font_weight"])
                        item.setFont(font)
                        
                        # 设置背景色
                        item.setBackground(row_bg_color)
                        
                        # 添加到表格
                        self.nutrition_table.setItem(row, col, item)
                else:
                    # 没有营养数据的情况
                    print(f"警告：没有找到食物 {food_name} 的营养数据")
                    name_item = QTableWidgetItem(name_text)
                    name_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                    name_item.setForeground(QColor("#2c3e50"))
                    
                    weight_item = QTableWidgetItem(f"{weight:.1f}")
                    weight_item.setTextAlignment(Qt.AlignCenter)
                    weight_item.setForeground(QColor("#27ae60"))
                    
                    # 设置背景色
                    row_bg_color = row_colors[row % 2]
                    name_item.setBackground(row_bg_color)
                    weight_item.setBackground(row_bg_color)
                    
                    self.nutrition_table.setItem(row, 0, name_item)
                    self.nutrition_table.setItem(row, 1, weight_item)
                    
                    for col in range(2, 6):
                        item = QTableWidgetItem("N/A")
                        item.setTextAlignment(Qt.AlignCenter)
                        item.setForeground(QColor("#95a5a6"))
                        item.setBackground(row_bg_color)
                        self.nutrition_table.setItem(row, col, item)
                
                # A移动到下一行
                row += 1
            except Exception as e:
                print(f"处理食物数据时出错: {e}")
        
        # 如果有检测到食物，添加合计行
        if row > 0:
            # 设置总计行样式
            total_row_bg = QColor(244, 236, 247)  # 淡紫色背景
            
            # 合计行
            total_item = QTableWidgetItem("总计")
            total_item.setTextAlignment(Qt.AlignCenter)
            total_item.setBackground(total_row_bg)
            font = QFont()
            font.setPointSize(10)
            font.setBold(True)
            total_item.setFont(font)
            total_item.setForeground(QColor("#2c3e50"))
            self.nutrition_table.setItem(row, 0, total_item)
            
            # 总重量列
            weight_total = QTableWidgetItem(f"{total_weight:.1f}")
            weight_total.setTextAlignment(Qt.AlignCenter)
            weight_total.setBackground(total_row_bg)
            weight_total.setFont(font)
            weight_total.setForeground(QColor("#27ae60"))
            self.nutrition_table.setItem(row, 1, weight_total)
            
            # 合计值
            cal_total = QTableWidgetItem(f"{total_calories:.1f}")
            cal_total.setTextAlignment(Qt.AlignCenter)
            cal_total.setBackground(total_row_bg)
            cal_total.setFont(font)
            cal_total.setForeground(QColor("#e74c3c"))
            self.nutrition_table.setItem(row, 2, cal_total)
            
            pro_total = QTableWidgetItem(f"{total_protein:.1f}")
            pro_total.setTextAlignment(Qt.AlignCenter)
            pro_total.setBackground(total_row_bg)
            pro_total.setFont(font)
            pro_total.setForeground(QColor("#3498db"))
            self.nutrition_table.setItem(row, 3, pro_total)
            
            fat_total = QTableWidgetItem(f"{total_fat:.1f}")
            fat_total.setTextAlignment(Qt.AlignCenter)
            fat_total.setBackground(total_row_bg)
            fat_total.setFont(font)
            fat_total.setForeground(QColor("#e67e22"))
            self.nutrition_table.setItem(row, 4, fat_total)
            
            carb_total = QTableWidgetItem(f"{total_carbs:.1f}")
            carb_total.setTextAlignment(Qt.AlignCenter)
            carb_total.setBackground(total_row_bg)
            carb_total.setFont(font)
            carb_total.setForeground(QColor("#9b59b6"))
            self.nutrition_table.setItem(row, 5, carb_total)
            
            # 更新表格总行数
            self.nutrition_table.setRowCount(row + 1)
        
        # 如果没有检测到食物，显示提示信息
        if row == 0:
            self.nutrition_table.setRowCount(1)
            empty_item = QTableWidgetItem("等待食物识别结果...")
            empty_item.setTextAlignment(Qt.AlignCenter)
            empty_item.setForeground(QColor("#7f8c8d"))
            font = QFont()
            font.setItalic(True)
            empty_item.setFont(font)
            self.nutrition_table.setItem(0, 0, empty_item)
            self.nutrition_table.setSpan(0, 0, 1, 6)  # 合并单元格

        # 更新分析线程
        self.update_analysis()
    
    def update_analysis(self):
        """更新营养建议"""
        if not self.detections:
            return
        
        # 打印检测到的食物种类
        food_types = set(d.get('class_name', '') for d in self.detections if d.get('confidence', 0) > 0.2)
        print(f"更新分析: 检测到 {len(food_types)} 种食物: {food_types}")
        
        # 如果分析线程正在运行，先停止它
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.terminate()
            self.analysis_thread.wait()
        
        # 检查多种可能的模型路径
        possible_model_paths = [
            r"D:\TEST3\DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",  # 指定TEST3目录
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "models", "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf")  # 打包后的资源目录
        ]
        
        model_path = None
        for path in possible_model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        # 检查模型文件是否存在
        if not model_path:
            self.analysis_text.setText("大模型文件不存在，无法进行营养分析。\n请确保DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf文件存在于D:\\TEST3目录下。")
            return
        
        # 如果是多菜品情况，在界面上显示提示
        if len(food_types) > 1:
            self.analysis_text.setText(f"正在分析 {len(food_types)} 种食物的营养搭配，请稍候...")
        else:
            self.analysis_text.setText("正在加载大模型，请稍候...")
            
        # 启动分析线程，传递所有检测到的食物数据
        self.analysis_thread = AnalysisThread(model_path, self.detections)
        self.analysis_thread.result_signal.connect(self.set_analysis_text)
        self.analysis_thread.start()
    
    def set_analysis_text(self, text):
        """设置分析文本，并添加美化格式"""
        
        # 格式化文本，添加颜色和样式
        formatted_text = text
        
        # 格式化标题
        formatted_text = re.sub(r"(【.*?】)", r"<span style='color:#8e44ad; font-weight:bold; font-size:16px;'>\1</span>", formatted_text)
        
        # 格式化重要数据
        formatted_text = re.sub(r"(\d+\.?\d*\s*千?卡)", r"<span style='color:#e74c3c; font-weight:bold;'>\1</span>", formatted_text)
        formatted_text = re.sub(r"(\d+\.?\d*\s*克?\s*蛋白质)", r"<span style='color:#3498db; font-weight:bold;'>\1</span>", formatted_text)
        formatted_text = re.sub(r"(\d+\.?\d*\s*克?\s*脂肪)", r"<span style='color:#e67e22; font-weight:bold;'>\1</span>", formatted_text)
        formatted_text = re.sub(r"(\d+\.?\d*\s*克?\s*碳水)", r"<span style='color:#9b59b6; font-weight:bold;'>\1</span>", formatted_text)
        
        # 添加段落格式
        formatted_text = formatted_text.replace("\n\n", "</p><p>")
        formatted_text = formatted_text.replace("\n", "<br>")
        formatted_text = f"<p style='line-height:1.6;'>{formatted_text}</p>"
        
        # 添加统一的文档样式
        formatted_text = f"""
        <div style='font-family:"Microsoft YaHei", "SimHei", sans-serif; color:#34495e; font-size:14px;'>
            {formatted_text}
        </div>
        """
        
        # 设置为富文本
        self.analysis_text.setHtml(formatted_text)
    
    def start_recognition(self):
        """开始识别"""
        if not self.model or not self.class_names:
            self.show_error("模型未加载，无法开始识别")
            return
            
        if not hasattr(self, 'video_thread') or not self.video_thread.isRunning():
            self.video_thread = VideoThread(self.model, self.class_names, self.camera_id, self)
            self.video_thread.change_pixmap_signal.connect(self.update_image)
            self.video_thread.detection_signal.connect(self.update_detection)
            self.video_thread.weight_update_signal.connect(self.update_weight_display)
            self.video_thread.start()
        else:
            self.video_thread.resume()
            
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
    
    def stop_recognition(self):
        """停止识别"""
        if hasattr(self, 'video_thread') and self.video_thread.isRunning():
            self.video_thread.pause()
            
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
    
    def switch_camera(self):
        """切换摄像头"""
        # 查找可用摄像头
        max_cameras = 3  # 最多检查3个摄像头
        available_cameras = []
        
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        
        if not available_cameras:
            self.show_error("未找到可用摄像头")
            return
            
        # 切换到下一个可用摄像头
        current_index = available_cameras.index(self.camera_id) if self.camera_id in available_cameras else -1
        next_index = (current_index + 1) % len(available_cameras)
        self.camera_id = available_cameras[next_index]
        
        # 如果视频线程正在运行，更新摄像头
        if hasattr(self, 'video_thread') and self.video_thread.isRunning():
            self.video_thread.switch_camera(self.camera_id)
        
        # 显示提示
        self.analysis_text.setText(f"已切换到摄像头 {self.camera_id}")
    
    def retrain_model(self):
        """重新训练模型"""
        # 选择数据集文件夹
        dataset_path = QFileDialog.getExistingDirectory(self, "选择训练数据集文件夹")
        if not dataset_path:
            return
            
        # 创建训练对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("模型训练")
        dialog.setMinimumWidth(400)
        
        # 对话框布局
        layout = QVBoxLayout(dialog)
        
        # 进度条和状态标签
        status_label = QLabel("准备训练...", dialog)
        progress_bar = QProgressBar(dialog)
        progress_bar.setRange(0, 100)
        
        layout.addWidget(status_label)
        layout.addWidget(progress_bar)
        
        # 创建训练线程
        training_thread = TrainingThread(dataset_path)
        
        # 连接信号
        def update_progress(value, message):
            progress_bar.setValue(value)
            status_label.setText(message)
        
        def training_finished(model, classes):
            self.model = model
            self.class_names = classes
            dialog.accept()
            
            # 保存类别信息
            import pickle
            with open('classes.pkl', 'wb') as f:
                pickle.dump(classes, f)
                
            # 提示用户
            QMessageBox.information(self, "训练完成", "模型训练已完成并已加载！")
        
        training_thread.progress_signal.connect(update_progress)
        training_thread.finished_signal.connect(training_finished)
        
        # 启动训练
        training_thread.start()
        
        # 显示对话框
        dialog.exec_()
    
    def show_error(self, message):
        """显示错误消息"""
        QMessageBox.critical(self, "错误", message)
    
    def closeEvent(self, event):
        """窗口关闭事件，确保所有线程安全停止"""
        try:
            # 停止视频线程
            if hasattr(self, 'video_thread') and self.video_thread.isRunning():
                print("正在停止视频线程...")
                self.video_thread.running = False  # 直接设置运行标志
                self.video_thread.stop()
                # 等待线程真正结束
                timeout = 0
                while self.video_thread.isRunning() and timeout < 30:
                    QApplication.processEvents()  # 处理其他事件
                    time.sleep(0.1)
                    timeout += 1
            
            # 停止分析线程
            if self.analysis_thread and self.analysis_thread.isRunning():
                print("正在停止分析线程...")
                self.analysis_thread.terminate()
                self.analysis_thread.wait(1000)  # 最多等待1秒
            
            # 停止串口处理器
            if hasattr(self, 'serial_processor'):
                print("正在关闭串口...")
                self.serial_processor.stop_reading()
            
            # 清除定时器
            if hasattr(self, 'analysis_timer') and self.analysis_timer.isActive():
                self.analysis_timer.stop()
            
            print("所有资源已释放，程序即将退出")
            
        except Exception as e:
            print(f"关闭窗口时出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 释放显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        event.accept()

    # 串口相关方法
    def refresh_ports(self):
        """刷新可用串口列表并自动连接到COM3"""
        self.serial_processor.update_available_ports()
        
        # 自动尝试连接到COM3
        success = self.serial_processor.connect_to_port('COM3')
        if success:
            self.port_status_label.setText("已连接到串口: COM3")
            self.port_status_label.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 14px;")
            self.analysis_text.setText("已自动连接到串口: COM3")
            self.updatePortStatusIcon(True)
        else:
            self.port_status_label.setText("无法连接到串口: COM3，使用模拟模式")
            self.port_status_label.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 14px;")
            self.analysis_text.setText("无法连接到串口COM3，已切换到模拟模式")
            self.updatePortStatusIcon(False)
    
    def connect_port(self):
        """连接到COM3串口"""
        port = 'COM3'
        success = self.serial_processor.connect_to_port(port)
        if success:
            self.port_status_label.setText(f"已连接到串口: {port}")
            self.port_status_label.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 14px;")
            self.analysis_text.setText(f"已连接到串口: {port}")
            self.updatePortStatusIcon(True)
        else:
            self.port_status_label.setText(f"无法连接到串口: {port}，使用模拟模式")
            self.port_status_label.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 14px;")
            self.analysis_text.setText(f"无法连接到串口: {port}，已切换到模拟模式")
            self.updatePortStatusIcon(False)
    
    def set_manual_weight(self, weight=100.0):
        """设置手动重量（仅用于后台调用）"""
        try:
            weight_val = float(weight)
            self.serial_processor.set_manual_weight(weight_val)
            print(f"设置重量: {weight_val}g")
        except (ValueError, TypeError) as e:
            print(f"设置重量时出错: {e}")
    
    def toggle_manual_mode(self, state=False):
        """切换手动/自动模式（仅用于后台调用）"""
        self.serial_processor.toggle_manual_mode(False)  # 始终使用自动模式
    
    def update_weight_display(self, weight):
        """更新当前重量显示"""
        try:
            # 确保weight是有效的数字
            weight_value = float(weight)
            
            # 处理负重量或零重量
            if weight_value <= 0:
                print(f"接收到无效重量值: {weight_value}g, 设置为最小值0.1g")
                weight_value = 0.1
                
            # 更新只读显示
            self.manual_weight_input.setText(f"{weight_value:.1f}")
            
            # 更新表格中相关食物的重量 - 如果有新识别的食物
            if hasattr(self, 'detections') and self.detections:
                # 通知视频线程更新重量信息
                if hasattr(self, 'video_thread') and self.video_thread.isRunning():
                    self.video_thread.last_weight_update = 0  # 强制在下次循环更新
        except (ValueError, TypeError) as e:
            print(f"更新重量显示时出错: {e}, 值: {weight}")

    def updatePortStatusIcon(self, connected=True):
        """更新串口状态图标"""
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if connected:
            # 已连接 - 绿色图标
            painter.setBrush(QBrush(QColor("#27ae60")))
        else:
            # 未连接 - 红色图标
            painter.setBrush(QBrush(QColor("#e74c3c")))
            
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(2, 2, 20, 20)
        
        # 绘制连接状态图案
        if connected:
            # 绘制打钩标记
            painter.setPen(QPen(QColor("white"), 2))
            painter.drawLine(6, 12, 10, 16)
            painter.drawLine(10, 16, 18, 8)
        else:
            # 绘制X标记
            painter.setPen(QPen(QColor("white"), 2))
            painter.drawLine(8, 8, 16, 16)
            painter.drawLine(16, 8, 8, 16)
            
        painter.end()
        
        if hasattr(self, 'port_status_icon'):
            self.port_status_icon.setPixmap(pixmap)
            
    def getStartIcon(self):
        """生成开始按钮图标"""
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制三角形
        painter.setBrush(QBrush(QColor("white")))
        painter.setPen(Qt.NoPen)
        points = [QPoint(5, 3), QPoint(17, 10), QPoint(5, 17)]
        painter.drawPolygon(points)
        
        painter.end()
        return QIcon(pixmap)
        
    def getStopIcon(self):
        """生成停止按钮图标"""
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制方块
        painter.setBrush(QBrush(QColor("white")))
        painter.setPen(Qt.NoPen)
        painter.drawRect(5, 5, 10, 10)
        
        painter.end()
        return QIcon(pixmap)
        
    def getCameraIcon(self):
        """生成摄像头按钮图标"""
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制摄像头形状
        painter.setBrush(QBrush(QColor("white")))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(3, 5, 14, 10, 2, 2)
        painter.drawPolygon([QPoint(17, 7), QPoint(17, 13), QPoint(20, 10)])
        
        # 绘制镜头
        painter.setBrush(QBrush(QColor("#2980b9")))
        painter.drawEllipse(7, 7, 6, 6)
        painter.setBrush(QBrush(QColor("white")))
        painter.drawEllipse(9, 9, 2, 2)
        
        painter.end()
        return QIcon(pixmap)
        
    def getRetrainIcon(self):
        """生成重新训练按钮图标"""
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制循环箭头
        painter.setPen(QPen(QColor("white"), 2))
        painter.drawArc(3, 3, 14, 14, 30 * 16, 300 * 16)
        
        # 绘制箭头头部
        points = [QPoint(16, 4), QPoint(19, 7), QPoint(13, 8)]
        painter.setBrush(QBrush(QColor("white")))
        painter.drawPolygon(points)
        
        painter.end()
        return QIcon(pixmap)
        
    def getExitIcon(self):
        """生成退出按钮图标"""
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制退出图标
        painter.setPen(QPen(QColor("white"), 2))
        # 绘制矩形
        painter.drawRoundedRect(3, 3, 14, 14, 2, 2)
        # 绘制箭头
        painter.drawLine(15, 10, 10, 10)
        points = [QPoint(12, 7), QPoint(15, 10), QPoint(12, 13)]
        painter.setBrush(QBrush(QColor("white")))
        painter.drawPolygon(points)
        
        painter.end()
        return QIcon(pixmap)
        
    def getRefreshIcon(self):
        """生成刷新按钮图标"""
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制循环箭头
        painter.setPen(QPen(QColor("white"), 1.5))
        painter.drawArc(2, 2, 12, 12, 30 * 16, 300 * 16)
        
        # 绘制箭头头部
        points = [QPoint(13, 3), QPoint(15, 5), QPoint(11, 6)]
        painter.setBrush(QBrush(QColor("white")))
        painter.drawPolygon(points)
        
        painter.end()
        return QIcon(pixmap)

# 主函数
def main():
    """程序主入口"""
    try:
        import sys
        app = QApplication(sys.argv)
        
        # 设置应用程序信息
        app.setApplicationName('"脂"能少年焕新系统')
        app.setOrganizationName('脂能少年团队')
        
        # 设置全局样式表
        app.setStyle('Fusion')
        
        # 创建并显示主窗口
        window = MainWindow()
        window.show()
        
        # 应用程序事件循环
        sys.exit(app.exec_())
    except Exception as e:
        print(f"程序启动失败: {e}")
        import traceback
        traceback.print_exc()
        # 显示错误对话框
        from PyQt5.QtWidgets import QMessageBox
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle('启动错误')
        error_dialog.setText(f'程序启动时发生错误:\n{e}')
        error_dialog.setDetailedText(traceback.format_exc())
        error_dialog.exec_()
        sys.exit(1)

if __name__ == "__main__":
    main()
