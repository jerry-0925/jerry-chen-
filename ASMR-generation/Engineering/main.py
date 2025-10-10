import sys
import asyncio
import threading
import numpy as np
import pyqtgraph as pg
import mediapipe as mp
import serial
import depthai as dai
import cv2
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QTimer, Qt
from bleak import BleakClient

# ===== BLE配置区域 =====
BLE_DEVICE_MAC = "C0:00:00:00:00:30"  # 设备MAC地址
BLE_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"   # 接收数据的特征UUID (TX)
BLE_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"   # 发送数据的特征UUID (RX)
BLE_START_COMMAND = b"\x11"  # 启动命令: 0x11

# ===== 肌电配置区域 =====
EMG_SERIAL_PORT = 'COM10'  # 肌电设备串口
EMG_BAUDRATE = 115200  # 串口波特率
EMG_CHANNELS = 6  # 肌电通道数

# ===== 面部表情配置 =====
FACIAL_DETECTION_ENABLED = True  # 是否启用面部表情检测
USE_DEPTHAI_CAMERA = True  # 使用DepthAI相机代替普通摄像头


class DepthAICamera:
    def __init__(self):
        # 创建DepthAI管道
        self.pipeline = dai.Pipeline()

        # 定义源和输出
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.xoutRgb = self.pipeline.create(dai.node.XLinkOut)

        self.xoutRgb.setStreamName("rgb")

        # 配置相机属性
        self.camRgb.setPreviewSize(1280, 720)
        self.camRgb.setInterleaved(False)
        self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        # 连接节点
        self.camRgb.preview.link(self.xoutRgb.input)

        # 连接到设备并启动管道
        self.device = dai.Device(self.pipeline)
        print('Connected cameras:', self.device.getConnectedCameraFeatures())
        print('Usb speed:', self.device.getUsbSpeed().name)
        if self.device.getBootloaderVersion() is not None:
            print('Bootloader version:', self.device.getBootloaderVersion())
        print('Device name:', self.device.getDeviceName(), 'Product name:', self.device.getProductName())

        # 获取输出队列
        self.qRgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        # 当前帧
        self.current_frame = None
        self.is_running = True

        # 启动相机线程
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()

    def _capture_loop(self):
        """相机捕获循环"""
        while self.is_running:
            try:
                inRgb = self.qRgb.get()  # 获取帧数据
                self.current_frame = inRgb.getCvFrame()
            except Exception as e:
                print(f"DepthAI相机错误: {e}")

    def get_frame(self):
        """获取当前帧"""
        return self.current_frame

    def stop(self):
        """停止相机"""
        self.is_running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.device.close()


class HealthMonitorApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # 初始化UI
        self.init_ui()

        # 初始化设备
        self.init_devices()

        # 启动BLE连接
        self.start_ble_connection()

        # 启动面部表情检测
        if FACIAL_DETECTION_ENABLED:
            self.start_facial_detection()

        # 启动肌电数据采集
        self.init_emg_serial()

        # 启动数据更新定时器
        self.data_update_timer = QTimer()
        self.data_update_timer.timeout.connect(self.update_display)
        self.data_update_timer.start(100)  # 每100ms更新一次显示

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("综合健康监测系统 (DepthAI相机)")
        self.setGeometry(100, 100, 1920, 1080)

        # 创建主布局
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # 创建健康数据面板
        health_data_layout = QtWidgets.QHBoxLayout()

        # 健康指标显示
        health_metrics_widget = QtWidgets.QGroupBox("健康指标")
        health_metrics_layout = QtWidgets.QVBoxLayout()

        # 心率显示
        self.heart_rate_label = QtWidgets.QLabel("心率: -- BPM")
        self.heart_rate_label.setStyleSheet("font-size: 24pt; color: #FF0000;")
        health_metrics_layout.addWidget(self.heart_rate_label)

        # 呼吸率显示
        self.respiration_rate_label = QtWidgets.QLabel("呼吸率: -- BPM")
        self.respiration_rate_label.setStyleSheet("font-size: 24pt; color: #00AA00;")
        health_metrics_layout.addWidget(self.respiration_rate_label)

        # 表情显示
        self.expression_label = QtWidgets.QLabel("面部表情: --")
        self.expression_label.setStyleSheet("font-size: 24pt; color: #0000FF;")
        health_metrics_layout.addWidget(self.expression_label)

        health_metrics_widget.setLayout(health_metrics_layout)
        health_data_layout.addWidget(health_metrics_widget)

        # 摄像头显示
        if FACIAL_DETECTION_ENABLED:
            camera_group = QtWidgets.QGroupBox("DepthAI相机 - 面部表情检测")
            camera_layout = QtWidgets.QVBoxLayout()
            self.camera_label = QtWidgets.QLabel()
            self.camera_label.setFixedSize(640, 480)
            camera_layout.addWidget(self.camera_label)
            camera_group.setLayout(camera_layout)
            health_data_layout.addWidget(camera_group)

        main_layout.addLayout(health_data_layout)

        # 创建肌电图
        emg_group = QtWidgets.QGroupBox("肌电信号")
        emg_layout = QtWidgets.QVBoxLayout()
        self.emg_widget = pg.PlotWidget()
        self.emg_widget.setLabel('left', '电压值')
        self.emg_widget.setLabel('bottom', '时间')
        self.emg_widget.setYRange(0, 3500)

        # 初始化肌电曲线
        self.x_data = np.arange(500)
        self.emg_curves = []
        colors = ['w', 'g', 'b', 'r', 'y', 'c']

        for i in range(EMG_CHANNELS):
            curve = self.emg_widget.plot(pen=colors[i])
            self.emg_curves.append(curve)

        emg_layout.addWidget(self.emg_widget)
        emg_group.setLayout(emg_layout)
        main_layout.addWidget(emg_group)

        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("系统初始化中...")

    def init_devices(self):
        """初始化设备"""
        # 初始化BLE数据
        self.heart_rate = 0
        self.respiration_rate = 0
        self.ble_connected = False

        # 初始化肌电数据
        self.emg_serial = None
        self.emg_data = np.zeros((EMG_CHANNELS, 500))

        # 初始化面部表情
        self.facial_expression = "未检测"
        self.camera_frame = None
        self.facial_detector = None
        self.depthai_camera = None

    def start_ble_connection(self):
        """启动BLE连接线程"""
        self.status_bar.showMessage("正在连接BLE设备...")

        # 创建并启动BLE线程
        self.ble_thread = threading.Thread(target=self.run_ble_client)
        self.ble_thread.daemon = True
        self.ble_thread.start()

    def run_ble_client(self):
        """运行BLE客户端"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.ble_main())

    async def ble_main(self):
        """BLE主函数"""
        try:
            async with BleakClient(BLE_DEVICE_MAC) as client:
                # 检查连接状态
                if not client.is_connected:
                    self.update_status("BLE连接失败")
                    return

                self.ble_connected = True
                self.update_status("BLE已连接")

                # 发送启动命令
                await client.write_gatt_char(BLE_RX_UUID, BLE_START_COMMAND)

                # 定义数据处理器
                def data_handler(sender, data):
                    self.process_ble_data(data)

                # 启动数据接收
                await client.start_notify(BLE_TX_UUID, data_handler)

                # 保持运行直到手动停止
                while client.is_connected:
                    await asyncio.sleep(0.1)

        except Exception as e:
            self.update_status(f"BLE错误: {str(e)}")
        finally:
            self.ble_connected = False
            self.update_status("BLE已断开")

    def process_ble_data(self, data):
        """处理BLE数据"""
        # 解析数据包
        heart_rate, respiration_rate = self.parse_ble_packet(data)

        # 更新数据
        if heart_rate is not None and respiration_rate is not None:
            self.heart_rate = heart_rate
            self.respiration_rate = respiration_rate

    def parse_ble_packet(self, data):
        """解析BLE数据包"""
        # 检查数据包长度
        if len(data) < 46:
            return None, None

        # 验证包头
        if data[0] != 0xFF or data[1] != 0x02 or data[2] != 0xFF:
            return None, None

        try:
            # 提取心率 (字节43，索引42)
            heart_rate = data[42]

            # 提取呼吸率 (字节46，索引45)
            respiration_rate = data[45]

            return heart_rate, respiration_rate

        except IndexError:
            return None, None

    def start_facial_detection(self):
        """启动面部表情检测"""
        if not FACIAL_DETECTION_ENABLED:
            return

        # 初始化相机
        if USE_DEPTHAI_CAMERA:
            try:
                self.depthai_camera = DepthAICamera()
                self.update_status("DepthAI相机已启动")
            except Exception as e:
                self.update_status(f"DepthAI相机启动失败: {str(e)}")
                return
        else:
            self.facial_detector = FacialExpressionDetector()
            self.facial_thread = threading.Thread(target=self.facial_detector.run)
            self.facial_thread.daemon = True
            self.facial_thread.start()

        # 初始化面部表情分析器
        self.face_analyzer = FaceExpressionAnalyzer()
        self.update_status("面部表情检测已启动")

    def init_emg_serial(self):
        """初始化肌电串口"""
        try:
            self.emg_serial = serial.Serial(EMG_SERIAL_PORT, EMG_BAUDRATE, timeout=1)
            self.update_status(f"肌电设备已连接: {EMG_SERIAL_PORT}")
            return True
        except Exception as e:
            self.update_status(f"肌电设备连接失败: {str(e)}")
            return False

    def update_emg_data(self):
        """更新肌电数据"""
        if not self.emg_serial or not self.emg_serial.is_open:
            if not self.init_emg_serial():
                return

        try:
            if self.emg_serial.in_waiting > 0:
                data = self.emg_serial.readline().decode('utf-8', errors='ignore')
                s = data.strip().split(',')

                if not s or s[0] == '':
                    return

                # 更新肌电数据
                for i in range(min(EMG_CHANNELS, len(s))):
                    try:
                        value = int(s[i])
                        # 滚动更新数据
                        self.emg_data[i] = np.roll(self.emg_data[i], -1)
                        self.emg_data[i, -1] = value
                    except ValueError:
                        pass

        except Exception as e:
            self.update_status(f"肌电数据错误: {str(e)}")

    def update_display(self):
        """更新UI显示"""
        # 更新健康数据
        self.heart_rate_label.setText(f"心率: {self.heart_rate} BPM")
        self.respiration_rate_label.setText(f"呼吸率: {self.respiration_rate} BPM")

        # 更新表情数据和摄像头显示
        if FACIAL_DETECTION_ENABLED:
            frame = None

            # 获取当前帧
            if USE_DEPTHAI_CAMERA and self.depthai_camera:
                frame = self.depthai_camera.get_frame()
            elif not USE_DEPTHAI_CAMERA and self.facial_detector:
                frame = self.facial_detector.get_frame()

            # 分析表情
            if frame is not None:
                # 分析表情
                self.facial_expression = self.face_analyzer.analyze_frame(frame)
                self.expression_label.setText(f"面部表情: {self.facial_expression}")

                # 在帧上绘制表情结果
                frame_with_result = frame.copy()
                cv2.putText(frame_with_result, f"表情: {self.facial_expression}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

                # 转换为Qt可显示的格式
                frame_rgb = cv2.cvtColor(frame_with_result, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                q_img = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(q_img)
                self.camera_label.setPixmap(pixmap)

        # 更新肌电数据
        self.update_emg_data()

        # 更新肌电图
        for i, curve in enumerate(self.emg_curves):
            curve.setData(self.x_data, self.emg_data[i])

    def update_status(self, message):
        """更新状态栏消息"""
        self.status_bar.showMessage(message)
        print(message)

    def closeEvent(self, event):
        """关闭事件处理"""
        # 停止深度相机
        if self.depthai_camera:
            self.depthai_camera.stop()

        # 停止普通摄像头
        if not USE_DEPTHAI_CAMERA and self.facial_detector:
            self.facial_detector.stop()

        # 关闭肌电串口
        if self.emg_serial and self.emg_serial.is_open:
            self.emg_serial.close()

        event.accept()


class FacialExpressionDetector:
    """普通摄像头面部表情检测器"""

    def __init__(self):
        # 初始化MediaPipe面部网格模型
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 初始化绘图工具
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        # 表情检测状态
        self.is_running = False
        self.current_expression = "未检测到面部"
        self.current_frame = None

    def run(self):
        """运行表情检测"""
        self.is_running = True
        cap = cv2.VideoCapture(0)  # 打开默认摄像头

        while self.is_running and cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            # 转换图像为RGB格式
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # 处理图像并检测面部
            results = self.face_mesh.process(image)

            # 恢复图像可写状态并转换回BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 保存当前帧
            self.current_frame = image

            # 检查退出键
            if cv2.waitKey(5) & 0xFF == 27:  # ESC键
                break

        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        self.face_mesh.close()

    def stop(self):
        """停止检测"""
        self.is_running = False

    def get_expression(self):
        """获取当前表情"""
        return self.current_expression

    def get_frame(self):
        """获取当前帧"""
        return self.current_frame


class FaceExpressionAnalyzer:
    """面部表情分析器"""

    def __init__(self):
        # 初始化MediaPipe面部网格模型
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def analyze_frame(self, frame):
        """分析帧中的面部表情"""
        if frame is None:
            return "无图像"

        # 转换图像为RGB格式
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 处理图像并检测面部
        results = self.face_mesh.process(image)

        # 分析表情
        expression = self._analyze_expression(results)
        return expression

    def _analyze_expression(self, results):
        """分析面部表情"""
        if not results.multi_face_landmarks:
            return "未检测到面部"

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        # 获取关键点索引
        # 1. 检测微笑（嘴角位置）
        left_mouth_corner = landmarks[61]
        right_mouth_corner = landmarks[291]
        upper_lip = landmarks[0]
        lower_lip = landmarks[17]

        # 计算嘴巴张开程度
        mouth_openness = abs(upper_lip.y - lower_lip.y)

        # 2. 检测眉毛（眉毛位置）
        left_eyebrow = landmarks[105]
        right_eyebrow = landmarks[334]
        left_eye = landmarks[33]
        right_eye = landmarks[263]

        # 计算眉毛位置相对眼睛的位置
        left_eyebrow_raise = left_eye.y - left_eyebrow.y
        right_eyebrow_raise = right_eye.y - right_eyebrow.y

        # 3. 检测眼睛睁开程度
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        right_eye_top = landmarks[386]
        right_eye_bottom = landmarks[374]

        left_eye_open = abs(left_eye_top.y - left_eye_bottom.y)
        right_eye_open = abs(right_eye_top.y - right_eye_bottom.y)

        # 分析表情
        expression = "中性"

        # 检测微笑
        if mouth_openness > 0.05:
            expression = "惊讶" if mouth_openness > 0.1 else "微笑"
        # 检测皱眉
        elif (left_eyebrow_raise < 0.02 or right_eyebrow_raise < 0.02) and mouth_openness < 0.03:
            expression = "皱眉"
        # 检测闭眼
        elif left_eye_open < 0.01 or right_eye_open < 0.01:
            expression = "闭眼"

        return expression


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # 设置高DPI支持
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    main_window = HealthMonitorApp()
    main_window.show()
    sys.exit(app.exec_())