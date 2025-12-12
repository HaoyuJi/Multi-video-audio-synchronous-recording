import sys
import cv2
import numpy as np
import pyaudio
import wave
import os
import time
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QLineEdit, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QGridLayout, \
    QMessageBox
import json
from threading import Lock, Event

# 视频基本参数
video_frame = 30
video_height = 1080
video_width = 1920

# 全局变量 - 将在MainWindow中根据配置动态初始化
latest_frame = []
latest_audio = []
frame_lock = []
audio_lock = []
video_set = []
audio_set = []
start_event = Event()
# 默认焦距值 - 也会在MainWindow中动态调整
DEFAULT_FOCUS_VALUES = []

GLOBAL_START_TIME = 0


def sync_barrier_function(device, index):
    """
    一个简化的同步屏障，用于对齐所有录制线程的起始时间。
    """
    global video_set, audio_set, start_event

    if device == "video":
        video_set[index] = True
    elif device == "audio":
        audio_set[index] = True

    # 如果所有视频和音频线程都已准备就绪
    if all(video_set) and all(audio_set):
        start_event.set()  # 发送信号，让所有等待的线程开始

    # 等待信号以继续执行
    start_event.wait()

    if device == "video":
        video_set[index] = False
    elif device == "audio":
        audio_set[index] = False

    start_event.clear()

    if all(not x for x in video_set) and all(not x for x in audio_set):
        start_event.set()

    # 等待信号以继续执行
    start_event.wait()

    start_event.clear()


class CameraThread(QThread):
    update_frame = pyqtSignal(np.ndarray)
    # 添加一个信号，用于在焦距设置失败时通知主线程
    focus_set_failed = pyqtSignal(int, str)

    def __init__(self, camera_id, initial_focus_value): # 修改：这里的 camera_id 是物理ID
        super().__init__()
        self.camera_id = camera_id
        # *** 修改：使用实际的物理 camera_id 打开摄像头 ***
        self.capture = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)

        # 检查摄像头是否打开
        if not self.capture.isOpened():
            # 这里基本不会被触发，因为主程序已经检查过一次，但作为保障保留
            QMessageBox.critical(None, "错误", f"无法打开摄像头 {self.camera_id}，程序将退出。")
            sys.exit(1)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, video_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, video_height)
        self.capture.set(cv2.CAP_PROP_FPS, video_frame)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        # 设置初始焦距
        self.set_focus(initial_focus_value)

        self.running = True

    def set_focus(self, value):
        """设置摄像头的焦距"""
        success = self.capture.set(cv2.CAP_PROP_FOCUS, value)
        if not success:
            self.focus_set_failed.emit(self.camera_id,
                                       f"摄像头 {self.camera_id} 焦距设置失败，请检查摄像头是否支持此属性或数值范围。")
            print(
                f"Warning: Failed to set focus for camera {self.camera_id} to {value}. It might not support CAP_PROP_FOCUS or the value is out of range.")
        else:
            print(f"Camera {self.camera_id} focus set to {value}.")
        return success

    def run(self):
        # *** 修改：使用 self.camera_id 来引用摄像头，但数据存储仍然依赖外部传入的逻辑索引 ***
        # MainWindow 的 connect 语句确保了 update_frame 信号会发送正确的逻辑索引
        # VideoSaverThread 也是通过逻辑索引来访问 latest_frame，因此这里不需要改动
        logical_index = self.parent().camera_threads.index(self)
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                frame = self.add_timestamp(frame)
                self.update_frame.emit(frame)

                with frame_lock[logical_index]:
                    latest_frame[logical_index] = frame

    def add_timestamp(self, frame):
        absolute_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        cv2.putText(frame, absolute_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return frame

    def stop(self):
        self.running = False
        self.wait()
        self.capture.release()


class VideoSaverThread(QThread):
    def __init__(self, save_path, video_filename, camera_index):
        super().__init__()
        os.makedirs(save_path, exist_ok=True)
        video_filepath = os.path.join(save_path, video_filename)
        self.camera_index = camera_index
        self.save_path = save_path
        self.running = True
        self.video_writer = cv2.VideoWriter(
            video_filepath,
            cv2.VideoWriter_fourcc('M', 'P', '4', '2'),
            video_frame,
            (video_width, video_height)
        )
        GLOBAL_START_TIME = time.perf_counter()
        self.frame_count = 0
        self.time_interval = 1.0 / video_frame

    def run(self):
        global latest_frame, frame_lock, start_event

        sync_barrier_function("video", self.camera_index)
        GLOBAL_START_TIME = time.perf_counter()
        # print(GLOBAL_START_TIME)
        date_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        if self.camera_index == 0:
            with open(f"{self.save_path}\\datetime.txt", "w") as f:
                f.write(f"start: {date_start_time}\n")

        while self.running:
            self.frame_count += 1
            self.target_time = GLOBAL_START_TIME + self.frame_count * self.time_interval

            if self.frame_count % 300 == 0 and self.camera_index == 0:
                with open(f"{self.save_path}\\datetime.txt", "a") as f:
                    f.write(f"frame_count{self.frame_count}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}\n")

            with frame_lock[self.camera_index]:
                if latest_frame[self.camera_index] is not None:
                    self.video_writer.write(latest_frame[self.camera_index])

            if self.running:
                sync_barrier_function("video", self.camera_index)

            self.wait_time = self.target_time - time.perf_counter()
            if self.wait_time > 0:
                time.sleep(self.wait_time)

    def stop(self):
        if self.running:
            self.running = False
            start_event.set()
            self.wait()
            self.video_writer.release()


class AudioThread(QThread):
    def __init__(self, device_index):
        super().__init__()
        self.device_index = device_index
        self.running = True
        self.p = pyaudio.PyAudio()
        self.stream = None
        try:
            self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True,
                                      input_device_index=self.device_index + 1,
                                      frames_per_buffer=44100 // video_frame)
        except IOError:
            QMessageBox.critical(None, "错误", f"无法打开音频设备 {self.device_index}，程序将退出。")
            sys.exit(1)
        self.save_path = None

    def run(self):
        global latest_audio, audio_lock
        while self.running:
            try:
                data = self.stream.read(44100 // video_frame)
                with audio_lock[self.device_index]:
                    latest_audio[self.device_index] = data
            except IOError as e:
                print(f"Error reading audio data from device {self.device_index}: {e}")
                self.stop()
                break

    def stop(self):
        self.running = False
        self.wait()
        self.release_audio()

    def release_audio(self):
        if self.stream.is_active():
            self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


class AudioSaverThread(QThread):
    def __init__(self, save_path, audio_filename, device_index):
        super().__init__()
        self.device_index = device_index
        self.save_path = save_path
        self.audio_filename = audio_filename
        GLOBAL_START_TIME = time.perf_counter()
        self.frame_count = 0
        self.time_interval = 1.0 / video_frame
        self.running = True
        self.frames = []

    def run(self):
        global latest_audio, audio_lock, start_event

        sync_barrier_function("audio", self.device_index)
        GLOBAL_START_TIME = time.perf_counter()
        # print(GLOBAL_START_TIME)

        while self.running:
            self.frame_count += 1
            self.target_time = GLOBAL_START_TIME + self.frame_count * self.time_interval

            with audio_lock[self.device_index]:
                if latest_audio[self.device_index] is not None:
                    self.frames.append(latest_audio[self.device_index])

            if self.running:
                sync_barrier_function("audio", self.device_index)

            self.wait_time = self.target_time - time.perf_counter()
            if self.wait_time > 0:
                time.sleep(self.wait_time)

    def stop(self):
        if self.running:
            self.running = False
            start_event.set()
            self.wait()
            self.save_audio()

    def save_audio(self):
        os.makedirs(self.save_path, exist_ok=True)
        audio_filepath = os.path.join(self.save_path, self.audio_filename)
        wf = wave.open(audio_filepath, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(self.frames))
        wf.close()


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.config_file = "config.json"

        # --- 动态化核心部分 ---
        self.setup_dynamic_variables()

        # *** 新增: 摄像头检测与验证 ***
        self.detected_camera_ids = self.detect_available_cameras()
        if len(self.detected_camera_ids) != self.video_num:
            error_message = (f"摄像头数量不匹配！\n\n"
                             f"配置文件 ('config.json') 要求开启 {self.video_num} 个摄像头。\n"
                             f"但系统实际检测到 {len(self.detected_camera_ids)} 个可用摄像头。\n\n"
                             f"检测到的摄像头ID为: {self.detected_camera_ids}\n\n"
                             f"请检查摄像头连接或修改 'config.json' 文件中的 'video_num' 值。\n"
                             f"程序即将退出。")
            QMessageBox.critical(None, "摄像头配置错误", error_message)
            sys.exit(1)
        # *** 检测结束 ***

        self.setWindowTitle(f"{self.video_num}路音视频采集程序")
        self.save_root = self.load_config("save_root", "D:\\guangzhou")
        self.focus_values = self.load_config("focus_values", DEFAULT_FOCUS_VALUES)

        self.display_width = 480
        self.display_height = 270

        if len(self.focus_values) != self.video_num:
            self.focus_values = [30] * self.video_num
            self.save_config("focus_values", self.focus_values)

        self.init_ui()
        self.init_threads()

        self.recording = False
        self.video_saver_threads = []
        self.audio_saver_threads = []

    # *** 新增方法: 检测可用摄像头 ***
    def detect_available_cameras(self):
        """
        扫描并返回所有可用的摄像头ID列表。
        """
        detected_ids = []
        # 检查前10个索引，通常足够了
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                detected_ids.append(i)
                cap.release()
        print(f"系统检测到可用摄像头ID: {detected_ids}")
        return detected_ids

    def setup_dynamic_variables(self):
        """根据配置文件设置所有动态变量"""
        global latest_frame, latest_audio, frame_lock, audio_lock, video_set, audio_set, DEFAULT_FOCUS_VALUES

        config = self.load_config("", {})
        self.video_num = config.get("video_num", 2)
        if "video_num" not in config:
            self.save_config("video_num", self.video_num)

        latest_frame = [None] * self.video_num
        latest_audio = [None] * self.video_num
        frame_lock = [Lock() for _ in range(self.video_num)]
        audio_lock = [Lock() for _ in range(self.video_num)]
        video_set = [False] * self.video_num
        audio_set = [False] * self.video_num
        DEFAULT_FOCUS_VALUES = [30] * self.video_num

    def init_ui(self):
        """初始化用户界面"""
        main_layout = QVBoxLayout()
        self.warning_label = QLabel("提示：请不要长时间打开该程序，在不用时请关闭程序，以免造成电脑卡死。")
        self.warning_label.setStyleSheet("color: red; font-size: 15px;")
        main_layout.addWidget(self.warning_label, alignment=QtCore.Qt.AlignCenter)
        path_layout = QHBoxLayout()
        self.path_label = QLabel("保存路径:")
        self.path_display = QLineEdit(self.save_root)
        self.path_display.setReadOnly(True)
        self.choose_path_button = QPushButton("更改保存路径")
        self.choose_path_button.clicked.connect(self.choose_save_path)
        path_layout.addWidget(self.path_label)
        path_layout.addWidget(self.path_display)
        path_layout.addWidget(self.choose_path_button)
        main_layout.addLayout(path_layout)
        focus_layout = QGridLayout()
        self.focus_labels = []
        self.focus_inputs = []
        for i in range(self.video_num):
            # *** 修改: UI标签使用检测到的ID ***
            label = QLabel(f"摄像头 ID {self.detected_camera_ids[i]} 焦距:")
            input_field = QLineEdit(str(self.focus_values[i]))
            input_field.setValidator(QtGui.QIntValidator(0, 255))
            self.focus_labels.append(label)
            self.focus_inputs.append(input_field)
            focus_layout.addWidget(label, 0, i)
            focus_layout.addWidget(input_field, 1, i)
        self.set_focus_button = QPushButton("设置焦距")
        self.set_focus_button.clicked.connect(self.apply_focus_settings)
        focus_group_box = QtWidgets.QGroupBox("焦距设置")
        focus_v_layout = QVBoxLayout()
        focus_v_layout.addLayout(focus_layout)
        focus_v_layout.addWidget(self.set_focus_button, alignment=QtCore.Qt.AlignCenter)
        focus_group_box.setLayout(focus_v_layout)
        main_layout.addWidget(focus_group_box)
        size_group_box = QtWidgets.QGroupBox("显示尺寸设置")
        size_h_layout = QHBoxLayout()
        resolutions = [(480, 270), (640, 360), (800, 450), (960, 540), (1280, 720), (1600, 900), (1920, 1080)]
        for w, h in resolutions:
            size_button = QPushButton(f"{w}x{h}")
            size_button.clicked.connect(lambda checked, w=w, h=h: self.change_display_size(w, h))
            size_h_layout.addWidget(size_button)
        size_group_box.setLayout(size_h_layout)
        main_layout.addWidget(size_group_box)
        self.image_labels = []
        video_grid_layout = QGridLayout()
        cols = 2
        for i in range(self.video_num):
            label = QLabel()
            label.setFixedSize(self.display_width, self.display_height)
            label.setStyleSheet("background-color: black;")
            self.image_labels.append(label)
            row, col = divmod(i, cols)
            video_grid_layout.addWidget(label, row, col)
        main_layout.addLayout(video_grid_layout)
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("开始录制")
        self.stop_button = QPushButton("停止录制")
        self.close_button = QPushButton("关闭程序")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        self.close_button.clicked.connect(self.close_program)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.close_button)
        main_layout.addLayout(button_layout)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.start_button.setFont(font)
        self.stop_button.setFont(font)
        self.choose_path_button.setFont(font)
        self.close_button.setFont(font)
        self.set_focus_button.setFont(font)
        self.start_button.setFixedSize(150, 30)
        self.stop_button.setFixedSize(150, 30)
        self.choose_path_button.setFixedSize(150, 30)
        self.close_button.setFixedSize(150, 30)
        self.set_focus_button.setFixedSize(150, 30)
        self.recording_label = QLabel("正在录制...")
        self.recording_label.setFont(QtGui.QFont("Arial", 24, QtGui.QFont.Bold))
        self.recording_label.setStyleSheet("color: red")
        self.recording_label.setAlignment(QtCore.Qt.AlignCenter)
        self.recording_label.setVisible(False)
        main_layout.addWidget(self.recording_label)
        self.setLayout(main_layout)
        self.display_black_frames()

    def init_threads(self):
        """动态初始化并启动所有摄像头和音频线程"""
        self.camera_threads = []
        self.audio_threads = []

        # *** 修改: 遍历检测到的摄像头ID列表 ***
        # logical_index 用于UI和数据列表 (0, 1, 2...)
        # physical_camera_id 用于 VideoCapture (例如 0, 2, 4...)
        for logical_index, physical_camera_id in enumerate(self.detected_camera_ids):
            # 创建并连接摄像头线程
            # 传递物理ID给线程用于打开设备，传递逻辑ID对应的焦距值
            cam_thread = CameraThread(physical_camera_id, self.focus_values[logical_index])
            cam_thread.setParent(self) # 方便在线程内回溯
            # 使用 lambda 传递逻辑索引，确保信号能正确更新对应的UI
            cam_thread.update_frame.connect(lambda frame, index=logical_index: self.update_image(index, frame))
            cam_thread.focus_set_failed.connect(self.handle_focus_set_failure)
            self.camera_threads.append(cam_thread)
            time.sleep(0.1)

        # 音频线程的创建保持不变，仍使用逻辑索引
        for i in range(self.video_num):
            audio_thread = AudioThread(i)
            self.audio_threads.append(audio_thread)
            time.sleep(0.1)

        # 启动所有线程
        for thread in self.camera_threads:
            thread.start()
        for thread in self.audio_threads:
            thread.start()

    def load_config(self, key, default_value):
        """从配置文件加载配置，如果不存在则返回默认值"""
        if not os.path.exists(self.config_file):
            return default_value
        with open(self.config_file, "r") as file:
            try:
                config = json.load(file)
            except json.JSONDecodeError:
                config = {}
        if key:
            return config.get(key, default_value)
        return config

    def save_config(self, key, value):
        config = self.load_config("", {})
        config[key] = value
        with open(self.config_file, "w") as file:
            json.dump(config, file, indent=4)

    def choose_save_path(self):
        selected_dir = QFileDialog.getExistingDirectory(self, "选择保存路径", self.save_root)
        if selected_dir:
            self.save_root = selected_dir
            self.path_display.setText(self.save_root)
            self.save_config("save_root", self.save_root)

    def apply_focus_settings(self):
        new_focus_values = []
        for i, input_field in enumerate(self.focus_inputs):
            try:
                value = int(input_field.text())
                if not (0 <= value <= 255):
                    QMessageBox.warning(self, "无效焦距", f"摄像头 {self.detected_camera_ids[i]} 的焦距值应在 0 到 255 之间。")
                    return
                new_focus_values.append(value)
            except ValueError:
                QMessageBox.warning(self, "输入错误", f"摄像头 {self.detected_camera_ids[i]} 的焦距值不是有效的整数。")
                return

        self.focus_values = new_focus_values
        self.save_config("focus_values", self.focus_values)
        for i, thread in enumerate(self.camera_threads):
            thread.set_focus(self.focus_values[i])
        QMessageBox.information(self, "焦距设置", "所有摄像头焦距已成功设置。")

    def handle_focus_set_failure(self, camera_id, message):
        QMessageBox.warning(self, f"摄像头ID {camera_id} 焦距设置失败", message)

    def change_display_size(self, width, height):
        """
        更改所有视频预览窗口的尺寸
        """
        self.display_width = width
        self.display_height = height
        for label in self.image_labels:
            label.setFixedSize(width, height)
        print(f"Display size changed to {width}x{height}")

    def close_program(self):
        if self.recording:
            self.stop_recording()
            time.sleep(1)
        self.close()

    def display_black_frames(self):
        black_image = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        for i in range(self.video_num):
            self.update_image(i, black_image)

    def start_recording(self):
        if not os.path.exists(self.save_root):
            QMessageBox.warning(self, "路径错误", "保存路径不存在，请更改路径后重试。")
            return

        self.start_time = datetime.now()
        self.save_path = os.path.join(self.save_root, self.start_time.strftime("%Y_%m_%d_%H_%M_%S"))

        self.video_saver_threads = []
        self.audio_saver_threads = []

        for i in range(self.video_num):
            video_saver = VideoSaverThread(self.save_path, f"output_video{i}.avi", i)
            self.video_saver_threads.append(video_saver)
            time.sleep(0.05)

        for i in range(self.video_num):
            audio_saver = AudioSaverThread(self.save_path, f"output_audio{i}.wav", i)
            self.audio_saver_threads.append(audio_saver)
            time.sleep(0.05)

        global video_set, audio_set, start_event
        start_event.clear()
        video_set = [False] * self.video_num
        audio_set = [False] * self.video_num
        for thread in self.video_saver_threads + self.audio_saver_threads:
            thread.start()

        self.update_recording_status(True)
        self.recording = True

    def stop_recording(self):
        if self.recording:
            start_event.set()

            for thread in self.video_saver_threads:
                thread.stop()
            for thread in self.audio_saver_threads:
                thread.stop()

            for thread in self.video_saver_threads + self.audio_saver_threads:
                thread.wait()

            self.update_recording_status(False)
            self.recording = False

    def update_recording_status(self, is_recording):
        self.recording = is_recording
        self.start_button.setEnabled(not is_recording)
        self.stop_button.setEnabled(is_recording)
        self.recording_label.setVisible(is_recording)
        if is_recording:
            self.start_button.setText("正在录制")
            self.start_button.setStyleSheet("background-color: red; color: white;")
        else:
            self.start_button.setText("开始录制")
            self.start_button.setStyleSheet("")

    def closeEvent(self, event):
        print("Closing window, stopping all threads...")
        if self.recording:
            self.stop_recording()

        for thread in self.camera_threads:
            thread.stop()
        for thread in self.audio_threads:
            thread.stop()
        event.accept()

    def update_image(self, index, frame):
        """动态更新指定索引的图像标签"""
        if index < len(self.image_labels):
            self.display_image(self.image_labels[index], frame)

    def display_image(self, label, frame):
        display_size = label.size()
        if frame is None or frame.size == 0:
            black_image = np.zeros((display_size.height(), display_size.width(), 3), dtype=np.uint8)
            rgb_image = cv2.cvtColor(black_image, cv2.COLOR_BGR2RGB)
        else:
            resized_frame = cv2.resize(frame, (display_size.width(), display_size.height()))
            rgb_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        label.setPixmap(QtGui.QPixmap.fromImage(qt_image))


if __name__ == "__main__":
    # *** 修改: 必须先创建 QApplication 实例才能使用 QMessageBox ***
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())