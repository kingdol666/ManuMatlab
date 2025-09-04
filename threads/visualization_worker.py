# -*- coding: utf-8 -*-
"""VisualizationWorker
异步可视化后台工作者，负责在独立线程中根据帧率计算当前帧索引和仿真时间，
通过信号将结果发送到主线程，由主线程完成 GUI 更新。
"""

from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot, Qt

class VisualizationWorker(QObject):
    """在后台线程计算动画帧、仿真时间等数据。"""

    update_frame = pyqtSignal(int, float)  # 当前帧索引, 当前仿真时间
    finished = pyqtSignal()

    def __init__(self, total_simulation_time: float, parent: QObject | None = None):
        super().__init__(parent)
        self.total_simulation_time = total_simulation_time
        self.animation_frame: int = 0
        self._running: bool = False
        self._timer: QTimer | None = None
        self._animation_duration_frames: int = 200  # 与 VisualizationManager 保持一致
        self.speed_multiplier: float = 1.0
        self.base_interval_ms: int = 30

    # ---------------- 控制接口 ----------------
    def start(self):
        """在调用方线程（通常是 QThread）中创建并启动定时器。"""
        if self._running:
            return
        self._running = True
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._tick)
        initial_interval = int(self.base_interval_ms / self.speed_multiplier)
        self._timer.start(initial_interval)

    @pyqtSlot()
    def pause(self):
        if self._timer and self._timer.isActive():
            self._timer.stop()
        elif self._timer:
            self._timer.start()

    @pyqtSlot()
    def stop(self):
        self._running = False
        if self._timer:
            self._timer.stop()
            self._timer.deleteLater()
            self._timer = None
        self.finished.emit()

    @pyqtSlot()
    def next_frame(self):
        self.animation_frame += 1
        self._emit_update()

    @pyqtSlot()
    def previous_frame(self):
        if self.animation_frame > 0:
            self.animation_frame -= 1
            self._emit_update()

    @pyqtSlot(float)
    def set_speed_multiplier(self, multiplier: float):
        """设置播放速度乘数并更新定时器间隔。"""
        if multiplier <= 0:
            return
        self.speed_multiplier = multiplier
        if self._timer and self._timer.isActive():
            new_interval = int(self.base_interval_ms / self.speed_multiplier)
            self._timer.setInterval(new_interval)

    # ---------------- 内部 ----------------
    def _tick(self):
        if not self._running:
            return
        self.animation_frame += 1
        self._emit_update()

    def _emit_update(self):
        time_ratio = (self.animation_frame % self._animation_duration_frames) / self._animation_duration_frames
        current_sim_time = time_ratio * self.total_simulation_time
        self.update_frame.emit(self.animation_frame, current_sim_time)
