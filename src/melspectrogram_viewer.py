"""Realtime mel-spectrogram viewer using PyQtGraph (compact setup)."""

import sys
from typing import Tuple, List
import numpy as np
import librosa
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt5.QtWidgets import QApplication

from pyqtgraph.graphicsItems.GradientEditorItem import Gradients


class MelSpectrogramViewer:
    """Compute and display a mel-spectrogram with a simple Qt timer loop."""

    def __init__(
        self,
        sr: int,
        shape: Tuple[int, int],
        n_mels: int = 128,
        n_frames: int = 50,
        fps: int = 50,
        size: Tuple[int, int] = (1000, 1000),
        title: str = "realtime melspectrogram",
    ) -> None:
        """Initialize buffers, mel filterbank, and PyQtGraph widgets."""
        # signal/fft/mel parameters and buffers
        self.n_frames: int = n_frames
        self.n_ch: int = shape[0]
        self.n_chunk: int = shape[1]
        self.n_freqs: int = self.n_chunk // 2 + 1
        self.n_mels: int = n_mels
        self.sig: np.ndarray = np.zeros(shape)
        self.x: np.ndarray = np.zeros(self.n_chunk)
        self.specs: np.ndarray = np.zeros((self.n_freqs))
        self.melspecs: np.ndarray = np.zeros((self.n_frames, self.n_mels))
        self.window: np.ndarray = np.hamming(self.n_chunk)
        self.fft = np.fft.rfft
        self.melfreqs: np.ndarray = librosa.mel_frequencies(n_mels=self.n_mels)
        self.melfb: np.ndarray = librosa.filters.mel(
            sr=sr, n_fft=self.n_chunk, n_mels=self.n_mels
        )
        self.fps: int = fps
        self.iter: int = 0

        # PyQtGraph setup
        app = QApplication([])
        win = pg.GraphicsLayoutWidget()
        win.resize(size[0], size[1])
        win.show()

        # ImageItem and colormap
        if "magma" in Gradients:
            cmap = pg.ColorMap(*zip(*Gradients["magma"]["ticks"]))
        else:
            raise ValueError("指定されたカラーマップ 'magma' は存在しません")

        imageitem = pg.ImageItem(border="k")
        # cmap = pg.colormap.getFromMatplotlib("magma")
        bar = pg.ColorBarItem(colorMap=cmap)
        bar.setImageItem(imageitem)
        imageitem.setLookupTable(cmap.getLookupTable())

        # ViewBox
        viewbox = win.addViewBox()
        viewbox.setAspectLocked(lock=True)
        viewbox.addItem(imageitem)

        # Axes
        axis_left = pg.AxisItem(orientation="left")
        n_ygrid = 5
        yticks: List[Tuple[int, str]] = []
        interval = max(1, self.n_mels // n_ygrid)
        for k in range(n_ygrid + 1):
            index = min(k * interval, self.n_mels - 1)
            yticks.append((index, str(int(self.melfreqs[index]))))
        axis_left.setTicks([yticks])

        # PlotItem
        plotitem = pg.PlotItem(viewBox=viewbox, axisItems={"left": axis_left})
        # グラフの範囲
        plotitem.setLimits(
            minXRange=0, maxXRange=self.n_frames, minYRange=0, maxYRange=self.n_mels
        )
        # アスペクト比固定
        plotitem.setAspectLocked(lock=True)
        # マウス操作無効
        plotitem.setMouseEnabled(x=False, y=False)
        # ラベルのセット
        plotitem.setLabels(bottom="Time (s)", left="Frequency (Hz)")

        def custom_xticks():
            ticks = []
            for i in range(0, self.n_frames + 1, self.n_frames // 2):
                ticks.append((i, str(i / self.fps)))
            return [ticks]

        axis_bottom = plotitem.getAxis("bottom")
        axis_bottom.setTicks(custom_xticks())

        win.addItem(plotitem)

        self.app = app
        self.win = win
        self.viewbox = viewbox
        self.plotitem = plotitem
        self.imageitem = imageitem

        pg.setConfigOptions(antialias=True)

    def run_app(self) -> None:
        """Start a Qt timer to update the image at the target FPS."""
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        # timer.start(1 / self.fps * 1000)

        fps = float(self.fps) if getattr(self, "fps", None) else 60.0
        interval_ms = max(1, int(round(1000.0 / fps)))
        timer.start(interval_ms)

        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QApplication.instance().exec_()

    def update(self) -> None:
        """Compute next mel frame and refresh the image."""
        if self.iter > 0:
            self.viewbox.disableAutoRange()

        # 最新をスペクトログラム格納するインデックス
        idx = self.iter % self.n_frames
        # モノラル信号算出
        self.x[:] = 0.5 * (self.sig[0] + self.sig[1])
        # FFT => パワー算出
        self.x[:] = self.x[:] * self.window
        self.specs[:] = np.abs(self.fft(self.x)) ** 2
        # メルスペクトログラム算出
        self.melspecs[idx, :] = np.dot(self.melfb, self.specs)

        # 描画
        pos = idx + 1 if idx < self.n_frames else 0
        self.imageitem.setImage(
            librosa.power_to_db(
                np.r_[self.melspecs[pos : self.n_frames], self.melspecs[0:pos]],
                ref=np.max,
            )
        )
        self.iter += 1

    def callback_sigproc(self, sig: np.ndarray) -> None:
        """Receive latest audio frame (shape: [channels, chunk])."""
        self.sig[:] = sig
