from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

class ExportProgress(QObject):

    progress = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.progress_value = 0

    @pyqtSlot()
    def update_progress(self):
        self.progress_value += 1
        self.progress.emit(self.progress_value)