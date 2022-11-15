from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import pyqtSignal
import math
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QWidget,QInputDialog,QPushButton,QLineEdit


class LabelMouse(QLabel):
    double_clicked = pyqtSignal()

    # 鼠标双击事件
    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit()

    def mouseMoveEvent(self):
        """
        当鼠标划过标签label2时触发事件
        :return:
        """
        print('当鼠标划过标签label2时触发事件')
# class Label_click_Mouse(QLabel):
#     clicked = pyqtSignal()
#
#     # 鼠标点击事件
#     def mousePressEvent(self, event):
#         self.clicked.emit()



class Label_click_Mouse(QLabel,QWidget):
    send_circle = pyqtSignal(list)
    def __init__(self, parent=None):
        super(Label_click_Mouse, self).__init__(parent)
        self.start_point = None
        self.end_point = None
        self.start = False
        self.drawing = False
        self.radius = 0
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)
        self.painter = QPainter()


    def paintEvent(self, event):
        super(Label_click_Mouse, self).paintEvent(event)
        if self.drawing:
            self.painter.begin(self)
            pen = QPen(Qt.green, 5)
            self.painter.setPen(pen)
            self.painter.drawEllipse(self.start_point, self.radius, self.radius)
            # print('圆心：{}，半径:{}'.format(self.start_point, self.radius))
            self.painter.end()

    def mousePressEvent(self, event):
        if not self.start:
            self.start_point = event.pos()
            self.start = True

    def mouseReleaseEvent(self, event):
        if self.drawing:
            self.drawing = False
            self.start = False
        print('------最终圆------')
        num, ok = QInputDialog.getInt(self, '螺栓数量', '请输入该螺栓所要求的垫片数量')
        print(num)
        print(ok)
        if ok:
            self.send_circle.emit([self.start_point.x(),self.start_point.y(), self.radius,num])

        print('圆心：{}，半径:{}'.format(self.start_point, self.radius))

    def mouseMoveEvent(self, event):
        if self.start:
            self.end_point = event.pos()
            self.radius = self.calc_radius()
            self.drawing = True
            self.update()

    def calc_radius(self):
        return math.sqrt(
            (self.start_point.x() - self.end_point.x()) ** 2 + (self.start_point.y() - self.end_point.y()) ** 2)