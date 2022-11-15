# coding:utf-8
# @Author     : HT
# @Time       : 2022/8/12 14:47
# @File       : circle.py.py
# @Software   : PyCharm

import math

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QWidget, QApplication


class DrawRect(QWidget):
    def __init__(self, parent=None):
        super(DrawRect, self).__init__(parent)
        self.resize(600, 400)
        self.start_point = None
        self.end_point = None
        self.start = False
        self.drawing = False
        self.radius = 0
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)

    def paintEvent(self, event):
        super(DrawRect, self).paintEvent(event)
        if self.drawing:
            painter = QPainter()
            painter.begin(self)
            pen = QPen(Qt.red, 5)
            painter.setPen(pen)
            painter.drawEllipse(self.start_point, self.radius, self.radius)
            print('圆心：{}，半径:{}'.format(self.start_point, self.radius))
            painter.end()

    def mousePressEvent(self, event):
        if not self.start:
            self.start_point = event.pos()
            self.start = True

    def mouseReleaseEvent(self, event):
        if self.drawing:
            self.drawing = False
            self.start = False
        print('------最终圆------')
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


if __name__ == '__main__':
    app = QApplication([])
    demo = DrawRect()
    demo.show()
    app.exec_()

