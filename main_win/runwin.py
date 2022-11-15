# coding:utf-8
# @Author     : HT
# @Time       : 2022/8/12 15:26
# @File       : runwin.py
# @Software   : PyCharm

import sys
import win
from PyQt5.QtWidgets import QApplication,QMainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = win.Ui_mainWindow()
    # 向主窗口上添加控件
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())