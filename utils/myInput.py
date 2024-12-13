# from PyQt5 import QtCore, QtGui, QtWidgets
# import sys


# import sys
#
# from PyQt5.QtCore import Qt
# from PyQt5.QtWidgets import (
#     QApplication,
#     QHBoxLayout,
#     QLabel,
#     QMainWindow,
#     QPushButton,
#     QStackedLayout,
#     QVBoxLayout,
#     QWidget,
# )
#
# # from layout_colorwidget import Color
# from PyQt5.QtGui import QPalette, QColor
#
# class Color(QWidget):
#
#     def __init__(self, color):
#         super(Color, self).__init__()
#         self.setAutoFillBackground(True)
#
#         palette = self.palette()
#         palette.setColor(QPalette.Window, QColor(color))
#         self.setPalette(palette)
#
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#
#         self.setWindowTitle("My App")
#
#         pagelayout = QVBoxLayout()
#         button_layout = QVBoxLayout()
#         self.stacklayout = QStackedLayout()
#
#         pagelayout.addLayout(button_layout)
#         pagelayout.addLayout(self.stacklayout)
#
#         btn = QPushButton("red")
#         btn.pressed.connect(self.activate_tab_1)
#         button_layout.addWidget(btn)
#         self.stacklayout.addWidget(Color("red"))
#
#         btn = QPushButton("green")
#         btn.pressed.connect(self.activate_tab_2)
#         button_layout.addWidget(btn)
#         self.stacklayout.addWidget(Color("green"))
#
#         btn = QPushButton("yellow")
#         btn.pressed.connect(self.activate_tab_3)
#         button_layout.addWidget(btn)
#         self.stacklayout.addWidget(Color("yellow"))
#
#         widget = QWidget()
#         widget.setLayout(pagelayout)
#         self.setCentralWidget(widget)
#
#     def activate_tab_1(self):
#         self.stacklayout.setCurrentIndex(0)
#
#     def activate_tab_2(self):
#         self.stacklayout.setCurrentIndex(1)
#
#     def activate_tab_3(self):
#         self.stacklayout.setCurrentIndex(2)
#
#
# app = QApplication(sys.argv)
#
# window = MainWindow()
# window.show()
#
# app.exec()


import sys

from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDateTimeEdit,
    QDial,
    QDoubleSpinBox,
    QFontComboBox,
    QLabel,
    QLCDNumber,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
    QInputDialog,
)

# Subclass QMainWindow to customize your application's main window
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#
#         self.setWindowTitle("Widgets App")
#
#         layout = QVBoxLayout()
#         widgets = [
#             QCheckBox,
#             QComboBox,
#             QDateEdit,
#             QDateTimeEdit,
#             QDial,
#             QDoubleSpinBox,
#             QFontComboBox,
#             QLCDNumber,
#             QLabel,
#             QLineEdit,
#             QProgressBar,
#             QPushButton,
#             QRadioButton,
#             QSlider,
#             QSpinBox,
#             QTimeEdit,
#         ]
#
#         for w in widgets:
#             layout.addWidget(w())
#
#         widget = QWidget()
#         widget.setLayout(layout)
#
#         # Set the central widget of the Window. Widget will expand
#         # to take up all the space in the window by default.
#         self.setCentralWidget(widget)
#
# app = QApplication(sys.argv)
# window = MainWindow()
# window.show()
# app.exec()


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("My App")

        widget = QComboBox()
        widget.addItems(["One", "Two", "Three"])

        # Sends the current index (position) of the selected item.
        widget.currentIndexChanged.connect( self.index_changed )

        # There is an alternate signal to send the text.
        widget.currentTextChanged.connect( self.text_changed )

        self.setCentralWidget(widget)

    def index_changed(self, i): # i is an int
        print(i)

    def text_changed(self, s): # s is a str
        print(s)



app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()



#
# class Ui_MainWindow(QtWidgets.QWidget):
#     def setupUi(self, MainWindow):
#         MainWindow.resize(422, 255)
#         self.centralwidget = QtWidgets.QWidget(MainWindow)
#
#         self.pushButton = QtWidgets.QPushButton(self.centralwidget)
#         self.pushButton.setGeometry(QtCore.QRect(160, 130, 93, 28))
#
#         # For displaying confirmation message along with user's info.
#         self.label = QtWidgets.QLabel(self.centralwidget)
#         self.label.setGeometry(QtCore.QRect(170, 40, 201, 111))
#
#         # Keeping the text of label empty initially.
#         self.label.setText("")
#
#         MainWindow.setCentralWidget(self.centralwidget)
#         self.retranslateUi(MainWindow)
#         QtCore.QMetaObject.connectSlotsByName(MainWindow)
#
#     def retranslateUi(self, MainWindow):
#         _translate = QtCore.QCoreApplication.translate
#         MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
#         self.pushButton.setText(_translate("MainWindow", "Proceed"))
#         self.pushButton.clicked.connect(self.takeinputs)
#
#     def takeinputs(self):
#
#         name, done1 = QtWidgets.QInputDialog.getText(
#             self, 'Input Dialog', 'Enter your name:')
#
#         roll, done2 = QtWidgets.QInputDialog.getInt(
#             self, 'Input Dialog', 'Enter your roll:')
#
#         cgpa, done3 = QtWidgets.QInputDialog.getDouble(
#             self, 'Input Dialog', 'Enter your CGPA:')
#
#         langs = ['C', 'c++', 'Java', 'Python', 'Javascript']
#         lang, done4 = QtWidgets.QInputDialog.getItem(
#             self, 'Input Dialog', 'Language you know:', langs)
#
#         if done1 and done2 and done3 and done4:
#             # Showing confirmation message along
#             # with information provided by user.
#             self.label.setText('Information stored Successfully\nName: '
#                                + str(name) + '(' + str(roll) + ')' + '\n' + 'CGPA: '
#                                + str(cgpa) + '\nSelected Language: ' + str(lang))
#
#             # Hide the pushbutton after inputs provided by the use.
#             self.pushButton.hide()
#
#
# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     MainWindow = QtWidgets.QMainWindow()
#     ui = Ui_MainWindow()
#     ui.setupUi(MainWindow)
#     MainWindow.show()
#     sys.exit(app.exec_())