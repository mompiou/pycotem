# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'intersectionsUI.ui'
#
# Created by: PyQt4 UI code generator 4.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Intersections(object):
    def setupUi(self, Intersections):
        Intersections.setObjectName(_fromUtf8("Intersections"))
        Intersections.resize(545, 687)
        self.layoutWidget = QtGui.QWidget(Intersections)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 531, 671))
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.gridLayout = QtGui.QGridLayout(self.layoutWidget)
        self.gridLayout.setMargin(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.intersection_cone_text_label = QtGui.QLabel(self.layoutWidget)
        self.intersection_cone_text_label.setObjectName(_fromUtf8("intersection_cone_text_label"))
        self.gridLayout.addWidget(self.intersection_cone_text_label, 16, 0, 1, 1)
        self.n_proj_text_label = QtGui.QLabel(self.layoutWidget)
        self.n_proj_text_label.setObjectName(_fromUtf8("n_proj_text_label"))
        self.gridLayout.addWidget(self.n_proj_text_label, 8, 0, 1, 1)
        self.n1n2_label = QtGui.QLabel(self.layoutWidget)
        self.n1n2_label.setText(_fromUtf8(""))
        self.n1n2_label.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByMouse)
        self.n1n2_label.setObjectName(_fromUtf8("n1n2_label"))
        self.gridLayout.addWidget(self.n1n2_label, 5, 0, 1, 2)
        self.cone_entry = QtGui.QLineEdit(self.layoutWidget)
        self.cone_entry.setObjectName(_fromUtf8("cone_entry"))
        self.gridLayout.addWidget(self.cone_entry, 18, 1, 1, 1)
        self.cone_angle_entry = QtGui.QLineEdit(self.layoutWidget)
        self.cone_angle_entry.setObjectName(_fromUtf8("cone_angle_entry"))
        self.gridLayout.addWidget(self.cone_angle_entry, 22, 1, 1, 1)
        self.n1_text_label = QtGui.QLabel(self.layoutWidget)
        self.n1_text_label.setObjectName(_fromUtf8("n1_text_label"))
        self.gridLayout.addWidget(self.n1_text_label, 1, 0, 1, 1)
        self.angle_proj_entry = QtGui.QLineEdit(self.layoutWidget)
        self.angle_proj_entry.setObjectName(_fromUtf8("angle_proj_entry"))
        self.gridLayout.addWidget(self.angle_proj_entry, 11, 1, 1, 1)
        self.n_cone_text_label = QtGui.QLabel(self.layoutWidget)
        self.n_cone_text_label.setObjectName(_fromUtf8("n_cone_text_label"))
        self.gridLayout.addWidget(self.n_cone_text_label, 17, 0, 1, 1)
        self.intersection_text_label = QtGui.QLabel(self.layoutWidget)
        self.intersection_text_label.setObjectName(_fromUtf8("intersection_text_label"))
        self.gridLayout.addWidget(self.intersection_text_label, 0, 0, 1, 1)
        self.pushButton_intersection_cone = QtGui.QPushButton(self.layoutWidget)
        self.pushButton_intersection_cone.setObjectName(_fromUtf8("pushButton_intersection_cone"))
        self.gridLayout.addWidget(self.pushButton_intersection_cone, 23, 0, 1, 1)
        self.checkBox = QtGui.QCheckBox(self.layoutWidget)
        self.checkBox.setObjectName(_fromUtf8("checkBox"))
        self.gridLayout.addWidget(self.checkBox, 4, 1, 1, 1)
        self.checkBox_3 = QtGui.QCheckBox(self.layoutWidget)
        self.checkBox_3.setObjectName(_fromUtf8("checkBox_3"))
        self.gridLayout.addWidget(self.checkBox_3, 23, 1, 1, 1)
        self.intersection_proj_text_label = QtGui.QLabel(self.layoutWidget)
        self.intersection_proj_text_label.setObjectName(_fromUtf8("intersection_proj_text_label"))
        self.gridLayout.addWidget(self.intersection_proj_text_label, 7, 0, 1, 2)
        self.angle_text_label = QtGui.QLabel(self.layoutWidget)
        self.angle_text_label.setObjectName(_fromUtf8("angle_text_label"))
        self.gridLayout.addWidget(self.angle_text_label, 11, 0, 1, 1)
        self.n2_text_label = QtGui.QLabel(self.layoutWidget)
        self.n2_text_label.setObjectName(_fromUtf8("n2_text_label"))
        self.gridLayout.addWidget(self.n2_text_label, 3, 0, 1, 1)
        self.inclinaison_text = QtGui.QLabel(self.layoutWidget)
        self.inclinaison_text.setObjectName(_fromUtf8("inclinaison_text"))
        self.gridLayout.addWidget(self.inclinaison_text, 22, 0, 1, 1)
        self.n1_entry = QtGui.QLineEdit(self.layoutWidget)
        self.n1_entry.setObjectName(_fromUtf8("n1_entry"))
        self.gridLayout.addWidget(self.n1_entry, 1, 1, 1, 1)
        self.n2_entry = QtGui.QLineEdit(self.layoutWidget)
        self.n2_entry.setObjectName(_fromUtf8("n2_entry"))
        self.gridLayout.addWidget(self.n2_entry, 3, 1, 1, 1)
        self.cone_intersection_text_label = QtGui.QLabel(self.layoutWidget)
        self.cone_intersection_text_label.setObjectName(_fromUtf8("cone_intersection_text_label"))
        self.gridLayout.addWidget(self.cone_intersection_text_label, 18, 0, 1, 1)
        self.pushButton_intersections_plans = QtGui.QPushButton(self.layoutWidget)
        self.pushButton_intersections_plans.setObjectName(_fromUtf8("pushButton_intersections_plans"))
        self.gridLayout.addWidget(self.pushButton_intersections_plans, 4, 0, 1, 1)
        self.pushButton_intersection_proj = QtGui.QPushButton(self.layoutWidget)
        self.pushButton_intersection_proj.setObjectName(_fromUtf8("pushButton_intersection_proj"))
        self.gridLayout.addWidget(self.pushButton_intersection_proj, 14, 0, 1, 1)
        self.checkBox_2 = QtGui.QCheckBox(self.layoutWidget)
        self.checkBox_2.setObjectName(_fromUtf8("checkBox_2"))
        self.gridLayout.addWidget(self.checkBox_2, 14, 1, 1, 1)
        self.n_proj_label = QtGui.QLabel(self.layoutWidget)
        self.n_proj_label.setText(_fromUtf8(""))
        self.n_proj_label.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByMouse)
        self.n_proj_label.setObjectName(_fromUtf8("n_proj_label"))
        self.gridLayout.addWidget(self.n_proj_label, 15, 0, 1, 2)
        self.n_proj_entry = QtGui.QLineEdit(self.layoutWidget)
        self.n_proj_entry.setObjectName(_fromUtf8("n_proj_entry"))
        self.gridLayout.addWidget(self.n_proj_entry, 8, 1, 1, 1)
        self.n_cone_entry = QtGui.QLineEdit(self.layoutWidget)
        self.n_cone_entry.setObjectName(_fromUtf8("n_cone_entry"))
        self.gridLayout.addWidget(self.n_cone_entry, 17, 1, 1, 1)
        self.cone_plane_label = QtGui.QLabel(self.layoutWidget)
        self.cone_plane_label.setText(_fromUtf8(""))
        self.cone_plane_label.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByMouse)
        self.cone_plane_label.setObjectName(_fromUtf8("cone_plane_label"))
        self.gridLayout.addWidget(self.cone_plane_label, 24, 0, 1, 2)

        self.retranslateUi(Intersections)
        QtCore.QMetaObject.connectSlotsByName(Intersections)
        Intersections.setTabOrder(self.n1_entry, self.n2_entry)
        Intersections.setTabOrder(self.n2_entry, self.pushButton_intersections_plans)
        Intersections.setTabOrder(self.pushButton_intersections_plans, self.checkBox)
        Intersections.setTabOrder(self.checkBox, self.n_proj_entry)
        Intersections.setTabOrder(self.n_proj_entry, self.angle_proj_entry)
        Intersections.setTabOrder(self.angle_proj_entry, self.pushButton_intersection_proj)
        Intersections.setTabOrder(self.pushButton_intersection_proj, self.checkBox_2)
        Intersections.setTabOrder(self.checkBox_2, self.n_cone_entry)
        Intersections.setTabOrder(self.n_cone_entry, self.cone_entry)
        Intersections.setTabOrder(self.cone_entry, self.cone_angle_entry)
        Intersections.setTabOrder(self.cone_angle_entry, self.pushButton_intersection_cone)
        Intersections.setTabOrder(self.pushButton_intersection_cone, self.checkBox_3)

    def retranslateUi(self, Intersections):
        Intersections.setWindowTitle(_translate("Intersections", "Intersections", None))
        self.intersection_cone_text_label.setText(_translate("Intersections", "Intersection plane n, cone c", None))
        self.n_proj_text_label.setText(_translate("Intersections", "n", None))
        self.n1_text_label.setText(_translate("Intersections", "n1", None))
        self.n_cone_text_label.setText(_translate("Intersections", "n", None))
        self.intersection_text_label.setText(_translate("Intersections", "Intersection plane n1 and n2", None))
        self.pushButton_intersection_cone.setText(_translate("Intersections", "OK", None))
        self.checkBox.setText(_translate("Intersections", "directions", None))
        self.checkBox_3.setText(_translate("Intersections", "directions", None))
        self.intersection_proj_text_label.setText(_translate("Intersections", "Intersection projected direction, plane n", None))
        self.angle_text_label.setText(_translate("Intersections", "angle", None))
        self.n2_text_label.setText(_translate("Intersections", "n2", None))
        self.inclinaison_text.setText(_translate("Intersections", "inclination angle", None))
        self.cone_intersection_text_label.setText(_translate("Intersections", "c", None))
        self.pushButton_intersections_plans.setText(_translate("Intersections", "OK", None))
        self.pushButton_intersection_proj.setText(_translate("Intersections", "OK", None))
        self.checkBox_2.setText(_translate("Intersections", "directions", None))

