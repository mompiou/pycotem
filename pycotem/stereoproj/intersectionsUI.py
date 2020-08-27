# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'intersectionsUI.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Intersections(object):
    def setupUi(self, Intersections):
        Intersections.setObjectName("Intersections")
        Intersections.resize(545, 687)
        self.layoutWidget = QtWidgets.QWidget(Intersections)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 531, 671))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.intersection_cone_text_label = QtWidgets.QLabel(self.layoutWidget)
        self.intersection_cone_text_label.setObjectName("intersection_cone_text_label")
        self.gridLayout.addWidget(self.intersection_cone_text_label, 16, 0, 1, 1)
        self.n_proj_text_label = QtWidgets.QLabel(self.layoutWidget)
        self.n_proj_text_label.setObjectName("n_proj_text_label")
        self.gridLayout.addWidget(self.n_proj_text_label, 8, 0, 1, 1)
        self.n1n2_label = QtWidgets.QLabel(self.layoutWidget)
        self.n1n2_label.setText("")
        self.n1n2_label.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByMouse)
        self.n1n2_label.setObjectName("n1n2_label")
        self.gridLayout.addWidget(self.n1n2_label, 5, 0, 1, 2)
        self.cone_entry = QtWidgets.QLineEdit(self.layoutWidget)
        self.cone_entry.setObjectName("cone_entry")
        self.gridLayout.addWidget(self.cone_entry, 18, 1, 1, 1)
        self.cone_angle_entry = QtWidgets.QLineEdit(self.layoutWidget)
        self.cone_angle_entry.setObjectName("cone_angle_entry")
        self.gridLayout.addWidget(self.cone_angle_entry, 22, 1, 1, 1)
        self.n1_text_label = QtWidgets.QLabel(self.layoutWidget)
        self.n1_text_label.setObjectName("n1_text_label")
        self.gridLayout.addWidget(self.n1_text_label, 1, 0, 1, 1)
        self.angle_proj_entry = QtWidgets.QLineEdit(self.layoutWidget)
        self.angle_proj_entry.setObjectName("angle_proj_entry")
        self.gridLayout.addWidget(self.angle_proj_entry, 11, 1, 1, 1)
        self.n_cone_text_label = QtWidgets.QLabel(self.layoutWidget)
        self.n_cone_text_label.setObjectName("n_cone_text_label")
        self.gridLayout.addWidget(self.n_cone_text_label, 17, 0, 1, 1)
        self.intersection_text_label = QtWidgets.QLabel(self.layoutWidget)
        self.intersection_text_label.setObjectName("intersection_text_label")
        self.gridLayout.addWidget(self.intersection_text_label, 0, 0, 1, 1)
        self.pushButton_intersection_cone = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_intersection_cone.setObjectName("pushButton_intersection_cone")
        self.gridLayout.addWidget(self.pushButton_intersection_cone, 23, 0, 1, 1)
        self.checkBox = QtWidgets.QCheckBox(self.layoutWidget)
        self.checkBox.setObjectName("checkBox")
        self.gridLayout.addWidget(self.checkBox, 4, 1, 1, 1)
        self.checkBox_3 = QtWidgets.QCheckBox(self.layoutWidget)
        self.checkBox_3.setObjectName("checkBox_3")
        self.gridLayout.addWidget(self.checkBox_3, 23, 1, 1, 1)
        self.intersection_proj_text_label = QtWidgets.QLabel(self.layoutWidget)
        self.intersection_proj_text_label.setObjectName("intersection_proj_text_label")
        self.gridLayout.addWidget(self.intersection_proj_text_label, 7, 0, 1, 2)
        self.angle_text_label = QtWidgets.QLabel(self.layoutWidget)
        self.angle_text_label.setObjectName("angle_text_label")
        self.gridLayout.addWidget(self.angle_text_label, 11, 0, 1, 1)
        self.n2_text_label = QtWidgets.QLabel(self.layoutWidget)
        self.n2_text_label.setObjectName("n2_text_label")
        self.gridLayout.addWidget(self.n2_text_label, 3, 0, 1, 1)
        self.inclinaison_text = QtWidgets.QLabel(self.layoutWidget)
        self.inclinaison_text.setObjectName("inclinaison_text")
        self.gridLayout.addWidget(self.inclinaison_text, 22, 0, 1, 1)
        self.n1_entry = QtWidgets.QLineEdit(self.layoutWidget)
        self.n1_entry.setObjectName("n1_entry")
        self.gridLayout.addWidget(self.n1_entry, 1, 1, 1, 1)
        self.n2_entry = QtWidgets.QLineEdit(self.layoutWidget)
        self.n2_entry.setObjectName("n2_entry")
        self.gridLayout.addWidget(self.n2_entry, 3, 1, 1, 1)
        self.cone_intersection_text_label = QtWidgets.QLabel(self.layoutWidget)
        self.cone_intersection_text_label.setObjectName("cone_intersection_text_label")
        self.gridLayout.addWidget(self.cone_intersection_text_label, 18, 0, 1, 1)
        self.pushButton_intersections_plans = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_intersections_plans.setObjectName("pushButton_intersections_plans")
        self.gridLayout.addWidget(self.pushButton_intersections_plans, 4, 0, 1, 1)
        self.pushButton_intersection_proj = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_intersection_proj.setObjectName("pushButton_intersection_proj")
        self.gridLayout.addWidget(self.pushButton_intersection_proj, 14, 0, 1, 1)
        self.checkBox_2 = QtWidgets.QCheckBox(self.layoutWidget)
        self.checkBox_2.setObjectName("checkBox_2")
        self.gridLayout.addWidget(self.checkBox_2, 14, 1, 1, 1)
        self.n_proj_label = QtWidgets.QLabel(self.layoutWidget)
        self.n_proj_label.setText("")
        self.n_proj_label.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByMouse)
        self.n_proj_label.setObjectName("n_proj_label")
        self.gridLayout.addWidget(self.n_proj_label, 15, 0, 1, 2)
        self.n_proj_entry = QtWidgets.QLineEdit(self.layoutWidget)
        self.n_proj_entry.setObjectName("n_proj_entry")
        self.gridLayout.addWidget(self.n_proj_entry, 8, 1, 1, 1)
        self.n_cone_entry = QtWidgets.QLineEdit(self.layoutWidget)
        self.n_cone_entry.setObjectName("n_cone_entry")
        self.gridLayout.addWidget(self.n_cone_entry, 17, 1, 1, 1)
        self.cone_plane_label = QtWidgets.QLabel(self.layoutWidget)
        self.cone_plane_label.setText("")
        self.cone_plane_label.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByMouse)
        self.cone_plane_label.setObjectName("cone_plane_label")
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
        _translate = QtCore.QCoreApplication.translate
        Intersections.setWindowTitle(_translate("Intersections", "Intersections"))
        self.intersection_cone_text_label.setText(_translate("Intersections", "Intersection plane n, cone c"))
        self.n_proj_text_label.setText(_translate("Intersections", "n"))
        self.n1_text_label.setText(_translate("Intersections", "n1"))
        self.n_cone_text_label.setText(_translate("Intersections", "n"))
        self.intersection_text_label.setText(_translate("Intersections", "Intersection plane n1 and n2"))
        self.pushButton_intersection_cone.setText(_translate("Intersections", "OK"))
        self.checkBox.setText(_translate("Intersections", "directions"))
        self.checkBox_3.setText(_translate("Intersections", "directions"))
        self.intersection_proj_text_label.setText(_translate("Intersections", "Intersection projected direction, plane n"))
        self.angle_text_label.setText(_translate("Intersections", "angle"))
        self.n2_text_label.setText(_translate("Intersections", "n2"))
        self.inclinaison_text.setText(_translate("Intersections", "inclination angle"))
        self.cone_intersection_text_label.setText(_translate("Intersections", "c"))
        self.pushButton_intersections_plans.setText(_translate("Intersections", "OK"))
        self.pushButton_intersection_proj.setText(_translate("Intersections", "OK"))
        self.checkBox_2.setText(_translate("Intersections", "directions"))
