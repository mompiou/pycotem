# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'stereoprojUI.ui'
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

class Ui_StereoProj(object):
    def setupUi(self, StereoProj):
        StereoProj.setObjectName(_fromUtf8("StereoProj"))
        StereoProj.resize(1403, 879)
        self.centralwidget = QtGui.QWidget(StereoProj)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout_7 = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout_7.setObjectName(_fromUtf8("gridLayout_7"))
        self.mplvl = QtGui.QGridLayout()
        self.mplvl.setObjectName(_fromUtf8("mplvl"))
        self.gridLayout_7.addLayout(self.mplvl, 0, 0, 1, 1)
        self.groupBox_6 = QtGui.QGroupBox(self.centralwidget)
        self.groupBox_6.setMaximumSize(QtCore.QSize(460, 16777215))
        self.groupBox_6.setTitle(_fromUtf8(""))
        self.groupBox_6.setObjectName(_fromUtf8("groupBox_6"))
        self.gridLayout = QtGui.QGridLayout(self.groupBox_6)
        self.gridLayout.setContentsMargins(-1, 0, -1, 0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.groupBox = QtGui.QGroupBox(self.groupBox_6)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setMaximumSize(QtCore.QSize(230, 16777215))
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.gridLayout_2 = QtGui.QGridLayout(self.groupBox)
        self.gridLayout_2.setMargin(5)
        self.gridLayout_2.setSpacing(5)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.d_label_var = QtGui.QLabel(self.groupBox)
        self.d_label_var.setText(_fromUtf8(""))
        self.d_label_var.setObjectName(_fromUtf8("d_label_var"))
        self.gridLayout_2.addWidget(self.d_label_var, 10, 6, 1, 1)
        self.abc_label = QtGui.QLabel(self.groupBox)
        self.abc_label.setObjectName(_fromUtf8("abc_label"))
        self.gridLayout_2.addWidget(self.abc_label, 0, 0, 1, 1)
        self.e_label = QtGui.QLabel(self.groupBox)
        self.e_label.setObjectName(_fromUtf8("e_label"))
        self.gridLayout_2.addWidget(self.e_label, 4, 0, 1, 4)
        self.d_label = QtGui.QLabel(self.groupBox)
        self.d_label.setObjectName(_fromUtf8("d_label"))
        self.gridLayout_2.addWidget(self.d_label, 10, 0, 1, 1)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem, 11, 4, 1, 1)
        self.space_group_Box = QtGui.QComboBox(self.groupBox)
        self.space_group_Box.setObjectName(_fromUtf8("space_group_Box"))
        self.gridLayout_2.addWidget(self.space_group_Box, 7, 0, 1, 6)
        self.alphabetagamma_label = QtGui.QLabel(self.groupBox)
        self.alphabetagamma_label.setObjectName(_fromUtf8("alphabetagamma_label"))
        self.gridLayout_2.addWidget(self.alphabetagamma_label, 2, 0, 1, 1)
        self.d_entry = QtGui.QLineEdit(self.groupBox)
        self.d_entry.setObjectName(_fromUtf8("d_entry"))
        self.gridLayout_2.addWidget(self.d_entry, 10, 4, 1, 1)
        self.e_entry = QtGui.QLineEdit(self.groupBox)
        self.e_entry.setObjectName(_fromUtf8("e_entry"))
        self.gridLayout_2.addWidget(self.e_entry, 4, 4, 1, 1)
        self.abc_entry = QtGui.QLineEdit(self.groupBox)
        self.abc_entry.setObjectName(_fromUtf8("abc_entry"))
        self.gridLayout_2.addWidget(self.abc_entry, 0, 3, 1, 3)
        self.dm_button = QtGui.QPushButton(self.groupBox)
        self.dm_button.setObjectName(_fromUtf8("dm_button"))
        self.gridLayout_2.addWidget(self.dm_button, 10, 2, 1, 1)
        self.reciprocal_checkBox = QtGui.QCheckBox(self.groupBox)
        self.reciprocal_checkBox.setObjectName(_fromUtf8("reciprocal_checkBox"))
        self.gridLayout_2.addWidget(self.reciprocal_checkBox, 5, 0, 1, 5)
        self.dp_button = QtGui.QPushButton(self.groupBox)
        self.dp_button.setObjectName(_fromUtf8("dp_button"))
        self.gridLayout_2.addWidget(self.dp_button, 10, 5, 1, 1)
        self.alphabetagamma_entry = QtGui.QLineEdit(self.groupBox)
        self.alphabetagamma_entry.setObjectName(_fromUtf8("alphabetagamma_entry"))
        self.gridLayout_2.addWidget(self.alphabetagamma_entry, 2, 3, 1, 3)
        self.gridLayout.addWidget(self.groupBox, 0, 0, 1, 1)
        self.groupBox_2 = QtGui.QGroupBox(self.groupBox_6)
        self.groupBox_2.setEnabled(True)
        self.groupBox_2.setMaximumSize(QtCore.QSize(230, 16777215))
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.gridLayout_3 = QtGui.QGridLayout(self.groupBox_2)
        self.gridLayout_3.setMargin(5)
        self.gridLayout_3.setSpacing(5)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.rot_gm_button = QtGui.QPushButton(self.groupBox_2)
        self.rot_gm_button.setObjectName(_fromUtf8("rot_gm_button"))
        self.gridLayout_3.addWidget(self.rot_gm_button, 9, 0, 1, 1)
        spacerItem1 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem1, 11, 1, 1, 1)
        self.diff_entry = QtGui.QLineEdit(self.groupBox_2)
        self.diff_entry.setObjectName(_fromUtf8("diff_entry"))
        self.gridLayout_3.addWidget(self.diff_entry, 0, 1, 1, 3)
        self.diff_label = QtGui.QLabel(self.groupBox_2)
        self.diff_label.setObjectName(_fromUtf8("diff_label"))
        self.gridLayout_3.addWidget(self.diff_label, 0, 0, 1, 1)
        self.tilt_label = QtGui.QLabel(self.groupBox_2)
        self.tilt_label.setObjectName(_fromUtf8("tilt_label"))
        self.gridLayout_3.addWidget(self.tilt_label, 1, 0, 1, 1)
        self.inclinaison_entry = QtGui.QLineEdit(self.groupBox_2)
        self.inclinaison_entry.setObjectName(_fromUtf8("inclinaison_entry"))
        self.gridLayout_3.addWidget(self.inclinaison_entry, 4, 1, 1, 1)
        self.rot_g_entry = QtGui.QLineEdit(self.groupBox_2)
        self.rot_g_entry.setObjectName(_fromUtf8("rot_g_entry"))
        self.gridLayout_3.addWidget(self.rot_g_entry, 9, 1, 1, 1)
        self.inclination_label = QtGui.QLabel(self.groupBox_2)
        self.inclination_label.setObjectName(_fromUtf8("inclination_label"))
        self.gridLayout_3.addWidget(self.inclination_label, 4, 0, 1, 1)
        self.rot_diff_label = QtGui.QLabel(self.groupBox_2)
        self.rot_diff_label.setObjectName(_fromUtf8("rot_diff_label"))
        self.gridLayout_3.addWidget(self.rot_diff_label, 8, 0, 1, 3)
        self.button_trace = QtGui.QPushButton(self.groupBox_2)
        self.button_trace.setObjectName(_fromUtf8("button_trace"))
        self.gridLayout_3.addWidget(self.button_trace, 5, 0, 1, 2)
        self.rot_gp_button = QtGui.QPushButton(self.groupBox_2)
        self.rot_gp_button.setObjectName(_fromUtf8("rot_gp_button"))
        self.gridLayout_3.addWidget(self.rot_gp_button, 9, 2, 1, 1)
        self.tilt_entry = QtGui.QLineEdit(self.groupBox_2)
        self.tilt_entry.setObjectName(_fromUtf8("tilt_entry"))
        self.gridLayout_3.addWidget(self.tilt_entry, 1, 1, 1, 3)
        self.rg_label = QtGui.QLabel(self.groupBox_2)
        self.rg_label.setText(_fromUtf8(""))
        self.rg_label.setObjectName(_fromUtf8("rg_label"))
        self.gridLayout_3.addWidget(self.rg_label, 10, 1, 1, 1)
        self.gridLayout.addWidget(self.groupBox_2, 2, 0, 1, 1)
        self.groupBox_5 = QtGui.QGroupBox(self.groupBox_6)
        self.groupBox_5.setMaximumSize(QtCore.QSize(200, 16777215))
        self.groupBox_5.setObjectName(_fromUtf8("groupBox_5"))
        self.gridLayout_6 = QtGui.QGridLayout(self.groupBox_5)
        self.gridLayout_6.setObjectName(_fromUtf8("gridLayout_6"))
        self.coord_label = QtGui.QLabel(self.groupBox_5)
        self.coord_label.setText(_fromUtf8(""))
        self.coord_label.setObjectName(_fromUtf8("coord_label"))
        self.gridLayout_6.addWidget(self.coord_label, 5, 0, 1, 3)
        self.lab_euler2 = QtGui.QLabel(self.groupBox_5)
        self.lab_euler2.setObjectName(_fromUtf8("lab_euler2"))
        self.gridLayout_6.addWidget(self.lab_euler2, 2, 0, 1, 3)
        self.button_trace2 = QtGui.QPushButton(self.groupBox_5)
        self.button_trace2.setObjectName(_fromUtf8("button_trace2"))
        self.gridLayout_6.addWidget(self.button_trace2, 1, 0, 1, 3)
        self.lab_coord = QtGui.QLabel(self.groupBox_5)
        self.lab_coord.setObjectName(_fromUtf8("lab_coord"))
        self.gridLayout_6.addWidget(self.lab_coord, 4, 0, 1, 3)
        self.phi1phiphi2_entry = QtGui.QLineEdit(self.groupBox_5)
        self.phi1phiphi2_entry.setObjectName(_fromUtf8("phi1phiphi2_entry"))
        self.gridLayout_6.addWidget(self.phi1phiphi2_entry, 0, 0, 1, 3)
        self.angle_euler_label = QtGui.QLabel(self.groupBox_5)
        self.angle_euler_label.setText(_fromUtf8(""))
        self.angle_euler_label.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByMouse)
        self.angle_euler_label.setObjectName(_fromUtf8("angle_euler_label"))
        self.gridLayout_6.addWidget(self.angle_euler_label, 3, 0, 1, 3)
        spacerItem2 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout_6.addItem(spacerItem2, 6, 0, 1, 1)
        self.gridLayout.addWidget(self.groupBox_5, 2, 1, 1, 1)
        self.groupBox_9 = QtGui.QGroupBox(self.groupBox_6)
        self.groupBox_9.setMaximumSize(QtCore.QSize(230, 16777215))
        self.groupBox_9.setObjectName(_fromUtf8("groupBox_9"))
        self.gridLayout_10 = QtGui.QGridLayout(self.groupBox_9)
        self.gridLayout_10.setObjectName(_fromUtf8("gridLayout_10"))
        self.style_box = QtGui.QCheckBox(self.groupBox_9)
        self.style_box.setObjectName(_fromUtf8("style_box"))
        self.gridLayout_10.addWidget(self.style_box, 2, 0, 1, 3)
        self.size_var_label = QtGui.QLabel(self.groupBox_9)
        self.size_var_label.setObjectName(_fromUtf8("size_var_label"))
        self.gridLayout_10.addWidget(self.size_var_label, 5, 0, 1, 3)
        self.color_trace_rouge = QtGui.QRadioButton(self.groupBox_9)
        self.color_trace_rouge.setObjectName(_fromUtf8("color_trace_rouge"))
        self.gridLayout_10.addWidget(self.color_trace_rouge, 0, 3, 1, 1)
        self.uvw_button = QtGui.QCheckBox(self.groupBox_9)
        self.uvw_button.setObjectName(_fromUtf8("uvw_button"))
        self.gridLayout_10.addWidget(self.uvw_button, 9, 3, 1, 1)
        self.color_trace_vert = QtGui.QRadioButton(self.groupBox_9)
        self.color_trace_vert.setObjectName(_fromUtf8("color_trace_vert"))
        self.gridLayout_10.addWidget(self.color_trace_vert, 0, 1, 1, 2)
        self.real_space_checkBox = QtGui.QCheckBox(self.groupBox_9)
        self.real_space_checkBox.setObjectName(_fromUtf8("real_space_checkBox"))
        self.gridLayout_10.addWidget(self.real_space_checkBox, 3, 2, 1, 2)
        self.size_var = QtGui.QLineEdit(self.groupBox_9)
        self.size_var.setObjectName(_fromUtf8("size_var"))
        self.gridLayout_10.addWidget(self.size_var, 5, 3, 1, 1)
        self.text_size_label = QtGui.QLabel(self.groupBox_9)
        self.text_size_label.setObjectName(_fromUtf8("text_size_label"))
        self.gridLayout_10.addWidget(self.text_size_label, 6, 0, 1, 2)
        self.wulff_button = QtGui.QCheckBox(self.groupBox_9)
        self.wulff_button.setObjectName(_fromUtf8("wulff_button"))
        self.gridLayout_10.addWidget(self.wulff_button, 3, 0, 1, 2)
        self.tilt_angle_entry = QtGui.QLineEdit(self.groupBox_9)
        self.tilt_angle_entry.setObjectName(_fromUtf8("tilt_angle_entry"))
        self.gridLayout_10.addWidget(self.tilt_angle_entry, 7, 3, 1, 1)
        self.text_size_entry = QtGui.QLineEdit(self.groupBox_9)
        self.text_size_entry.setObjectName(_fromUtf8("text_size_entry"))
        self.gridLayout_10.addWidget(self.text_size_entry, 6, 3, 1, 1)
        self.color_trace_bleu = QtGui.QRadioButton(self.groupBox_9)
        self.color_trace_bleu.setObjectName(_fromUtf8("color_trace_bleu"))
        self.gridLayout_10.addWidget(self.color_trace_bleu, 0, 0, 1, 1)
        self.reset_view_button = QtGui.QPushButton(self.groupBox_9)
        self.reset_view_button.setObjectName(_fromUtf8("reset_view_button"))
        self.gridLayout_10.addWidget(self.reset_view_button, 10, 0, 1, 4)
        self.image_angle_entry = QtGui.QLineEdit(self.groupBox_9)
        self.image_angle_entry.setObjectName(_fromUtf8("image_angle_entry"))
        self.gridLayout_10.addWidget(self.image_angle_entry, 8, 3, 1, 1)
        self.label_2 = QtGui.QLabel(self.groupBox_9)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout_10.addWidget(self.label_2, 8, 0, 1, 3)
        self.tilt_angle_label = QtGui.QLabel(self.groupBox_9)
        self.tilt_angle_label.setObjectName(_fromUtf8("tilt_angle_label"))
        self.gridLayout_10.addWidget(self.tilt_angle_label, 7, 0, 1, 3)
        self.hexa_button = QtGui.QCheckBox(self.groupBox_9)
        self.hexa_button.setObjectName(_fromUtf8("hexa_button"))
        self.gridLayout_10.addWidget(self.hexa_button, 9, 0, 1, 2)
        spacerItem3 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout_10.addItem(spacerItem3, 11, 0, 1, 1)
        self.gridLayout.addWidget(self.groupBox_9, 1, 0, 1, 1)
        self.groupBox_4 = QtGui.QGroupBox(self.groupBox_6)
        self.groupBox_4.setMaximumSize(QtCore.QSize(200, 16777215))
        self.groupBox_4.setObjectName(_fromUtf8("groupBox_4"))
        self.gridLayout_5 = QtGui.QGridLayout(self.groupBox_4)
        self.gridLayout_5.setObjectName(_fromUtf8("gridLayout_5"))
        self.trace_plan_sym_button = QtGui.QPushButton(self.groupBox_4)
        self.trace_plan_sym_button.setObjectName(_fromUtf8("trace_plan_sym_button"))
        self.gridLayout_5.addWidget(self.trace_plan_sym_button, 4, 0, 1, 3)
        self.sym_button = QtGui.QPushButton(self.groupBox_4)
        self.sym_button.setObjectName(_fromUtf8("sym_button"))
        self.gridLayout_5.addWidget(self.sym_button, 2, 0, 1, 3)
        self.undo_trace_cone_button = QtGui.QPushButton(self.groupBox_4)
        self.undo_trace_cone_button.setObjectName(_fromUtf8("undo_trace_cone_button"))
        self.gridLayout_5.addWidget(self.undo_trace_cone_button, 6, 3, 1, 1)
        self.trace_plan_button = QtGui.QPushButton(self.groupBox_4)
        self.trace_plan_button.setObjectName(_fromUtf8("trace_plan_button"))
        self.gridLayout_5.addWidget(self.trace_plan_button, 3, 0, 1, 3)
        self.undo_trace_plan_sym_button = QtGui.QPushButton(self.groupBox_4)
        self.undo_trace_plan_sym_button.setObjectName(_fromUtf8("undo_trace_plan_sym_button"))
        self.gridLayout_5.addWidget(self.undo_trace_plan_sym_button, 4, 3, 1, 1)
        self.inclination_entry = QtGui.QLineEdit(self.groupBox_4)
        self.inclination_entry.setObjectName(_fromUtf8("inclination_entry"))
        self.gridLayout_5.addWidget(self.inclination_entry, 6, 2, 1, 1)
        self.trace_cone_button = QtGui.QPushButton(self.groupBox_4)
        self.trace_cone_button.setObjectName(_fromUtf8("trace_cone_button"))
        self.gridLayout_5.addWidget(self.trace_cone_button, 6, 0, 1, 2)
        self.undo_trace_schmid = QtGui.QPushButton(self.groupBox_4)
        self.undo_trace_schmid.setObjectName(_fromUtf8("undo_trace_schmid"))
        self.gridLayout_5.addWidget(self.undo_trace_schmid, 5, 3, 1, 1)
        self.norm_button = QtGui.QPushButton(self.groupBox_4)
        self.norm_button.setObjectName(_fromUtf8("norm_button"))
        self.gridLayout_5.addWidget(self.norm_button, 7, 0, 1, 1)
        self.pole_entry = QtGui.QLineEdit(self.groupBox_4)
        self.pole_entry.setObjectName(_fromUtf8("pole_entry"))
        self.gridLayout_5.addWidget(self.pole_entry, 0, 0, 1, 4)
        self.undo_trace_plan_button = QtGui.QPushButton(self.groupBox_4)
        self.undo_trace_plan_button.setObjectName(_fromUtf8("undo_trace_plan_button"))
        self.gridLayout_5.addWidget(self.undo_trace_plan_button, 3, 3, 1, 1)
        self.undo_addpole_button = QtGui.QPushButton(self.groupBox_4)
        self.undo_addpole_button.setObjectName(_fromUtf8("undo_addpole_button"))
        self.gridLayout_5.addWidget(self.undo_addpole_button, 1, 3, 1, 1)
        self.addpole_button = QtGui.QPushButton(self.groupBox_4)
        self.addpole_button.setObjectName(_fromUtf8("addpole_button"))
        self.gridLayout_5.addWidget(self.addpole_button, 1, 0, 1, 3)
        self.dhkl_label = QtGui.QLabel(self.groupBox_4)
        self.dhkl_label.setText(_fromUtf8(""))
        self.dhkl_label.setObjectName(_fromUtf8("dhkl_label"))
        self.gridLayout_5.addWidget(self.dhkl_label, 7, 2, 1, 2)
        self.undo_sym_button = QtGui.QPushButton(self.groupBox_4)
        self.undo_sym_button.setObjectName(_fromUtf8("undo_sym_button"))
        self.gridLayout_5.addWidget(self.undo_sym_button, 2, 3, 1, 1)
        self.trace_schmid_button = QtGui.QPushButton(self.groupBox_4)
        self.trace_schmid_button.setObjectName(_fromUtf8("trace_schmid_button"))
        self.gridLayout_5.addWidget(self.trace_schmid_button, 5, 0, 1, 3)
        spacerItem4 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout_5.addItem(spacerItem4, 8, 2, 1, 1)
        self.gridLayout.addWidget(self.groupBox_4, 1, 1, 1, 1)
        self.groupBox_3 = QtGui.QGroupBox(self.groupBox_6)
        self.groupBox_3.setMaximumSize(QtCore.QSize(200, 16777215))
        self.groupBox_3.setObjectName(_fromUtf8("groupBox_3"))
        self.gridLayout_4 = QtGui.QGridLayout(self.groupBox_3)
        self.gridLayout_4.setMargin(5)
        self.gridLayout_4.setSpacing(5)
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        self.angle_beta_entry = QtGui.QLineEdit(self.groupBox_3)
        self.angle_beta_entry.setObjectName(_fromUtf8("angle_beta_entry"))
        self.gridLayout_4.addWidget(self.angle_beta_entry, 8, 2, 1, 1)
        self.angle_beta_label = QtGui.QLabel(self.groupBox_3)
        self.angle_beta_label.setObjectName(_fromUtf8("angle_beta_label"))
        self.gridLayout_4.addWidget(self.angle_beta_label, 8, 0, 1, 1)
        self.angle_z_buttonm = QtGui.QPushButton(self.groupBox_3)
        self.angle_z_buttonm.setObjectName(_fromUtf8("angle_z_buttonm"))
        self.gridLayout_4.addWidget(self.angle_z_buttonm, 11, 1, 1, 1)
        self.angle_alpha_label_2 = QtGui.QLabel(self.groupBox_3)
        self.angle_alpha_label_2.setText(_fromUtf8(""))
        self.angle_alpha_label_2.setObjectName(_fromUtf8("angle_alpha_label_2"))
        self.gridLayout_4.addWidget(self.angle_alpha_label_2, 2, 5, 1, 1)
        self.angle_alpha_entry = QtGui.QLineEdit(self.groupBox_3)
        self.angle_alpha_entry.setObjectName(_fromUtf8("angle_alpha_entry"))
        self.gridLayout_4.addWidget(self.angle_alpha_entry, 2, 2, 1, 1)
        self.angle_beta_buttonp = QtGui.QPushButton(self.groupBox_3)
        self.angle_beta_buttonp.setObjectName(_fromUtf8("angle_beta_buttonp"))
        self.gridLayout_4.addWidget(self.angle_beta_buttonp, 8, 3, 1, 1)
        self.label = QtGui.QLabel(self.groupBox_3)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout_4.addWidget(self.label, 0, 4, 1, 2)
        self.angle_beta_buttonm = QtGui.QPushButton(self.groupBox_3)
        self.angle_beta_buttonm.setObjectName(_fromUtf8("angle_beta_buttonm"))
        self.gridLayout_4.addWidget(self.angle_beta_buttonm, 8, 1, 1, 1)
        self.angle_z_label_2 = QtGui.QLabel(self.groupBox_3)
        self.angle_z_label_2.setText(_fromUtf8(""))
        self.angle_z_label_2.setObjectName(_fromUtf8("angle_z_label_2"))
        self.gridLayout_4.addWidget(self.angle_z_label_2, 11, 5, 1, 1)
        self.angle_z_label = QtGui.QLabel(self.groupBox_3)
        self.angle_z_label.setObjectName(_fromUtf8("angle_z_label"))
        self.gridLayout_4.addWidget(self.angle_z_label, 11, 0, 1, 1)
        self.angle_z_entry = QtGui.QLineEdit(self.groupBox_3)
        self.angle_z_entry.setObjectName(_fromUtf8("angle_z_entry"))
        self.gridLayout_4.addWidget(self.angle_z_entry, 11, 2, 1, 1)
        self.lock_checkButton = QtGui.QCheckBox(self.groupBox_3)
        self.lock_checkButton.setObjectName(_fromUtf8("lock_checkButton"))
        self.gridLayout_4.addWidget(self.lock_checkButton, 0, 0, 1, 4)
        self.angle_alpha_buttonm = QtGui.QPushButton(self.groupBox_3)
        self.angle_alpha_buttonm.setObjectName(_fromUtf8("angle_alpha_buttonm"))
        self.gridLayout_4.addWidget(self.angle_alpha_buttonm, 2, 1, 1, 1)
        self.beta_signBox = QtGui.QCheckBox(self.groupBox_3)
        self.beta_signBox.setText(_fromUtf8(""))
        self.beta_signBox.setObjectName(_fromUtf8("beta_signBox"))
        self.gridLayout_4.addWidget(self.beta_signBox, 8, 4, 1, 1)
        self.angle_z_buttonp = QtGui.QPushButton(self.groupBox_3)
        self.angle_z_buttonp.setObjectName(_fromUtf8("angle_z_buttonp"))
        self.gridLayout_4.addWidget(self.angle_z_buttonp, 11, 3, 1, 1)
        self.alpha_signBox = QtGui.QCheckBox(self.groupBox_3)
        self.alpha_signBox.setText(_fromUtf8(""))
        self.alpha_signBox.setObjectName(_fromUtf8("alpha_signBox"))
        self.gridLayout_4.addWidget(self.alpha_signBox, 2, 4, 1, 1)
        self.theta_signBox = QtGui.QCheckBox(self.groupBox_3)
        self.theta_signBox.setText(_fromUtf8(""))
        self.theta_signBox.setObjectName(_fromUtf8("theta_signBox"))
        self.gridLayout_4.addWidget(self.theta_signBox, 11, 4, 1, 1)
        self.angle_alpha_buttonp = QtGui.QPushButton(self.groupBox_3)
        self.angle_alpha_buttonp.setObjectName(_fromUtf8("angle_alpha_buttonp"))
        self.gridLayout_4.addWidget(self.angle_alpha_buttonp, 2, 3, 1, 1)
        self.angle_beta_label_2 = QtGui.QLabel(self.groupBox_3)
        self.angle_beta_label_2.setText(_fromUtf8(""))
        self.angle_beta_label_2.setObjectName(_fromUtf8("angle_beta_label_2"))
        self.gridLayout_4.addWidget(self.angle_beta_label_2, 8, 5, 1, 1)
        self.angle_alpha_label = QtGui.QLabel(self.groupBox_3)
        self.angle_alpha_label.setObjectName(_fromUtf8("angle_alpha_label"))
        self.gridLayout_4.addWidget(self.angle_alpha_label, 2, 0, 1, 1)
        self.reset_angle_button = QtGui.QPushButton(self.groupBox_3)
        self.reset_angle_button.setObjectName(_fromUtf8("reset_angle_button"))
        self.gridLayout_4.addWidget(self.reset_angle_button, 12, 4, 1, 2)
        self.gridLayout.addWidget(self.groupBox_3, 0, 1, 1, 1)
        self.gridLayout_7.addWidget(self.groupBox_6, 0, 1, 1, 1)
        StereoProj.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(StereoProj)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1403, 23))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuSave = QtGui.QMenu(self.menubar)
        self.menuSave.setObjectName(_fromUtf8("menuSave"))
        self.menuStructure = QtGui.QMenu(self.menubar)
        self.menuStructure.setObjectName(_fromUtf8("menuStructure"))
        self.menuAngle = QtGui.QMenu(self.menubar)
        self.menuAngle.setObjectName(_fromUtf8("menuAngle"))
        self.menuSchmid_factor = QtGui.QMenu(self.menubar)
        self.menuSchmid_factor.setObjectName(_fromUtf8("menuSchmid_factor"))
        self.menuXyz_directions = QtGui.QMenu(self.menubar)
        self.menuXyz_directions.setObjectName(_fromUtf8("menuXyz_directions"))
        self.menuWidth = QtGui.QMenu(self.menubar)
        self.menuWidth.setObjectName(_fromUtf8("menuWidth"))
        self.menuIntersections = QtGui.QMenu(self.menubar)
        self.menuIntersections.setObjectName(_fromUtf8("menuIntersections"))
        self.menuDiffraction = QtGui.QMenu(self.menubar)
        self.menuDiffraction.setObjectName(_fromUtf8("menuDiffraction"))
        self.menuList = QtGui.QMenu(self.menubar)
        self.menuList.setObjectName(_fromUtf8("menuList"))
        StereoProj.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(StereoProj)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        StereoProj.setStatusBar(self.statusbar)
        self.actionSave_figure = QtGui.QAction(StereoProj)
        self.actionSave_figure.setObjectName(_fromUtf8("actionSave_figure"))
        self.actionCalculate_Schmid_factor = QtGui.QAction(StereoProj)
        self.actionCalculate_Schmid_factor.setObjectName(_fromUtf8("actionCalculate_Schmid_factor"))
        self.actionCalculate_angle = QtGui.QAction(StereoProj)
        self.actionCalculate_angle.setObjectName(_fromUtf8("actionCalculate_angle"))
        self.actionCalculate_xyz = QtGui.QAction(StereoProj)
        self.actionCalculate_xyz.setObjectName(_fromUtf8("actionCalculate_xyz"))
        self.actionCalculate_apparent_width = QtGui.QAction(StereoProj)
        self.actionCalculate_apparent_width.setObjectName(_fromUtf8("actionCalculate_apparent_width"))
        self.actionPlanes = QtGui.QAction(StereoProj)
        self.actionPlanes.setObjectName(_fromUtf8("actionPlanes"))
        self.actionProj_directions = QtGui.QAction(StereoProj)
        self.actionProj_directions.setObjectName(_fromUtf8("actionProj_directions"))
        self.actionPlane_cone = QtGui.QAction(StereoProj)
        self.actionPlane_cone.setObjectName(_fromUtf8("actionPlane_cone"))
        self.actionCalculate_intersections = QtGui.QAction(StereoProj)
        self.actionCalculate_intersections.setObjectName(_fromUtf8("actionCalculate_intersections"))
        self.actionHkl_uvw = QtGui.QAction(StereoProj)
        self.actionHkl_uvw.setObjectName(_fromUtf8("actionHkl_uvw"))
        self.actionPlot_Kikuchi_lines = QtGui.QAction(StereoProj)
        self.actionPlot_Kikuchi_lines.setObjectName(_fromUtf8("actionPlot_Kikuchi_lines"))
        self.actionShow_list_of_poles_directions = QtGui.QAction(StereoProj)
        self.actionShow_list_of_poles_directions.setObjectName(_fromUtf8("actionShow_list_of_poles_directions"))
        self.menuSave.addAction(self.actionSave_figure)
        self.menuAngle.addAction(self.actionCalculate_angle)
        self.menuSchmid_factor.addAction(self.actionCalculate_Schmid_factor)
        self.menuXyz_directions.addAction(self.actionCalculate_xyz)
        self.menuXyz_directions.addAction(self.actionHkl_uvw)
        self.menuWidth.addAction(self.actionCalculate_apparent_width)
        self.menuIntersections.addAction(self.actionCalculate_intersections)
        self.menuDiffraction.addAction(self.actionPlot_Kikuchi_lines)
        self.menuList.addAction(self.actionShow_list_of_poles_directions)
        self.menubar.addAction(self.menuSave.menuAction())
        self.menubar.addAction(self.menuStructure.menuAction())
        self.menubar.addAction(self.menuAngle.menuAction())
        self.menubar.addAction(self.menuSchmid_factor.menuAction())
        self.menubar.addAction(self.menuXyz_directions.menuAction())
        self.menubar.addAction(self.menuList.menuAction())
        self.menubar.addAction(self.menuWidth.menuAction())
        self.menubar.addAction(self.menuIntersections.menuAction())
        self.menubar.addAction(self.menuDiffraction.menuAction())

        self.retranslateUi(StereoProj)
        QtCore.QMetaObject.connectSlotsByName(StereoProj)
        StereoProj.setTabOrder(self.abc_entry, self.alphabetagamma_entry)
        StereoProj.setTabOrder(self.alphabetagamma_entry, self.e_entry)
        StereoProj.setTabOrder(self.e_entry, self.reciprocal_checkBox)
        StereoProj.setTabOrder(self.reciprocal_checkBox, self.space_group_Box)
        StereoProj.setTabOrder(self.space_group_Box, self.dm_button)
        StereoProj.setTabOrder(self.dm_button, self.d_entry)
        StereoProj.setTabOrder(self.d_entry, self.dp_button)
        StereoProj.setTabOrder(self.dp_button, self.color_trace_bleu)
        StereoProj.setTabOrder(self.color_trace_bleu, self.color_trace_vert)
        StereoProj.setTabOrder(self.color_trace_vert, self.color_trace_rouge)
        StereoProj.setTabOrder(self.color_trace_rouge, self.style_box)
        StereoProj.setTabOrder(self.style_box, self.wulff_button)
        StereoProj.setTabOrder(self.wulff_button, self.real_space_checkBox)
        StereoProj.setTabOrder(self.real_space_checkBox, self.size_var)
        StereoProj.setTabOrder(self.size_var, self.text_size_entry)
        StereoProj.setTabOrder(self.text_size_entry, self.tilt_angle_entry)
        StereoProj.setTabOrder(self.tilt_angle_entry, self.image_angle_entry)
        StereoProj.setTabOrder(self.image_angle_entry, self.hexa_button)
        StereoProj.setTabOrder(self.hexa_button, self.uvw_button)
        StereoProj.setTabOrder(self.uvw_button, self.reset_view_button)
        StereoProj.setTabOrder(self.reset_view_button, self.diff_entry)
        StereoProj.setTabOrder(self.diff_entry, self.tilt_entry)
        StereoProj.setTabOrder(self.tilt_entry, self.inclinaison_entry)
        StereoProj.setTabOrder(self.inclinaison_entry, self.button_trace)
        StereoProj.setTabOrder(self.button_trace, self.rot_gm_button)
        StereoProj.setTabOrder(self.rot_gm_button, self.rot_g_entry)
        StereoProj.setTabOrder(self.rot_g_entry, self.rot_gp_button)
        StereoProj.setTabOrder(self.rot_gp_button, self.lock_checkButton)
        StereoProj.setTabOrder(self.lock_checkButton, self.angle_alpha_buttonm)
        StereoProj.setTabOrder(self.angle_alpha_buttonm, self.angle_alpha_entry)
        StereoProj.setTabOrder(self.angle_alpha_entry, self.angle_alpha_buttonp)
        StereoProj.setTabOrder(self.angle_alpha_buttonp, self.alpha_signBox)
        StereoProj.setTabOrder(self.alpha_signBox, self.angle_beta_buttonm)
        StereoProj.setTabOrder(self.angle_beta_buttonm, self.angle_beta_entry)
        StereoProj.setTabOrder(self.angle_beta_entry, self.angle_beta_buttonp)
        StereoProj.setTabOrder(self.angle_beta_buttonp, self.beta_signBox)
        StereoProj.setTabOrder(self.beta_signBox, self.angle_z_buttonm)
        StereoProj.setTabOrder(self.angle_z_buttonm, self.angle_z_entry)
        StereoProj.setTabOrder(self.angle_z_entry, self.angle_z_buttonp)
        StereoProj.setTabOrder(self.angle_z_buttonp, self.theta_signBox)
        StereoProj.setTabOrder(self.theta_signBox, self.reset_angle_button)
        StereoProj.setTabOrder(self.reset_angle_button, self.pole_entry)
        StereoProj.setTabOrder(self.pole_entry, self.addpole_button)
        StereoProj.setTabOrder(self.addpole_button, self.undo_addpole_button)
        StereoProj.setTabOrder(self.undo_addpole_button, self.sym_button)
        StereoProj.setTabOrder(self.sym_button, self.undo_sym_button)
        StereoProj.setTabOrder(self.undo_sym_button, self.trace_plan_button)
        StereoProj.setTabOrder(self.trace_plan_button, self.undo_trace_plan_button)
        StereoProj.setTabOrder(self.undo_trace_plan_button, self.trace_plan_sym_button)
        StereoProj.setTabOrder(self.trace_plan_sym_button, self.undo_trace_plan_sym_button)
        StereoProj.setTabOrder(self.undo_trace_plan_sym_button, self.trace_schmid_button)
        StereoProj.setTabOrder(self.trace_schmid_button, self.undo_trace_schmid)
        StereoProj.setTabOrder(self.undo_trace_schmid, self.trace_cone_button)
        StereoProj.setTabOrder(self.trace_cone_button, self.inclination_entry)
        StereoProj.setTabOrder(self.inclination_entry, self.undo_trace_cone_button)
        StereoProj.setTabOrder(self.undo_trace_cone_button, self.norm_button)
        StereoProj.setTabOrder(self.norm_button, self.phi1phiphi2_entry)
        StereoProj.setTabOrder(self.phi1phiphi2_entry, self.button_trace2)

    def retranslateUi(self, StereoProj):
        StereoProj.setWindowTitle(_translate("StereoProj", "Stereo-Proj", None))
        self.groupBox.setTitle(_translate("StereoProj", "Crystal Parameters", None))
        self.abc_label.setText(_translate("StereoProj", "a,b,c", None))
        self.e_label.setText(_translate("StereoProj", "max indices", None))
        self.d_label.setText(_translate("StereoProj", "d", None))
        self.alphabetagamma_label.setText(_translate("StereoProj", "<p>&alpha;, &beta;, &gamma;</p>", None))
        self.dm_button.setText(_translate("StereoProj", "-", None))
        self.reciprocal_checkBox.setText(_translate("StereoProj", "Reciprocal indices", None))
        self.dp_button.setText(_translate("StereoProj", "+", None))
        self.groupBox_2.setTitle(_translate("StereoProj", "Axis/Rotation", None))
        self.rot_gm_button.setText(_translate("StereoProj", "-", None))
        self.diff_label.setText(_translate("StereoProj", "g-vector", None))
        self.tilt_label.setText(_translate("StereoProj", "Tilt (α,β,z)", None))
        self.inclination_label.setText(_translate("StereoProj", "Inclination", None))
        self.rot_diff_label.setText(_translate("StereoProj", "Rotation along g", None))
        self.button_trace.setText(_translate("StereoProj", "PLOT", None))
        self.rot_gp_button.setText(_translate("StereoProj", "+", None))
        self.groupBox_5.setTitle(_translate("StereoProj", "Euler Angles", None))
        self.lab_euler2.setText(_translate("StereoProj", "φ 1 , Φ , φ2", None))
        self.button_trace2.setText(_translate("StereoProj", "PLOT", None))
        self.lab_coord.setText(_translate("StereoProj", "Tilt, Inclination", None))
        self.groupBox_9.setTitle(_translate("StereoProj", "Layout", None))
        self.style_box.setText(_translate("StereoProj", "open/filled", None))
        self.size_var_label.setText(_translate("StereoProj", "Marker size", None))
        self.color_trace_rouge.setText(_translate("StereoProj", "red", None))
        self.uvw_button.setText(_translate("StereoProj", "uvw", None))
        self.color_trace_vert.setText(_translate("StereoProj", "green", None))
        self.real_space_checkBox.setText(_translate("StereoProj", "img/diff", None))
        self.text_size_label.setText(_translate("StereoProj", "Text size", None))
        self.wulff_button.setText(_translate("StereoProj", "Wulff net", None))
        self.color_trace_bleu.setText(_translate("StereoProj", "blue", None))
        self.reset_view_button.setText(_translate("StereoProj", "Update/Reset view", None))
        self.label_2.setText(_translate("StereoProj", "img α-tilt/y angle", None))
        self.tilt_angle_label.setText(_translate("StereoProj", "diff α-tilt/y angle", None))
        self.hexa_button.setText(_translate("StereoProj", "hexa", None))
        self.groupBox_4.setTitle(_translate("StereoProj", "Pole/Plane", None))
        self.trace_plan_sym_button.setText(_translate("StereoProj", "Sym Plane", None))
        self.sym_button.setText(_translate("StereoProj", "Symmetry", None))
        self.undo_trace_cone_button.setText(_translate("StereoProj", "-", None))
        self.trace_plan_button.setText(_translate("StereoProj", " Plane", None))
        self.undo_trace_plan_sym_button.setText(_translate("StereoProj", "-", None))
        self.trace_cone_button.setText(_translate("StereoProj", "Cone", None))
        self.undo_trace_schmid.setText(_translate("StereoProj", "-", None))
        self.norm_button.setText(_translate("StereoProj", "dhkl", None))
        self.undo_trace_plan_button.setText(_translate("StereoProj", "-", None))
        self.undo_addpole_button.setText(_translate("StereoProj", "-", None))
        self.addpole_button.setText(_translate("StereoProj", "Add", None))
        self.undo_sym_button.setText(_translate("StereoProj", "-", None))
        self.trace_schmid_button.setText(_translate("StereoProj", "Schmid", None))
        self.groupBox_3.setTitle(_translate("StereoProj", "Rotation", None))
        self.angle_beta_label.setText(_translate("StereoProj", "<html><head/><body><p>β</p></body></html>", None))
        self.angle_z_buttonm.setText(_translate("StereoProj", "-", None))
        self.angle_beta_buttonp.setText(_translate("StereoProj", "+", None))
        self.label.setText(_translate("StereoProj", "AC", None))
        self.angle_beta_buttonm.setText(_translate("StereoProj", "-", None))
        self.angle_z_label.setText(_translate("StereoProj", "<html><head/><body><p>θ</p></body></html>", None))
        self.lock_checkButton.setText(_translate("StereoProj", "Lock Axes", None))
        self.angle_alpha_buttonm.setText(_translate("StereoProj", "-", None))
        self.angle_z_buttonp.setText(_translate("StereoProj", "+", None))
        self.angle_alpha_buttonp.setText(_translate("StereoProj", "+", None))
        self.angle_alpha_label.setText(_translate("StereoProj", "<html><head/><body><p>α</p></body></html>", None))
        self.reset_angle_button.setText(_translate("StereoProj", "Reset", None))
        self.menuSave.setTitle(_translate("StereoProj", "Save", None))
        self.menuStructure.setTitle(_translate("StereoProj", "Structure", None))
        self.menuAngle.setTitle(_translate("StereoProj", "Angle", None))
        self.menuSchmid_factor.setTitle(_translate("StereoProj", "Schmid factor", None))
        self.menuXyz_directions.setTitle(_translate("StereoProj", "xyz directions", None))
        self.menuWidth.setTitle(_translate("StereoProj", "Width", None))
        self.menuIntersections.setTitle(_translate("StereoProj", "intersections", None))
        self.menuDiffraction.setTitle(_translate("StereoProj", "diffraction", None))
        self.menuList.setTitle(_translate("StereoProj", "list", None))
        self.actionSave_figure.setText(_translate("StereoProj", "Save figure", None))
        self.actionCalculate_Schmid_factor.setText(_translate("StereoProj", "calculate Schmid factor", None))
        self.actionCalculate_angle.setText(_translate("StereoProj", "Calculate angle", None))
        self.actionCalculate_xyz.setText(_translate("StereoProj", "calculate xyz directions", None))
        self.actionCalculate_apparent_width.setText(_translate("StereoProj", "Calculate apparent width", None))
        self.actionPlanes.setText(_translate("StereoProj", "planes", None))
        self.actionProj_directions.setText(_translate("StereoProj", "proj. directions", None))
        self.actionPlane_cone.setText(_translate("StereoProj", "plane-cone", None))
        self.actionCalculate_intersections.setText(_translate("StereoProj", "Calculate intersections", None))
        self.actionHkl_uvw.setText(_translate("StereoProj", "hkl <> uvw", None))
        self.actionPlot_Kikuchi_lines.setText(_translate("StereoProj", "plot Kikuchi lines or diffraction pattern", None))
        self.actionShow_list_of_poles_directions.setText(_translate("StereoProj", "show list of poles/directions", None))

