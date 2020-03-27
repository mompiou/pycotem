######################################################
#
# Interface is a script to solve interface planes from TEM BF/DF images at
# different tilts. It uses the approach developped by R.X. Xie from Tsinghua University
#
#######################################################


from __future__ import division
import numpy as np
from PyQt4 import QtGui, QtCore
import sys
import os
from PIL import Image
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib import pyplot as plt
import interfaceUI
import DrawInterfaceUI

######################
#
# click event control. Prevent to define a point when zooming
#
######################


def onpress(event):
    global press, move
    press = True


def onmove(event):
    global press, move
    if press:
        move = True


def onrelease(event):
    global press, move

    if press and not move:
        click(event)
    press = False
    move = False


def click(event):
    global gclick, s, minx, maxx, miny, maxy

    a = figure.add_subplot(111)
    minx, maxx = a.get_xlim()
    miny, maxy = a.get_ylim()
    x = event.xdata
    y = event.ydata
    r = str(np.around(x, decimals=2)) + ',' + str(np.around(y, decimals=2))
    a.annotate(str(s), (x, y))
    a.plot(x, y, 'b+')
    a.axis('off')
    a.figure.canvas.draw()
    s = s + 1
    gclick = np.vstack((gclick, np.array([x, y])))

    return gclick, s


#########################
#
# Reset points
#
############################


def reset_points():
    global image_diff, gclick, s, minx, maxx, miny, maxy

    a = figure.add_subplot(111)
    a.figure.clear()
    a = figure.add_subplot(111)
    img = Image.open(str(image_diff))
    img = np.array(img)
    figure.suptitle(str(image_diff))
    a.imshow(img, origin="upper")
    a.axis([minx, maxx, miny, maxy])
    a.axis('off')
    a.figure.canvas.draw()
    s = 1
    gclick = np.zeros((1, 2))

    return s, gclick

######################
#
# reset view
#
#########################


def reset():
    global image_diff, gclick, s, minx, maxx, miny, maxy

    a = figure.add_subplot(111)
    a.figure.clear()
    a = figure.add_subplot(111)
    img = Image.open(str(image_diff))
    img = np.array(img)
    figure.suptitle(str(image_diff))
    a.imshow(img, origin="upper")
    minx = 0
    maxx = width
    miny = height
    maxy = 0
    a.axis([minx, maxx, miny, maxy])
    a.axis('off')
    a.figure.canvas.draw()
    gclick = np.zeros((1, 2))
    # ui.conditions_Listbox.clear()
    ui.euler_Listbox.clear()
    s = 1
    return s, gclick


########################
#
# Add/Remove spot
#
#########################

def add_condition():
    global gclick, d, counter, x_calib
    x1 = gclick[1, 0]
    y1 = gclick[1, 1]
    x2 = gclick[2, 0]
    y2 = gclick[2, 1]
    if ui.direction_button.isChecked() is False:
        x3 = gclick[3, 0]
        y3 = gclick[3, 1]
        d = np.abs((y2 - y1) * x3 - (x2 - x1) * y3 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    else:
        d = 0

    d12 = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    if d12 == 0:
        bet = 0
    else:
        if x2 - x1 > 0:
            bet = 180 - np.arccos((y2 - y1) / (d12)) * 180 / np.pi
        else:
            bet = -180 + np.arccos((y2 - y1) / (d12)) * 180 / np.pi

    s3 = ui.tilt_y_entry.text()
    s4 = ui.tilt_x_entry.text()
    s5 = ui.tilt_z_entry.text()

    ii = 0
    while ii < len(x_micro):
        if x_micro[ii][0] == ui.micro_box.currentText():
            if ui.magnification_entry.text() == x_micro[ii][1]:
                s1 = d * float(x_micro[ii][2])
                s2 = bet + float(x_micro[ii][3])
        ii = ii + 1

    s = str(np.around(s1, decimals=3)) + ',' + str(np.around(s2, decimals=3)) + ',' + s3 + ',' + s4 + ',' + s5
    ui.conditions_Listbox.addItem(s)


def remove_condition():
    ui.conditions_Listbox.takeItem(ui.conditions_Listbox.currentRow())

######################################################
#
# Determine orientation from a set of at least 3 diff
#
####################################################


def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def Rot(th, a, b, c):
    th = th * np.pi / 180
    aa = a / np.linalg.norm([a, b, c])
    bb = b / np.linalg.norm([a, b, c])
    cc = c / np.linalg.norm([a, b, c])
    c1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    c2 = np.array([[aa**2, aa * bb, aa * cc], [bb * aa, bb**2, bb * cc], [cc * aa,
                                                                          cc * bb, cc**2]], float)
    c3 = np.array([[0, -cc, bb], [cc, 0, -aa], [-bb, aa, 0]], float)
    R = np.cos(th) * c1 + (1 - np.cos(th)) * c2 + np.sin(th) * c3

    return R


def rotation(phi1, phi, phi2):
    phi1 = phi1 * np.pi / 180
    phi = phi * np.pi / 180
    phi2 = phi2 * np.pi / 180
    R = np.array([[np.cos(phi1) * np.cos(phi2) - np.cos(phi) * np.sin(phi1) * np.sin(phi2),
                   -np.cos(phi) * np.cos(phi2) * np.sin(phi1) - np.cos(phi1) *
                   np.sin(phi2), np.sin(phi) * np.sin(phi1)], [np.cos(phi2) * np.sin(phi1)
                                                               + np.cos(phi) * np.cos(phi1) * np.sin(phi2), np.cos(phi) * np.cos(phi1)
                                                               * np.cos(phi2) - np.sin(phi1) * np.sin(phi2), -np.cos(phi1) * np.sin(phi)],
                  [np.sin(phi) * np.sin(phi2), np.cos(phi2) * np.sin(phi), np.cos(phi)]], float)
    return R

#################
#
# Crystal
#
#################


def cryst():

    abc = ui.abc_entry.text().split(",")
    a = np.float(abc[0])
    b = np.float(abc[1])
    c = np.float(abc[2])
    alphabetagamma = ui.alphabetagamma_entry.text().split(",")
    alpha = np.float(alphabetagamma[0]) * np.pi / 180
    beta = np.float(alphabetagamma[1]) * np.pi / 180
    gamma = np.float(alphabetagamma[2]) * np.pi / 180
    V = a * b * c * np.sqrt(1 - (np.cos(alpha)**2) - (np.cos(beta))**2 - (np.cos(gamma))**2 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma))
    D = np.array([[a, b * np.cos(gamma), c * np.cos(beta)], [0, b * np.sin(gamma), c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)], [0, 0, V / (a * b * np.sin(gamma))]])
    M = np.transpose(D)
    return M


def tilt_axes():
    global s_a, s_b, s_z
    s_a, s_b, s_z = -1, -1, -1
    if ui.alpha_signBox.isChecked():
        s_a = 1
    if ui.beta_signBox.isChecked():
        s_b = 1
    if ui.theta_signBox.isChecked():
        s_b = 1
    return s_a, s_b, s_z

##############################################
#
# Get normal:
# Convention in holder coordinates: y is the primary tilt angle fixed in the holder frame. x is the secondary tilt fixed in the sample frame. z is the tilt axis fixed in the sample frame.
# B is the beam direction. The sample is tilted along y by -alpha, then tilted along x or z by -beta/-z angle. The projected trace is referenced with respect to the y direction
# The trace is solved by SVD using [BxTp]T=0
# The normal is solved using pseudo inverse [B,T]dn=[sign w sqrt(1-BT^2)]
# Residual errors are computed using the boostrap resampling
#
##################################################


def bootstrap_norm(data1, data2, nor, num_samples, alpha):
    n = len(data1)
    idx = np.random.randint(0, n, (num_samples, n))
    idx[:, -1] = n - 1
    samples1 = data1[idx]
    samples2 = data2[idx]
    sumvec = np.array([0, 0, 0])
    dev = []
    w = []
    nor = nor.reshape(3,)
    for i in range(0, num_samples):
        vec = np.dot(np.linalg.pinv(samples1[i]), samples2[i])
        vec = vec.reshape(3,)
        sumvec = vec.T + sumvec
        w = np.append(w, np.linalg.norm(vec))
        vec = vec / np.linalg.norm(vec)
        dev = np.append(dev, np.arccos(np.dot(nor, vec)) * 180 / np.pi)

    dev = np.nan_to_num(dev)
    stat = np.sort(dev, axis=None)
    w = np.nan_to_num(w)
    stat_w = np.sort(w, axis=None)

    return (sumvec / num_samples, stat[int(alpha * num_samples / 100)], np.abs(0.5 * (stat_w[int((100 - alpha) * num_samples / 100)] - stat_w[int(alpha * num_samples / 100)])))


def bootstrap_dir(data1, nor, num_samples, alpha):
    n = len(data1)
    idx = np.random.randint(0, n, (num_samples, n))
    idx[:, -1] = n - 1
    samples1 = data1[idx]
    sumvec = np.array([0, 0, 0])
    dev = []
    nor = nor.reshape(3,)
    for i in range(0, num_samples):
        u, s, v = np.linalg.svd(samples1[i].T)
        vec = u[:, 2]
        sumvec = vec.T + sumvec
        vec = vec / np.linalg.norm(vec)
        dev = np.append(dev, np.arccos(np.dot(nor, vec)) * 180 / np.pi)

    dev = np.nan_to_num(dev)
    stat = np.sort(dev, axis=None)

    return (sumvec / num_samples, stat[int(alpha * num_samples / 100)])


def get():
    if ui.direction_button.isChecked():
        get_direction()
    else:
        get_normal()


def get_direction():
    ui.euler_Listbox.clear()
    s_a, s_b, s_z = tilt_axes()
    s = [str(x.text()) for x in ui.conditions_Listbox.selectedItems()]
    if s < 3:
        ui.euler_Listbox.addItem("Number of inputs should be more than 3")
    else:
        tilt_a = []
        tilt_b = []
        tilt_z = []
        inclination = []

        for i in range(0, len(s)):
            l = map(float, s[i].split(','))
            tilt_a.append(l[2])
            tilt_b.append(l[3])
            tilt_z.append(l[4])
            inclination.append(l[1])

        inclination = np.array(inclination)
        tilt_a = np.array(tilt_a)
        tilt_b = np.array(tilt_b)
        tilt_z = np.array(tilt_z)

        B = np.zeros((3, np.shape(tilt_a)[0]))
        BxTp = np.zeros((3, np.shape(tilt_a)[0]))

        t_ang = np.float(ui.image_angle_entry.text())

        for i in range(0, tilt_a.shape[0]):
            R = np.dot(Rot(s_z * tilt_z[i], 0, 0, 1), np.dot(Rot(s_b * tilt_b[i], 1, 0, 0), Rot(s_a * tilt_a[i], 0, 1, 0)))
            B[:, i] = np.dot(R, np.array([0, 0, 1]))
            ny = inclination[i] * np.pi / 180
            t = np.array([-np.cos(ny), np.sin(ny), 0])
            BxTp[:, i] = np.dot(R, np.dot(Rot(t_ang, 0, 0, 1), t))
            BxTp[:, i] = BxTp[:, i] / np.linalg.norm(BxTp[:, i])

        u, s, v = np.linalg.svd(BxTp)
        T = u[:, 2]
        T_mean, dev_angle = bootstrap_dir(BxTp.T, T / np.linalg.norm(T), 10000, 95)
        T = T_mean
        if ui.crystal_checkBox.isChecked():
            euler = ui.euler_entry.text().split(",")
            phi1 = np.float(euler[0])
            phi = np.float(euler[1])
            phi2 = np.float(euler[2])
            M = cryst()
            T_cc = np.dot(np.linalg.inv(rotation(phi1, phi, phi2)), T / np.linalg.norm(T)).T
            T_cc = np.dot(M, T_cc.T)

        else:
            T_cc = T

        np.set_printoptions(suppress=True)
        ui.euler_Listbox.addItem('Mean direction')
        ui.euler_Listbox.addItem(str(np.around(T_cc[0], decimals=3)) + ',' + str(np.around(T_cc[1], decimals=3)) + ',' + str(np.around(T_cc[2], decimals=3)))
        ui.euler_Listbox.addItem('95% confidence interval (deg)')
        ui.euler_Listbox.addItem(str(np.around(dev_angle, decimals=3)))


def get_normal():

    ui.euler_Listbox.clear()
    s_a, s_b, s_z = tilt_axes()
    s = [str(x.text()) for x in ui.conditions_Listbox.selectedItems()]
    if s < 3:
        ui.euler_Listbox.addItem("Number of spots should be between 3 and 5")
    else:
        tilt_a = []
        tilt_b = []
        tilt_z = []
        width_inclination = []

        for i in range(0, len(s)):
            l = map(float, s[i].split(','))
            tilt_a.append(l[2])
            tilt_b.append(l[3])
            tilt_z.append(l[4])
            width_inclination.append(l[0:2])

        width_inclination = np.array(width_inclination)
        tilt_a = np.array(tilt_a)
        tilt_b = np.array(tilt_b)
        tilt_z = np.array(tilt_z)

        B = np.zeros((3, np.shape(tilt_a)[0]))
        BxTp = np.zeros((3, np.shape(tilt_a)[0]))

        t_ang = np.float(ui.image_angle_entry.text())

        for i in range(0, tilt_a.shape[0]):
            R = np.dot(Rot(s_z * tilt_z[i], 0, 0, 1), np.dot(Rot(s_b * tilt_b[i], 1, 0, 0), Rot(s_a * tilt_a[i], 0, 1, 0)))
            B[:, i] = np.dot(R, np.array([0, 0, 1]))
            ny = width_inclination[i, 1] * np.pi / 180
            t = np.array([-np.cos(ny), np.sin(ny), 0])
            BxTp[:, i] = np.dot(R, np.dot(Rot(t_ang, 0, 0, 1), t))
            BxTp[:, i] = BxTp[:, i] / np.linalg.norm(BxTp[:, i])

        u, s, v = np.linalg.svd(BxTp)
        T = u[:, 2]
        T_mean, dev_angle_t = bootstrap_dir(BxTp.T, T / np.linalg.norm(T), 10000, 95)
        # T=T_mean
        N = tilt_a.shape[0]
        shuffle = np.zeros((2**(N - 1) + 1, N))
        B = np.vstack((B.T, T))
        res = np.zeros(2**(N - 1))

        for i in range(0, 2**(N - 1)):
            b = np.zeros((1, N + 1))
            shuffle[i, :] = np.array(list(np.binary_repr(i, width=N)), dtype=int) * -1
            shuffle[shuffle == 0] = 1  # sgn(i)
            for j in range(0, N):
                b[:, j] = width_inclination[j, 0] * shuffle[i, j] * np.sqrt(1 - np.dot(B[j, :], T)**2)
            n = np.dot(np.linalg.pinv(B), b.T)
            res[i] = np.linalg.norm(np.dot(B, n) - b.T)

        mini = np.argmin(res)

        for j in range(0, N):
            b[:, j] = width_inclination[j, 0] * shuffle[mini, j] * np.sqrt(1 - np.dot(B[j, :], T)**2)
        n = np.dot(np.linalg.pinv(B), b.T)
        n, dev_angle_n, dev_width = bootstrap_norm(B, b.T, n / np.linalg.norm(n), 10000, 95)

        dn = n / np.linalg.norm(n)
        thick = np.linalg.norm(n) * np.sqrt(1 - dn[2]**2)
        dev_thick = dev_width * np.sqrt(1 - dn[2]**2)
        # in crystal coordinate
        if ui.crystal_checkBox.isChecked():
            euler = ui.euler_entry.text().split(",")
            phi1 = np.float(euler[0])
            phi = np.float(euler[1])
            phi2 = np.float(euler[2])
            M = cryst()
            n_cc = np.dot(np.linalg.inv(rotation(phi1, phi, phi2)), n / np.linalg.norm(n)).T
            T_cc = np.dot(np.linalg.inv(rotation(phi1, phi, phi2)), T / np.linalg.norm(T)).T
            n_cc = np.dot(M, n_cc.T)
            T_cc = np.dot(M, T_cc.T)

        else:
            n_cc = dn
            T_cc = T

        np.set_printoptions(suppress=True)
        ui.euler_Listbox.addItem('Plane normal')
        ui.euler_Listbox.addItem(str(np.around(n_cc[0], decimals=3)) + ',' + str(np.around(n_cc[1], decimals=3)) + ',' + str(np.around(n_cc[2], decimals=3)))
        ui.euler_Listbox.addItem('Trace')
        ui.euler_Listbox.addItem(str(np.around(T_cc[0], decimals=3)) + ',' + str(np.around(T_cc[1], decimals=3)) + ',' + str(np.around(T_cc[2], decimals=3)))
        ui.euler_Listbox.addItem('Error on trace direction (deg)')
        ui.euler_Listbox.addItem(str(np.around(dev_angle_t, decimals=3)))
        ui.euler_Listbox.addItem('Error on plane normal (deg)')
        ui.euler_Listbox.addItem(str(np.around(dev_angle_n, decimals=3)))
        ui.euler_Listbox.addItem('Plane width')
        ui.euler_Listbox.addItem(str(np.around(np.linalg.norm(n), decimals=3)))
        ui.euler_Listbox.addItem('Error on plane width')
        ui.euler_Listbox.addItem(str(np.around(dev_width, decimals=3)))
        ui.euler_Listbox.addItem('Estimated thickness')
        ui.euler_Listbox.addItem(str(np.around(thick, decimals=3)) + '+/-' + str(np.around(dev_thick, decimals=3)))

##############################
#
# Draw planes or directions
#
################################


def draw_planes_dir():
    global gclick, minx, miny, maxx, maxy

    t = np.float(ui_draw.thickness_entry.text())
    t_ang = np.float(ui.image_angle_entry.text())
    tilt_x = np.float(ui.tilt_x_entry.text())
    tilt_y = np.float(ui.tilt_y_entry.text())
    tilt_z = np.float(ui.tilt_z_entry.text())

    ii = 0
    while ii < len(x_micro):
        if x_micro[ii][0] == ui.micro_box.currentText():
            if ui.magnification_entry.text() == x_micro[ii][1]:
                mag_conv = float(x_micro[ii][2])

        ii = ii + 1

    nd = ui_draw.plane_entry.text().split(",")
    nd = np.array([np.float(nd[0]), np.float(nd[1]), np.float(nd[2])])
    if ui.crystal_checkBox.isChecked():
        euler = ui.euler_entry.text().split(",")
        phi1 = np.float(euler[0])
        phi = np.float(euler[1])
        phi2 = np.float(euler[2])
    else:
        phi1 = 0
        phi = 0
        phi2 = 0
    M = cryst()
    D = np.transpose(M)
    Dstar = np.transpose(np.linalg.inv(D))
    x = gclick[1, 0]
    y = gclick[1, 1]
    a = figure.add_subplot(111)
    minx, maxx = a.get_xlim()
    miny, maxy = a.get_ylim()

    if ui_draw.surf_checkBox.isChecked():
        s = ui_draw.surface_entry.text().split(",")
        s = np.array([np.float(s[0]), np.float(s[1]), np.float(s[2])])
        if ui_draw.dir_checkBox.isChecked():
            s = np.dot(D, s)
        else:
            s = np.dot(Dstar, s)

        s = np.dot(rotation(phi1, phi, phi2), s)
        s = s / np.linalg.norm(s)

    else:
        s = np.array([0, 0, 1])

    if ui_draw.dir_checkBox.isChecked():
        dire = np.dot(D, nd)
        dire = np.dot(rotation(phi1, phi, phi2), dire)
        dire = dire / np.linalg.norm(dire)
        dire = np.dot(Rot(tilt_y, 0, 1, 0), np.dot(Rot(tilt_x, 1, 0, 0), np.dot(Rot(tilt_z, 0, 0, 1), np.dot(Rot(t_ang, 0, 0, 1), dire))))
        d_proj = (dire / np.dot(dire, s) - s) * t / mag_conv

        a.plot([x, x + d_proj[0]], [y, y - d_proj[1]], 'r-')
        a.axis('off')
        if ui_draw.label_checkBox.isChecked():
            st = str(np.float(nd[0])) + ',' + str(np.float(nd[1])) + ',' + str(np.float(nd[2]))
            a.annotate(st, (x + d_proj[0], y - d_proj[1]))
        a.figure.canvas.draw()

    else:
        plan = np.dot(Dstar, nd)
        plan = np.dot(rotation(phi1, phi, phi2), plan)
        plan = plan / np.linalg.norm(plan)
        plan = np.dot(Rot(tilt_y, 0, 1, 0), np.dot(Rot(tilt_x, 1, 0, 0), np.dot(Rot(tilt_z, 0, 0, 1), np.dot(Rot(t_ang, 0, 0, 1), plan))))

        if ui_draw.surf_checkBox.isChecked():
            s = ui_draw.surface_entry.text().split(",")
            s = np.array([np.float(s[0]), np.float(s[1]), np.float(s[2])])
            s = np.dot(Dstar, s)
            s = np.dot(rotation(phi1, phi, phi2), s)
            s = s / np.linalg.norm(s)
        else:
            s = np.array([0, 0, 1])

        s = np.dot(Rot(tilt_y, 0, 1, 0), np.dot(Rot(tilt_x, 1, 0, 0), np.dot(Rot(tilt_z, 0, 0, 1), np.dot(Rot(t_ang, 0, 0, 1), s))))
        T = np.cross(plan, s)
        T = T / np.linalg.norm(T)
        w = plan[2] * t / np.sqrt((1 - np.dot(plan, s)**2) * (1 - T[2]**2)) / mag_conv
        xw = x - w * T[1]
        yw = y - w * T[0]

        a.plot([x - 100 * T[0], x + 100 * T[0]], [y + 100 * T[1], y - 100 * T[1]], 'b-')
        a.plot([xw - 100 * T[0], xw + 100 * T[0]], [yw + 100 * T[1], yw - 100 * T[1]], 'g-')
        a.plot([x, xw], [y, yw], 'r-')
        a.axis('off')
        if ui_draw.label_checkBox.isChecked():
            st = str(np.float(nd[0])) + ',' + str(np.float(nd[1])) + ',' + str(np.float(nd[2]))
            angp = np.arctan2(T[1], T[0]) * 180 / np.pi
            a.annotate(st, (xw, yw), rotation=angp)
        a.figure.canvas.draw()


#########################
#
# Open image
#
############################


def open_image():
    global width, height, image_diff, s, gclick, minx, maxx, miny, maxy, press, move
    press = False
    move = False
    a = figure.add_subplot(111)
    a.figure.clear()
    a = figure.add_subplot(111)
    image_diff = QtGui.QFileDialog.getOpenFileName(Interface, "Open image file", "", "*.png *.jpg *.bmp *.tiff *.tif *.jpeg")
    img = Image.open(str(image_diff))
    img = np.array(img)
    a.imshow(img, origin='upper')
    figure.suptitle(str(image_diff))
    height, width = img.shape[0], img.shape[1]
    a.axis([0, width, height, 0])
    a.axis('off')
    a.figure.canvas.draw()
    s = 1
    gclick = np.zeros((1, 2))
    ui.euler_Listbox.clear()
    minx = 0
    maxx = width
    miny = height
    maxy = 0
    return s, gclick


##################################################
#
# Add matplotlib toolbar to zoom and pan
#
###################################################


class NavigationToolbar(NavigationToolbar):
    # only display the buttons we need
    toolitems = [t for t in NavigationToolbar.toolitems if
                 t[0] in ('Pan', 'Zoom')]

    def set_message(self, msg):
        pass

######################################################
# import crystal structures from un txt file Name,a,b,c,alpha,beta,gamma,space group
#
######################################################


def structure(item):
    global x0, e_entry
    ui.abc_entry.setText(str(item[1]) + ',' + str(item[2]) + ',' + str(item[3]))
    ui.alphabetagamma_entry.setText(str(item[4]) + ',' + str(item[5]) + ',' + str(item[6]))

##################
#
# Set tilting buttons
#
################


def single_tilt():
    ui.tilt_y_entry.setEnabled(True)
    ui.tilt_z_entry.setDisabled(True)
    ui.tilt_x_entry.setDisabled(True)
    ui.tilt_y_entry.setText('0')


def double_tilt():
    ui.tilt_z_entry.setDisabled(True)
    ui.tilt_x_entry.setEnabled(True)
    ui.tilt_y_entry.setEnabled(True)
    ui.tilt_y_entry.setText('0')
    ui.tilt_x_entry.setText('0')


def tilt_rot():
    ui.tilt_z_entry.setEnabled(True)
    ui.tilt_x_entry.setDisabled(True)
    ui.tilt_y_entry.setEnabled(True)
    ui.tilt_y_entry.setText('0')
    ui.tilt_z_entry.setText('0')


######################################################
#
# Launch
#
######################################################
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

if __name__ == "__main__":

    app = QtGui.QApplication(sys.argv)
    Interface = QtGui.QMainWindow()
    ui = interfaceUI.Ui_Interface()
    ui.setupUi(Interface)
    figure = plt.figure()
    canvas = FigureCanvas(figure)
    ui.mplvl.addWidget(canvas)
    toolbar = NavigationToolbar(canvas, canvas)
    toolbar.setMinimumWidth(601)

    Draw = QtGui.QDialog()
    ui_draw = DrawInterfaceUI.Ui_Draw_plane_directions()
    ui_draw.setupUi(Draw)
    Interface.connect(ui.actionDirections_planes, QtCore.SIGNAL('triggered()'), Draw.show)
    ui_draw.buttonBox.rejected.connect(Draw.close)
    ui_draw.buttonBox.accepted.connect(draw_planes_dir)

    single_tilt()
    ui.single_button.setChecked(True)
    ui.tilt_x_entry.setText('0')
    ui.tilt_z_entry.setText('0')
    ui.single_button.toggled.connect(single_tilt)
    ui.double_button.toggled.connect(double_tilt)
    ui.tilt_rot_button.toggled.connect(tilt_rot)
    ui.magnification_entry.setText('1')
    ui.image_angle_entry.setText('0')
    s = 1
    gclick = np.zeros((1, 2))

    file_struct = open(os.path.join(os.path.dirname(__file__), 'structure.txt'), "r")

    x0 = []

    for line in file_struct:
        x0.append(map(str, line.split()))

    i = 0
    file_struct.close()

    for item in x0:
        entry = ui.menuStructure.addAction(item[0])
        Interface.connect(entry, QtCore.SIGNAL('triggered()'), lambda item=item: structure(item))
        i = i + 1

#########################
#
# Get magnification and rotation angle between holder and image from the microscope.txt file
#
############################

    f_micro = open(os.path.join(os.path.dirname(__file__), 'microscope.txt'), "r")

    x_micro = []

    for line in f_micro:
        x_micro.append(map(str, line.split()))

    x_micro = np.array(x_micro)
    x_micro_unique, ind_x = np.unique(x_micro[:, 0], return_index=True)

    for i in ind_x[::-1]:
        ui.micro_box.addItem(x_micro[i][0])

    f_micro.close()

    Interface.connect(ui.actionSave_figure, QtCore.SIGNAL('triggered()'), open_image)

    figure.canvas.mpl_connect('button_press_event', onpress)
    figure.canvas.mpl_connect('button_release_event', onrelease)
    figure.canvas.mpl_connect('motion_notify_event', onmove)
    press = False
    move = False

    ui.Button_reset.clicked.connect(reset_points)
    ui.reset_view_button.clicked.connect(reset)

    ui.add_condition_button.clicked.connect(add_condition)
    ui.remove_condition_button.clicked.connect(remove_condition)
    ui.normal_button.clicked.connect(get)

    Interface.show()
    sys.exit(app.exec_())
