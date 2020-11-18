######################################################
#
# Interface is a script to solve interface planes from TEM BF/DF images at
# different tilts. It uses the approach developped by R.X. Xie from Tsinghua University
#
#######################################################


from __future__ import division
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import sys
import os
from PIL import Image
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib import pyplot as plt
from . import interfaceUI, DrawInterfaceUI, DrawStretchedUI

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
def reset_last_point():
    global image_diff, gclick, s, minx, maxx, miny, maxy

    a = figure.add_subplot(111)
    a.figure.clear()
    a = figure.add_subplot(111)
    img = Image.open(str(image_diff[0]))
    img = np.array(img)
    figure.suptitle(str(image_diff[0]))
    a.imshow(img, origin="upper")
    gclick = gclick[:-1, :]
    s = s - 1
    for i in range(1, gclick.shape[0]):
        a.annotate(str(i), (gclick[i, 0], gclick[i, 1]))
    a.plot(gclick[1:, 0], gclick[1:, 1], 'b+')

    a.axis([minx, maxx, miny, maxy])
    a.axis('off')
    a.figure.canvas.draw()

    return s, gclick


def reset_points():
    global image_diff, gclick, s, minx, maxx, miny, maxy, D_p

    a = figure.add_subplot(111)
    a.figure.clear()
    a = figure.add_subplot(111)
    img = Image.open(str(image_diff[0]))
    img = np.array(img)
    figure.suptitle(str(image_diff[0]))
    a.imshow(img, origin="upper")
    a.axis([minx, maxx, miny, maxy])
    a.axis('off')
    a.figure.canvas.draw()
    s = 1
    gclick = np.zeros((1, 2))
    D_p = np.zeros(9)
    return s, gclick

######################
#
# reset view
#
#########################


def reset():
    global image_diff, gclick, s, minx, maxx, miny, maxy, D_p

    a = figure.add_subplot(111)
    a.figure.clear()
    a = figure.add_subplot(111)
    img = Image.open(str(image_diff[0]))
    img = np.array(img)
    figure.suptitle(str(image_diff[0]))
    a.imshow(img, origin="upper")
    minx = 0
    maxx = width
    miny = height
    maxy = 0
    a.axis([minx, maxx, miny, maxy])
    D_p = np.zeros(9)
    for i in range(1, gclick.shape[0]):
        a.annotate(str(i), (gclick[i, 0], gclick[i, 1]))
    a.plot(gclick[1:, 0], gclick[1:, 1], 'b+')

    a.axis('off')
    a.figure.canvas.draw()
    ui.euler_Listbox.clear()

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
        d = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

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
    ss = ui.conditions_Listbox.count()
    item = ui.conditions_Listbox.item(ss - 1)
    item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)


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


def deviation_thick(angle, S):
    r = np.sqrt(S[0]**2 + S[1]**2 + S[2]**2)
    t = np.arctan2(S[1], S[0]) * 180 / np.pi
    ph = np.arccos(S[2] / r) * 180 / np.pi
    Aa = []
    for g in np.linspace(-np.pi, np.pi, 10):
        for j in range(1, 10):
            Aa = np.append(Aa, np.dot(Rot(t, 0, 0, 1), np.dot(Rot(ph, 0, 1, 0), np.array([np.sin(g) * np.sin(angle / j * np.pi / 180), np.cos(g) * np.sin(angle / j * np.pi / 180), np.cos(angle / j * np.pi / 180)])))[2])

    m = np.abs(S[2] - np.amax(Aa))
    return m


def get_direction():
    ui.euler_Listbox.clear()
    s_a, s_b, s_z = tilt_axes()
    s = [str(x.text()) for x in ui.conditions_Listbox.selectedItems()]
    if len(s) < 3:
        ui.euler_Listbox.addItem("Number of inputs should be more than 3")
    else:
        tilt_a = []
        tilt_b = []
        tilt_z = []
        inclination = []
        d = []

        for i in range(0, len(s)):
            l = list(map(float, s[i].split(',')))
            tilt_a.append(l[2])
            tilt_b.append(l[3])
            tilt_z.append(l[4])
            inclination.append(l[1])
            d.append(l[0])

        inclination = np.array(inclination)
        tilt_a = np.array(tilt_a)
        tilt_b = np.array(tilt_b)
        tilt_z = np.array(tilt_z)
        d = np.array(d)

        L = np.zeros((3, np.shape(tilt_a)[0]))
        BxTp = np.zeros((3, np.shape(tilt_a)[0]))
        t_ang = np.float(ui.image_angle_entry.text())

        for i in range(0, tilt_a.shape[0]):
            R = np.dot(Rot(s_z * tilt_z[i], 0, 0, 1), np.dot(Rot(s_b * tilt_b[i], 1, 0, 0), Rot(s_a * tilt_a[i], 0, 1, 0)))
            ny = inclination[i] * np.pi / 180
            t = np.array([-np.cos(ny), np.sin(ny), 0])
            BxTp[:, i] = np.dot(R, np.dot(Rot(t_ang, 0, 0, 1), t))
            BxTp[:, i] = BxTp[:, i] / np.linalg.norm(BxTp[:, i])
            L[:, i] = np.dot(R, np.dot(Rot(t_ang, 0, 0, 1), np.array([np.sin(ny), np.cos(ny), 0])))

        BxTp = np.hstack((BxTp, L)).T
        N = tilt_a.shape[0]
        shuffle = np.zeros((2**(N - 1) + 1, N))
        res = np.zeros(2**(N - 1))

        for i in range(0, 2**(N - 1)):
            bb = np.zeros((N))
            shuffle[i, :] = np.array(list(np.binary_repr(i, width=N)), dtype=int) * -1
            shuffle[shuffle == 0] = 1  # sgn(i)
            for j in range(0, N):
                bb[j] = d[j] * shuffle[i, j]

            b = np.hstack((np.zeros((tilt_a.shape[0])), bb))
            T = np.dot(np.linalg.pinv(BxTp), b.T)
            res[i] = np.linalg.norm(np.dot(BxTp, T) - b.T)

        mini = np.argmin(res)

        for j in range(0, N):
            bb[j] = d[j] * shuffle[mini, j]
        b = np.hstack((np.zeros((tilt_a.shape[0])), bb))
        T = np.dot(np.linalg.pinv(BxTp), b.T)
        T, dev_angle_T, dev_width = bootstrap_norm(BxTp, b.T, T / np.linalg.norm(T), 10000, 95)

        dT = T / np.linalg.norm(T)
        thick = np.abs(T[2])
        dev_t = deviation_thick(dev_angle_T, dT) * np.linalg.norm(T)
        dev_thick = (dev_t + dev_width) * np.abs(T[2])

        dev_thick = deviation_thick(dev_angle_T, dT) * (np.linalg.norm(T) + dev_width * np.abs(dT[2]))

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
        ui.euler_Listbox.addItem('Direction')
        ui.euler_Listbox.addItem(str(np.around(T_cc[0], decimals=3)) + ',' + str(np.around(T_cc[1], decimals=3)) + ',' + str(np.around(T_cc[2], decimals=3)))
        ui.euler_Listbox.addItem('Error on trace direction (deg)')
        ui.euler_Listbox.addItem(str(np.around(dev_angle_T, decimals=3)))
        ui.euler_Listbox.addItem('Direction length')
        ui.euler_Listbox.addItem(str(np.around(np.linalg.norm(T), decimals=3)))
        ui.euler_Listbox.addItem('Error on direction width')
        ui.euler_Listbox.addItem(str(np.around(dev_width, decimals=3)))
        ui.euler_Listbox.addItem('Estimated thickness')
        ui.euler_Listbox.addItem(str(np.around(thick, decimals=3)) + '+/-' + str(np.around(dev_thick, decimals=3)))


def get_normal():

    ui.euler_Listbox.clear()
    s_a, s_b, s_z = tilt_axes()
    s = [str(x.text()) for x in ui.conditions_Listbox.selectedItems()]
    if len(s) < 3:
        ui.euler_Listbox.addItem("Number of spots should be between 3 and 5")
    else:
        tilt_a = []
        tilt_b = []
        tilt_z = []
        width_inclination = []

        for i in range(0, len(s)):
            l = list(map(float, s[i].split(',')))
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
        dev_n = deviation_thick(dev_angle_n, dn) * np.linalg.norm(n)
        dev_thick = (dev_n + dev_width) * np.sqrt(1 - dn[2]**2)
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


def plot_dir_planes():
    global D_p, minx, miny, maxx, maxy, image_diff
    a = figure.add_subplot(111)
    a.figure.clear()
    a = figure.add_subplot(111)
    img = Image.open(str(image_diff[0]))
    img = np.array(img)
    figure.suptitle(str(image_diff[0]))
    a.imshow(img, origin="upper")
    a.axis([minx, maxx, miny, maxy])
    a.axis('off')

    for i in range(1, D_p.shape[0]):
        if D_p[i, 8] == 1:
            a.plot([D_p[i, 2], D_p[i, 2] + D_p[i, 0]], [D_p[i, 3], D_p[i, 3] - D_p[i, 1]], 'r-')
            if ui_draw.label_checkBox.isChecked():
                st = str(np.float(D_p[i, 5])) + ',' + str(np.float(D_p[i, 6])) + ',' + str(np.float(D_p[i, 7]))
                a.annotate(st, (D_p[i, 2] + D_p[i, 0], D_p[i, 3] - D_p[i, 1]))
            a.axis('off')

        else:
            xw = D_p[i, 3] - D_p[i, 2] * D_p[i, 1] / np.sqrt(D_p[i, 1]**2 + D_p[i, 0]**2)
            yw = D_p[i, 4] - D_p[i, 2] * D_p[i, 0] / np.sqrt(D_p[i, 1]**2 + D_p[i, 0]**2)

            a.plot([D_p[i, 3] - 100 * D_p[i, 0], D_p[i, 3] + 100 * D_p[i, 0]], [D_p[i, 4] + 100 * D_p[i, 1], D_p[i, 4] - 100 * D_p[i, 1]], 'b-')
            a.plot([xw - 100 * D_p[i, 0], xw + 100 * D_p[i, 0]], [yw + 100 * D_p[i, 1], yw - 100 * D_p[i, 1]], 'g-')
            a.plot([D_p[i, 3], xw], [D_p[i, 4], yw], 'r-')
            a.axis('off')

            if ui_draw.label_checkBox.isChecked():
                st = str(np.float(D_p[i, 5])) + ',' + str(np.float(D_p[i, 6])) + ',' + str(np.float(D_p[i, 7]))
                angp = np.arctan2(D_p[i, 1], D_p[i, 0]) * 180 / np.pi
                a.annotate(st, (xw, yw), rotation=angp, ha='center', va='center')

    a.figure.canvas.draw()


def undo_plot_dir_planes():
    global D_p
    D_p = D_p[:-1, :]
    plot_dir_planes()


def draw_planes_dir():
    global gclick, minx, miny, maxx, maxy, image_diff, D_p

    t = np.float(ui_draw.thickness_entry.text())
    s_a, s_b, s_z = tilt_axes()
    t_ang = np.float(ui.image_angle_entry.text())
    tilt_x = -s_b * np.float(ui.tilt_x_entry.text())
    tilt_y = -s_a * np.float(ui.tilt_y_entry.text())
    tilt_z = -s_z * np.float(ui.tilt_z_entry.text())

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
    x = gclick[-1, 0]
    y = gclick[-1, 1]
    a = figure.add_subplot(111)
    minx, maxx = a.get_xlim()
    miny, maxy = a.get_ylim()

    if ui_draw.surf_checkBox.isChecked():
        s = ui_draw.surface_entry.text().split(",")
        s = np.array([np.float(s[0]), np.float(s[1]), np.float(s[2])])
        s = np.dot(Dstar, s)
        s = np.dot(rotation(phi1, phi, phi2), s)
        s = s / np.linalg.norm(s)
    else:
        s = np.array([0, 0, 1])

    s = np.dot(Rot(tilt_y, 0, 1, 0), np.dot(Rot(tilt_x, 1, 0, 0), np.dot(Rot(tilt_z, 0, 0, 1), np.dot(Rot(t_ang, 0, 0, 1), s))))

    if ui_draw.measure_checkBox.isChecked():
        x0 = gclick[-2, 0]
        x1 = gclick[-1, 0]
        y0 = gclick[-2, 1]
        y1 = gclick[-1, 1]
        vn = np.array([-y1 + y0, -x1 + x0, 0])
        leng = np.linalg.norm(vn)
        vn = vn / leng
        v_plan = np.dot(Rot(-tilt_z, 0, 0, 1), np.dot(Rot(-tilt_x, 1, 0, 0), np.dot(Rot(-tilt_y, 0, 1, 0), np.dot(Rot(t_ang, 0, 0, 1), vn))))
        v_plan = np.dot(np.linalg.inv(rotation(phi1, phi, phi2)), v_plan)
        nd = np.dot(Dstar, nd)
        intersec = np.cross(nd, v_plan)
        intersec_d = np.dot(np.linalg.inv(D), intersec)
        if ui_draw.hexa_Button.isChecked():
            intersec_d = np.array([(2 * intersec_d[0] - intersec_d[1]) / 3, (2 * intersec_d[1] - intersec_d[0]) / 3, intersec_d[2]])

        dire = intersec
        dire = np.dot(rotation(phi1, phi, phi2), dire)
        dire = dire / np.linalg.norm(dire)
        dire = np.dot(Rot(tilt_y, 0, 1, 0), np.dot(Rot(tilt_x, 1, 0, 0), np.dot(Rot(tilt_z, 0, 0, 1), np.dot(Rot(t_ang, 0, 0, 1), dire))))
        b = np.array([0, 0, 1])
        d_proj = (dire - np.dot(dire, b) * b) * t / np.dot(dire, s)
        dr = np.abs(leng * mag_conv * t / np.dot(dire, s) / np.linalg.norm(d_proj))

        te = 'Distance measured in the plane: ' + str(np.around(dr, decimals=2)) + '\n' + 'Direction in the plane: ' + str(np.around(100 * intersec_d[0], decimals=3)) + ',' + str(np.around(100 * intersec_d[1], decimals=3)) + ',' + str(np.around(100 * intersec_d[2], decimals=3)) + '\n' + 'Max proj length: ' + str(np.around(np.linalg.norm(d_proj), decimals=2)) + '\n' + 'Measured proj length: ' + str(np.around(leng * mag_conv, decimals=2))
        ui_draw.measure_label.setText(te)
        if leng * mag_conv > np.linalg.norm(d_proj):
            ui_draw.measure_label.setText(te + '\n' + str("Measure exceeds sample thickness"))

    else:
        ui_draw.measure_label.clear()
        sens = 1
        if ui_draw.inv_checkBox.isChecked():
            sens = -1

        if ui_draw.dir_checkBox.isChecked():
            if ui_draw.hexa_Button.isChecked():
                na = 2 * nd[0] + nd[1]
                n2a = 2 * nd[1] + nd[0]
                nd[0] = na
                nd[1] = n2a

            dire = np.dot(D, nd)
            dire = np.dot(rotation(phi1, phi, phi2), dire)
            dire = dire / np.linalg.norm(dire)
            dire = np.dot(Rot(tilt_y, 0, 1, 0), np.dot(Rot(tilt_x, 1, 0, 0), np.dot(Rot(tilt_z, 0, 0, 1), np.dot(Rot(t_ang, 0, 0, 1), dire))))
            b = np.array([0, 0, 1])
            d_proj = sens * (dire - np.dot(dire, b) * b) * t / np.dot(dire, s) / mag_conv
            D_p = np.vstack((D_p, np.array([d_proj[0], d_proj[1], x, y, nd[0], nd[1], nd[2], 0, 1])))

        else:
            plan = np.dot(Dstar, nd)
            plan = np.dot(rotation(phi1, phi, phi2), plan)
            plan = plan / np.linalg.norm(plan)
            plan = np.dot(Rot(tilt_y, 0, 1, 0), np.dot(Rot(tilt_x, 1, 0, 0), np.dot(Rot(tilt_z, 0, 0, 1), np.dot(Rot(t_ang, 0, 0, 1), plan))))

            T = sens * np.cross(plan, s)
            T = T / np.linalg.norm(T)
            w = plan[2] * t / np.sqrt((1 - np.dot(plan, s)**2) * (1 - T[2]**2)) / mag_conv
            D_p = np.vstack((D_p, np.array([T[0], T[1], w, x, y, nd[0], nd[1], nd[2], 0])))

            if ui_draw.proj_image_checkBox.isChecked():

                a_r = np.arctan2(T[0], T[1])
                a_g = np.abs(np.sqrt(1 - T[2]**2) / plan[2])
                ui_draw.measure_label.setText('Rotation, Stretch' + '\n' + str(np.around(a_r * 180 / np.pi, decimals=3)) + ',' + str(np.around(a_g, decimals=3)))
                a_stretched = figure_stretched.add_subplot(111)
                a_stretched.figure.clear()
                a_stretched = figure_stretched.add_subplot(111)
                img = Image.open(str(image_diff[0]))
                im_rotate = img.rotate(a_r * 180 / np.pi, expand=1)
                stretched_size = (int(im_rotate.size[0] * a_g), im_rotate.size[1])
                im_stretched = im_rotate.resize(stretched_size)
                a_stretched.imshow(im_stretched, origin='upper')
                if ui_draw.scale_checkBox.isChecked():
                    scale = np.float(ui_draw.scale_entry.text())
                    a_stretched.plot([stretched_size[0] / 20, stretched_size[0] / 20 + scale / mag_conv], [stretched_size[1] / 20, stretched_size[1] / 20], 'w-', linewidth=1)
                    a_stretched.annotate(str(scale) + 'nm', (stretched_size[0] / 20 + scale / mag_conv + 5, stretched_size[1] / 20), color="white", backgroundcolor="black")

                a_stretched.axis('off')
                a_stretched.set_facecolor('m')
                a_stretched.figure.canvas.draw()
                Stretched.show()
        plot_dir_planes()


#########################
#
# Open image
#
############################


def open_image():
    global width, height, image_diff, s, gclick, minx, maxx, miny, maxy, press, move, D_p
    press = False
    move = False
    a = figure.add_subplot(111)
    a.figure.clear()
    a = figure.add_subplot(111)
    image_diff = QtWidgets.QFileDialog.getOpenFileName(Interface, "Open image file", "", "*.png *.jpg *.bmp *.tiff *.tif *.jpeg")
    img = Image.open(str(image_diff[0]))
    img = np.array(img)
    a.imshow(img, origin='upper')
    figure.suptitle(str(image_diff[0]))
    height, width = img.shape[0], img.shape[1]
    a.axis([0, width, height, 0])
    a.axis('off')
    a.figure.canvas.draw()
    s = 1
    gclick = np.zeros((1, 2))
    D_p = np.zeros(9)
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


def import_data():
    ui.conditions_Listbox.clear()
    ui.euler_Listbox.clear()
    data_file = QtWidgets.QFileDialog.getOpenFileName(Interface, "Open a data file", "", "*.txt")
    data = open(data_file[0], 'r')
    x0 = []

    for line in data:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        x0.append(list(map(str, line.split())))

    data.close()
    if len(x0[0][0].split(',')) == 3:
        ui.euler_entry.setText(str(x0[0][0]))
        x0 = x0[1:]
    for item in x0:
        ui.conditions_Listbox.addItem(item[0])


def export_data():
    s = [str(x.text()) for x in ui.conditions_Listbox.selectedItems()]
    res = [str(ui.euler_Listbox.item(i).text()) for i in range(ui.euler_Listbox.count())]
    name = QtWidgets.QFileDialog.getSaveFileName(Interface, 'Save File')
    fout = open(name[0], 'w')
    fout.write('# Interface data file \n')
    fout.write('# Euler angles \n')
    fout.write(str(ui.euler_entry.text()) + '\n')
    if ui.direction_button.isChecked():
        d = 'direction'
    else:
        d = 'plane'
    fout.write('# Data for a  ' + str(d) + '\n')
    for item in s:
        fout.write("%s\n" % item)

    fout.write('\n')
    fout.write('# Results \n')
    for item in res:
        fout.write("# %s\n" % item)

    fout.close()


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


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

    app = QtWidgets.QApplication(sys.argv)
    sys.excepthook = except_hook
    QtWidgets.qApp.setApplicationName("Interface")
    Interface = QtWidgets.QMainWindow()
    ui = interfaceUI.Ui_Interface()
    ui.setupUi(Interface)
    figure = plt.figure()
    canvas = FigureCanvas(figure)
    ui.mplvl.addWidget(canvas)
    toolbar = NavigationToolbar(canvas, canvas)
    toolbar.setMinimumWidth(601)

    Draw = QtWidgets.QDialog()
    ui_draw = DrawInterfaceUI.Ui_Draw_plane_directions()
    ui_draw.setupUi(Draw)
    ui.actionDirections_planes.triggered.connect(Draw.show)
    ui_draw.buttonBox.rejected.connect(Draw.close)
    ui_draw.buttonBox.accepted.connect(draw_planes_dir)

    ui.actionImport.triggered.connect(import_data)
    ui.actionExport.triggered.connect(export_data)

    Stretched = QtWidgets.QDialog()
    ui_stretched = DrawStretchedUI.Ui_Draw_Stretched()
    ui_stretched.setupUi(Stretched)
    figure_stretched = plt.figure()
    figure_stretched.patch.set_facecolor('black')
    canvas_stretched = FigureCanvas(figure_stretched)
    ui_stretched.mplvl.addWidget(canvas_stretched)
    toolbar_stretched = NavigationToolbar(canvas_stretched, canvas_stretched)
    toolbar_stretched.setMinimumWidth(101)
    toolbar_stretched.setStyleSheet("background-color:White;")

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
    D_p = np.zeros(9)

    file_struct = open(os.path.join(os.path.dirname(__file__), 'structure.txt'), "r")

    x0 = []

    for line in file_struct:
        x0.append(list(map(str, line.split())))

    i = 0
    file_struct.close()

    for item in x0:
        entry = ui.menuStructure.addAction(item[0])
        entry.triggered.connect(lambda checked, item=item: structure(item))
        i = i + 1

#########################
#
# Get magnification and rotation angle between holder and image from the microscope.txt file
#
############################

    f_micro = open(os.path.join(os.path.dirname(__file__), 'microscope.txt'), "r")

    x_micro = []

    for line in f_micro:
        x_micro.append(list(map(str, line.split())))

    x_micro = np.array(x_micro)
    x_micro_unique, ind_x = np.unique(x_micro[:, 0], return_index=True)

    for i in ind_x[::-1]:
        ui.micro_box.addItem(x_micro[i][0])

    f_micro.close()
# Ctrl+w shortcut to remove clicked pole

    shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+w"), Interface)
    shortcut.activated.connect(reset_last_point)

    ui.actionSave_figure.triggered.connect(open_image)

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
    shortcut_plot = QtWidgets .QShortcut(QtGui.QKeySequence("Ctrl+w"), Draw)
    shortcut_plot.activated.connect(undo_plot_dir_planes)

    Interface.show()
    sys.exit(app.exec_())
