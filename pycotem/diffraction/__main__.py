from __future__ import division
import numpy as np
from PyQt4 import QtGui, QtCore
import sys
import os
from PIL import Image
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib import pyplot as plt
import diffractionUI

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
    distance()
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
    ui.ListBox_d_2.clear()
    ui.ListBox_theo.clear()
    return s, gclick


######################
#
# reset view
#
#########################

def reset():
    global width, height, gclick, s, image_diff

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
    ui.ListBox_d_2.clear()
    ui.ListBox_theo.clear()
    s = 1


#########################
#
# Get experimental interplanar distance
#
############################


def distance():
    global gclick, d, counter, x_calib
    for i in range(1, gclick.shape[0]):
        x = gclick[1, 0]
        y = gclick[1, 1]
        x1 = gclick[i, 0]
        y1 = gclick[i, 1]
        n = np.int(ui.n_entry.text())
        d = np.sqrt((x1 - x)**2 + (y1 - y)**2) / n

        i = 0
        if d == 0:
            bet = 0
        else:
            bet = 180 - np.arccos((y1 - y) / (n * d)) * 180 / np.pi
        d = eval(x_calib[ui.Calib_box.currentIndex()][4]) / d

    if np.isinf(d) == 0:
        ui.ListBox_d_2.addItem(str(np.around(d, decimals=2)) + ',' + str(np.around(bet, decimals=2)))
    return d


#########################
#
# Get theoretical distance
#
############################

def distance_theo():
    global d, G, Dstar

    abc = ui.abc_entry.text().split(",")
    a = np.float(abc[0])
    b = np.float(abc[1])
    c = np.float(abc[2])
    alphabetagamma = ui.alphabetagamma_entry.text().split(",")
    alpha = np.float(alphabetagamma[0])
    beta = np.float(alphabetagamma[1])
    gamma = np.float(alphabetagamma[2])
    e = np.int(ui.indice_entry.text())
    alpha = alpha * np.pi / 180
    beta = beta * np.pi / 180
    gamma = gamma * np.pi / 180
    Dist = np.zeros(((2 * e + 1)**3 - 1, 5))
    V = a * b * c * np.sqrt(1 - (np.cos(alpha)**2) - (np.cos(beta))**2 - (np.cos(gamma))**2 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma))
    D = np.array([[a, b * np.cos(gamma), c * np.cos(beta)], [0, b * np.sin(gamma), c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)], [0, 0, V / (a * b * np.sin(gamma))]])
    G = np.array([[a**2, a * b * np.cos(gamma), a * c * np.cos(beta)], [a * b * np.cos(gamma), b**2, b * c * np.cos(alpha)], [a * c * np.cos(beta), b * c * np.cos(alpha), c**2]])
    Dstar = np.transpose(np.linalg.inv(D))
    ui.ListBox_theo.clear()
    w = 0
    for i in range(-e, e + 1):
        for j in range(-e, e + 1):
            for k in range(-e, e + 1):
                if (i, j, k) != (0, 0, 0):
                    di = 1 / (np.sqrt(np.dot(np.array([i, j, k]), np.dot(np.linalg.inv(G), np.array([i, j, k])))))
                    if di < (d + 0.1) and di > (d - 0.1):
                        I = extinction(ui.SpaceGroup_box.currentText(), i, j, k)
                        Dist[w, :] = np.array([np.around(di, decimals=3), i, j, k, I])
                        w = w + 1

    for k in range(0, w):
        ui.ListBox_theo.addItem(str(Dist[k, 0]) + ',' + str(int(Dist[k, 1])) + ',' + str(int(Dist[k, 2])) + ',' + str(int(Dist[k, 3])) + ',' + str(Dist[k, 4]))
    return

########################
#
# Add/Remove spot
#
#########################


def add_spot():
    s1 = ui.ListBox_theo.currentItem().text().split(',')
    s2 = ui.ListBox_d_2.currentItem().text().split(',')
    s3 = ui.tilt_a_entry.text()
    s4 = ui.tilt_b_entry.text()
    s5 = ui.tilt_z_entry.text()

    s = s3 + ',' + s4 + ',' + s5 + ',' + s2[1] + ',' + s1[1] + ',' + s1[2] + ',' + s1[3]
    ui.diff_spot_Listbox.addItem(s)


def remove_spot():
    ui.diff_spot_Listbox.takeItem(ui.diff_spot_Listbox.currentRow())

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


def dhkl(g1, g2, g3):
    global G
    return 1 / (np.sqrt(np.dot(np.array([g1, g2, g3]), np.dot(np.linalg.inv(G), np.array([g1, g2, g3])))))


#
# Get all equivalent diffraction +/-g
# from permutation
#

def perm_g(pole1, pole2, pole3):

    v = dhkl(pole1, pole2, pole3)

    g = np.array([[pole1, pole2, pole3], [pole1, pole2, -pole3], [pole1, -pole2, pole3], [-pole1, pole2, pole3], [pole2, pole1, pole3], [pole2, pole1, -pole3], [pole2, -pole1, pole3], [-pole2, pole1, pole3], [pole2, pole3, pole1], [pole2, pole3, -pole1], [pole2, -pole3, pole1], [-pole2, pole3, pole1], [pole1, pole3, pole2], [pole1, pole3, -pole2], [pole1, -pole3, pole2], [-pole1, pole3, pole2], [pole3, pole1, pole2], [pole3, pole1, -pole2], [pole3, -pole1, pole2], [-pole3, pole1, pole2], [pole3, pole2, pole1], [pole3, pole2, -pole1], [pole3, -pole2, pole1]])

    o = []
    for i in range(0, g.shape[0]):
        if np.abs(dhkl(g[i, 0], g[i, 1], g[i, 2]) - v) < 1e-9:
            o = np.append(o, i)
    g = g[np.array(o, dtype=int), :]
    g = np.vstack((g, -g))
    g = unique_rows(g)
    return g

#
# Determine x,y,z axis in crystal coordinates for all the possible sets
# of diffraction vectors. Retrieve the Euler angles from x,y,z. Define the
# index=(x.y)^z as the measure of the orthogonality of the three vectors (should
# be very close to 1). Incorrect index can be the consequence of a choice of coplanar vectors.
#


def euler_determine_3(g_perm, g_sample):
    R = []
    for i in range(0, g_perm[0].shape[0]):
        for j in range(0, g_perm[1].shape[0]):
            for k in range(0, g_perm[2].shape[0]):
                g0 = g_perm[0][i]
                g1 = g_perm[1][j]
                g2 = g_perm[2][k]
                Gs = np.array([g0, g1, g2])
                Gs = np.dot(Dstar, Gs.T)
                Gs = (Gs / np.linalg.norm(Gs.T, axis=1)).T
                x = np.linalg.lstsq(Gs, g_sample[:, 0])[0]
                x = x / np.linalg.norm(x)
                y = np.linalg.lstsq(Gs, g_sample[:, 1])[0]
                y = y / np.linalg.norm(y)
                z, res = np.linalg.lstsq(Gs, g_sample[:, 2])[0:2]
                z = z / np.linalg.norm(z)
                phi_1 = np.arctan2(x[2], -y[2]) * 180 / np.pi
                phi = np.arccos(z[2]) * 180 / np.pi
                phi_2 = np.arctan2(z[0], z[1]) * 180 / np.pi
                t = 0
                for r in range(0, 3):
                    t = t + np.abs(np.arccos(np.dot(np.dot(rotation(phi_1, phi, phi_2), Gs[r, :]), g_sample[r, :])) * 180 / np.pi)
                t = t / r
                R.append([np.dot(np.cross(x, y), z), 0, phi_1, phi, phi_2, t])

    R = np.array(R)
    R = R[np.argmin(R[:, 5]), :]
    return R


def euler_determine_4(g_perm, g_sample):
    R = []
    N = []
    for i in range(0, g_perm[0].shape[0]):
        for j in range(0, g_perm[1].shape[0]):
            for k in range(0, g_perm[2].shape[0]):
                g0 = g_perm[0][i]
                g1 = g_perm[1][j]
                g2 = g_perm[2][k]
                Gs = np.array([g0, g1, g2])
                Gs = np.dot(Dstar, Gs.T)
                Gs = (Gs / np.linalg.norm(Gs.T, axis=1)).T
                x = np.linalg.lstsq(Gs, g_sample[0:3, 0])[0]
                x = x / np.linalg.norm(x)
                y = np.linalg.lstsq(Gs, g_sample[0:3, 1])[0]
                y = y / np.linalg.norm(y)
                z = np.linalg.lstsq(Gs, g_sample[0:3, 2])[0]
                z = z / np.linalg.norm(z)
                c = np.dot(np.cross(x, y), z)
                N.append([c, g0[0], g0[1], g0[2], g1[0], g1[1], g1[2], g2[0], g2[1], g2[2]])

    N = np.array(N)
    N = N[np.argmax(N[:, 0]), 1:]
    g0 = N[0:3]
    g1 = N[3:6]
    g2 = N[6:9]

    for l in range(0, g_perm[3].shape[0]):
        g3 = g_perm[3][l]
        Gs = np.array([g0, g1, g2, g3], dtype=float)
        Gs = np.dot(Dstar, Gs.T)
        Gs = (Gs / np.linalg.norm(Gs.T, axis=1)).T
        x = np.linalg.lstsq(Gs, g_sample[:, 0])[0]
        x = x / np.linalg.norm(x)
        y = np.linalg.lstsq(Gs, g_sample[:, 1])[0]
        y = y / np.linalg.norm(y)
        z, res = np.linalg.lstsq(Gs, g_sample[:, 2])[0:2]
        z = z / np.linalg.norm(z)
        phi_1 = np.arctan2(x[2], -y[2]) * 180 / np.pi
        phi = np.arccos(z[2]) * 180 / np.pi
        phi_2 = np.arctan2(z[0], z[1]) * 180 / np.pi
        t = 0
        for r in range(0, 4):
            t = t + np.abs(np.arccos(np.dot(np.dot(rotation(phi_1, phi, phi_2), Gs[r, :]), g_sample[r, :])) * 180 / np.pi)
        t = t / r

        if res:
            R.append([np.dot(np.cross(x, y), z), res[0], phi_1, phi, phi_2, t])

    R = np.array(R)
    R = R[np.argmin(R[:, 5]), :]
    return R


def euler_determine_5(g_perm, g_sample):
    R = []
    N = []
    L = []
    for i in range(0, g_perm[0].shape[0]):
        for j in range(0, g_perm[1].shape[0]):
            for k in range(0, g_perm[2].shape[0]):
                g0 = g_perm[0][i]
                g1 = g_perm[1][j]
                g2 = g_perm[2][k]
                Gs = np.array([g0, g1, g2])
                Gs = np.dot(Dstar, Gs.T)
                Gs = (Gs / np.linalg.norm(Gs.T, axis=1)).T
                x = np.linalg.lstsq(Gs, g_sample[0:3, 0])[0]
                x = x / np.linalg.norm(x)
                y = np.linalg.lstsq(Gs, g_sample[0:3, 1])[0]
                y = y / np.linalg.norm(y)
                z = np.linalg.lstsq(Gs, g_sample[0:3, 2])[0]
                z = z / np.linalg.norm(z)
                c = np.abs(np.dot(np.cross(x, y), z))
                N.append([c, g0[0], g0[1], g0[2], g1[0], g1[1], g1[2], g2[0], g2[1], g2[2]])

    N = np.array(N)
    N = N[np.argmax(N[:, 0]), 1:]

    g0 = N[0:3]
    g1 = N[3:6]
    g2 = N[6:9]

    for l in range(0, g_perm[3].shape[0]):
        g3 = g_perm[3][l]
        Gs = np.array([g0, g1, g2, g3], dtype=float)
        Gs = np.dot(Dstar, Gs.T)
        Gs = (Gs / np.linalg.norm(Gs.T, axis=1)).T
        z, res = np.linalg.lstsq(Gs, g_sample[0:4, 2])[0:2]
        L.append([res, g0[0], g0[1], g0[2], g1[0], g1[1], g1[2], g2[0], g2[1], g2[2], g3[0], g3[1], g3[2]])

    L = np.array(L)
    L = L[np.argmin(L[:, 0]), 1:]
    g0 = L[0:3]
    g1 = L[3:6]
    g2 = L[6:9]
    g3 = L[9:12]

    for n in range(0, g_perm[4].shape[0]):
        g4 = g_perm[4][n]
        Gs = np.array([g0, g1, g2, g3, g4], dtype=float)
        Gs = np.dot(Dstar, Gs.T)
        Gs = (Gs / np.linalg.norm(Gs.T, axis=1)).T
        x = np.linalg.lstsq(Gs, g_sample[:, 0])[0]
        x = x / np.linalg.norm(x)
        y = np.linalg.lstsq(Gs, g_sample[:, 1])[0]
        y = y / np.linalg.norm(y)
        z, res = np.linalg.lstsq(Gs, g_sample[:, 2])[0:2]
        z = z / np.linalg.norm(z)
        phi_1 = np.arctan2(x[2], -y[2]) * 180 / np.pi
        phi = np.arccos(z[2]) * 180 / np.pi
        phi_2 = np.arctan2(z[0], z[1]) * 180 / np.pi
        t = 0
        for r in range(0, 5):
            t = t + np.abs(np.arccos(np.dot(np.dot(rotation(phi_1, phi, phi_2), Gs[r, :]), g_sample[r, :])) * 180 / np.pi)
        t = t / r
        if res:
            R.append([np.dot(np.cross(x, y), z), res[0], phi_1, phi, phi_2, t])
    R = np.array(R)
    R = R[np.argmin(R[:, 5]), :]
    return R

########################################
#
# Determine orientation from the selected spots.
#
#########################################


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


def get_orientation():
    global G, Dstar
    s_a, s_b, s_z = tilt_axes()
    ui.euler_listbox.clear()
    s = [str(x.text()) for x in ui.diff_spot_Listbox.selectedItems()]
    tilt_a = []
    tilt_b = []
    tilt_z = []
    inclination = []
    g_hkl = []

    for i in range(0, len(s)):
        l = map(float, s[i].split(','))
        tilt_a.append(l[0])
        tilt_b.append(l[1])
        tilt_z.append(l[2])
        inclination.append(l[3])
        g_hkl.append(l[4:7])
    inclination = np.array(inclination)
    tilt_a = np.array(tilt_a)
    tilt_b = np.array(tilt_b)
    tilt_z = np.array(tilt_z)
    g_hkl = np.array(g_hkl)
    t_ang = np.float(ui.tilt_axis_angle_entry.text())
    g_perm = []
    for i in range(0, np.shape(g_hkl)[0]):
        g_perm.append(perm_g(g_hkl[i, 0], g_hkl[i, 1], g_hkl[i, 2]))

    g_sample = np.zeros((np.shape(tilt_a)[0], 3))
    for i in range(0, np.shape(tilt_a)[0]):
        R = np.dot(Rot(s_z * tilt_z[i], 0, 0, 1), np.dot(Rot(s_b * tilt_b[i], 1, 0, 0), Rot(s_a * tilt_a[i], 0, 1, 0)))
        ny = (-inclination[i]) * np.pi / 180
        t = np.array([-np.sin(ny), np.cos(ny), 0])
        g_sample[i, :] = np.dot(R, np.dot(Rot(t_ang, 0, 0, 1), t))

    if g_hkl.shape[0] == 3:
        R = euler_determine_3(g_perm, g_sample)

    elif g_hkl.shape[0] == 4:
        R = euler_determine_4(g_perm, g_sample)

    elif g_hkl.shape[0] == 5:
        R = euler_determine_5(g_perm, g_sample)

    else:
        ui.euler_listbox.addItem("Number of spots should be between 3 and 5")
    np.set_printoptions(suppress=True)
    if R.shape[0] > 0:
        ui.euler_listbox.addItem('Phi1,Phi,Phi2')
        ui.euler_listbox.addItem(str(np.around(R[2], decimals=3)) + ',' + str(np.around(R[3], decimals=3)) + ',' + str(np.around(R[4], decimals=3)))
        ui.euler_listbox.addItem('Mean angular deviation, Orthogonality, Residual')
        ui.euler_listbox.addItem(str(np.around(R[5], decimals=3)) + ',' + str(np.around(R[0], decimals=5)) + ',' + str(np.around(R[1], decimals=6)))


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
    image_diff = QtGui.QFileDialog.getOpenFileName(Index, "Open image file", "", "*.png *.jpg *.bmp *.tiff *.tif *.jpeg")
    img = Image.open(str(image_diff))
    img = np.array(img)
    a.imshow(img, origin='upper')
    figure.suptitle(str(image_diff))
    height, width = img.shape[0], img.shape[1]
    a.axis([0, width, height, 0])
    a.axis('off')
    a.figure.canvas.draw()
    ui.ListBox_d_2.clear()
    s = 1
    gclick = np.zeros((1, 2))
    ui.ListBox_d_2.clear()
    minx = 0
    maxx = width
    miny = height
    maxy = 0
    return s, gclick


#####################################################
#
# Get diffraction spectrum
#
###################################################

def pair(number):
    if number % 2 == 0:
        return 1


def impair(number):
    if number % 2 == 0:
        return 0


class Spect(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Spect, self).__init__(parent)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        self.toolbar = NavigationToolbar(self.canvas, self)

        gridLayout = QtGui.QGridLayout()
        self.lineEdit = QtGui.QLineEdit()
        self.lineEdit.setObjectName(_fromUtf8("lineEdit"))
        gridLayout.addWidget(self.lineEdit, 1, 2, 1, 1)
        self.label = QtGui.QLabel('Max indices')
        self.label.setObjectName(_fromUtf8("Max indices"))
        gridLayout.addWidget(self.label, 1, 0, 1, 2)

        self.buttonBox = QtGui.QDialogButtonBox()
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.Ok)
        gridLayout.addWidget(self.buttonBox, 2, 0, 1, 3)
        gridLayout.addWidget(self.canvas, 0, 0, 1, 3)
        gridLayout.addWidget(self.toolbar, 3, 0, 1, 3)

        self.buttonBox.rejected.connect(self.close)
        self.buttonBox.accepted.connect(self.spectre)

        self.setLayout(gridLayout)

    def spectre(self):
        global x_space
        a2 = self.figure.add_subplot(111)
        a2.clear()
        abc = ui.abc_entry.text().split(",")
        a = np.float(abc[0])
        b = np.float(abc[1])
        c = np.float(abc[2])
        alphabetagamma = ui.alphabetagamma_entry.text().split(",")
        alpha = np.float(alphabetagamma[0])
        beta = np.float(alphabetagamma[1])
        gamma = np.float(alphabetagamma[2])
        e = np.int(self.lineEdit.text())
        alp = alpha * np.pi / 180
        bet = beta * np.pi / 180
        gam = gamma * np.pi / 180
        G = np.array([[a**2, a * b * np.cos(gam), a * c * np.cos(bet)], [a * b * np.cos(gam), b**2, b * c * np.cos(alp)], [a * c * np.cos(bet), b * c * np.cos(alp), c**2]])

        di = np.zeros((1, 4))
        cont = np.zeros((1, 1))
        space_group = ui.SpaceGroup_box.currentText()

        for tt in range(0, len(x_space)):
            if space_group == x_space[tt][0]:
                s = tt
        rr = s + 1

        while len(x_space[rr]) == 4:
            rr = rr + 1

        for h in range(-e, e + 1):
            for k in range(-e, e + 1):
                for l in range(-e, e + 1):
                    if (h, k, l) != (0, 0, 0):
                        gg = 0

                        for c in range(0, cont.shape[0]):
                            if 1 / (np.sqrt(np.dot(np.array([h, k, l]), np.dot(np.linalg.inv(G), np.array([h, k, l]))))) > cont[c] - 0.00000001 and 1 / (np.sqrt(np.dot(np.array([h, k, l]), np.dot(np.linalg.inv(G), np.array([h, k, l]))))) < cont[c] + 0.00000001:
                                gg = 1

                        if gg == 0:
                            di = np.vstack((di, (1 / (np.sqrt(np.dot(np.array([h, k, l]), np.dot(np.linalg.inv(G), np.array([h, k, l]))))), h, k, l)))
                            cont = np.vstack((cont, 1 / (np.sqrt(np.dot(np.array([h, k, l]), np.dot(np.linalg.inv(G), np.array([h, k, l])))))))

        for dc in range(1, di.shape[0]):
            F = 0
            q = 2 * np.pi * 1e-10 / di[dc, 0]
            for ii in range(s + 1, rr):
                f = str(x_space[ii][0])

                for z in range(0, len(x_scatt)):

                    if f == x_scatt[z][0]:
                        f = eval(x_scatt[z][1]) * np.exp(-eval(x_scatt[z][2]) * (q / 4 / np.pi)**2) + eval(x_scatt[z][3]) * np.exp(-eval(x_scatt[z][4]) * (q / 4 / np.pi)**2) + eval(x_scatt[z][5]) * np.exp(-eval(x_scatt[z][6]) * (q / 4 / np.pi)**2) + eval(x_scatt[z][7]) * np.exp(-eval(x_scatt[z][8]) * (q / 4 / np.pi)**2) + eval(x_scatt[z][9])

                F = F + f * np.exp(2j * np.pi * (eval(x_space[ii][1]) * di[dc, 1] + eval(x_space[ii][2]) * di[dc, 2] + eval(x_space[ii][3]) * di[dc, 3]))

            ff = float(np.real(F * np.conj(F)))
            bar_width = 1 / np.shape(di)[0]
            if ff > 0.0000001:
                ann = str(int(di[dc, 1])) + str(int(di[dc, 2])) + str(int(di[dc, 3]))
                a2.text(di[dc, 0], ff, ann, rotation='vertical')
                a2.bar(di[dc, 0], ff, width=bar_width, align='center')

        plt.xlabel('Interplanar distance (nm)')
        plt.ylabel('Intensity (a.u.)')
        self.canvas.draw()

#########################
#
# Compute extinction conditions
#
############################


def extinction(space_group, h, k, l):
    global x_space, G
    F = 0
    s = 0
    q = 2 * np.pi * 1e-10 / (np.sqrt(np.dot(np.array([h, k, l]), np.dot(np.linalg.inv(G), np.array([h, k, l])))))

    for i in range(0, len(x_space)):
        if space_group == x_space[i][0]:
            s = i

    while (s < (len(x_space) - 1) and (len(x_space[s + 1]) == 4)):

        f = str(x_space[s + 1][0])
        for z in range(0, len(x_scatt)):
            if f == x_scatt[z][0]:
                f = eval(x_scatt[z][1]) * np.exp(-eval(x_scatt[z][2]) * (q / 4 / np.pi)**2) + eval(x_scatt[z][3]) * np.exp(-eval(x_scatt[z][4]) * (q / 4 / np.pi)**2) + eval(x_scatt[z][5]) * np.exp(-eval(x_scatt[z][6]) * (q / 4 / np.pi)**2) + eval(x_scatt[z][7]) * np.exp(-eval(x_scatt[z][8]) * (q / 4 / np.pi)**2) + eval(x_scatt[z][9])
        F = F + f * np.exp(2j * np.pi * (eval(x_space[s + 1][1]) * h + eval(x_space[s + 1][2]) * k + eval(x_space[s + 1][3]) * l))
        s = s + 1
    I = np.around(float(np.real(F * np.conj(F))), decimals=2)
    return I


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

#############################################################
#
# Launch
#
# "


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
    Index = QtGui.QMainWindow()
    ui = diffractionUI.Ui_Diffraction()
    ui.setupUi(Index)
    figure = plt.figure()
    canvas = FigureCanvas(figure)
    ui.mplvl.addWidget(canvas)
    toolbar = NavigationToolbar(canvas, canvas)
    toolbar.setMinimumWidth(601)
    # Index.setWindowOpacity(0.8)
######################################################
#
# Import calibrations from txt files: microscope, E(kV),camera lebghth (cm), binning,px/A
#
######################################################

f_calib = open(os.path.join(os.path.dirname(__file__), 'calibrations.txt'), "r")

x_calib = []

for line in f_calib:
    x_calib.append(map(str, line.split()))

f_calib.close()
counter = len(x_calib)

for i in range(counter):
    ui.Calib_box.addItem(x_calib[i][0] + ' ' + x_calib[i][1] + 'keV ' + x_calib[i][2] + 'cm' + ', Binning:' + x_calib[i][3])

#########################
#
# Get space group data from inpt file space_group.txt
#
############################

f_space = open(os.path.join(os.path.dirname(__file__), 'space_group.txt'), "r")

x_space = []

for line in f_space:
    x_space.append(map(str, line.split()))

list_space_group = []
for i in range(0, len(x_space)):
    if len(x_space[i]) == 1:
        list_space_group.append(x_space[i][0])

f_space.close()


#################################################################################
# import crystal structures from un txt file Name,a,b,c,alpha,beta,gamma,space group
#
################################################################################

def structure(item):
    global x0, e_entry
    ui.abc_entry.setText(str(item[1]) + ',' + str(item[2]) + ',' + str(item[3]))
    ui.alphabetagamma_entry.setText(str(item[4]) + ',' + str(item[5]) + ',' + str(item[6]))
    ii = ui.SpaceGroup_box.findText(item[7])
    ui.SpaceGroup_box.setCurrentIndex(ii)


file_struct = open(os.path.join(os.path.dirname(__file__), 'structure.txt'), "r")

x0 = []

for line in file_struct:
    x0.append(map(str, line.split()))

i = 0
file_struct.close()

for item in x0:
    entry = ui.menuStructure.addAction(item[0])
    Index.connect(entry, QtCore.SIGNAL('triggered()'), lambda item=item: structure(item))
    ui.SpaceGroup_box.addItem(x0[i][7])
    i = i + 1

f_scatt = open(os.path.join(os.path.dirname(__file__), 'scattering.txt'), "r")

x_scatt = []

for line in f_scatt:
    x_scatt.append(map(str, line.split()))


f_scatt.close()


Index.connect(ui.actionSave_figure, QtCore.SIGNAL('triggered()'), open_image)
#figure.canvas.mpl_connect('button_press_event', click)

figure.canvas.mpl_connect('button_press_event', onpress)
figure.canvas.mpl_connect('button_release_event', onrelease)
figure.canvas.mpl_connect('motion_notify_event', onmove)
press = False
move = False

ui.Button_reset.clicked.connect(reset_points)
ui.Button_reset_all.clicked.connect(reset)
ui.distance_button.clicked.connect(distance_theo)
dialSpect = Spect()
dialSpect.setWindowTitle("Spectrum")
Index.connect(ui.actionCalculate_spectrum, QtCore.SIGNAL('triggered()'), dialSpect.exec_)
ui.add_spot_button.clicked.connect(add_spot)
ui.remove_spot_button.clicked.connect(remove_spot)
ui.orientation_button.clicked.connect(get_orientation)

ui.diff_spot_Listbox.setSelectionMode(QtGui.QListWidget.ExtendedSelection)
ui.n_entry.setText('1')
ui.indice_entry.setText('5')
ui.tilt_axis_angle_entry.setText('0')
ui.tilt_a_entry.setText('0')
ui.tilt_b_entry.setText('0')
ui.tilt_z_entry.setText('0')

s = 1
gclick = np.zeros((1, 2))
Index.show()
app.exec_()
