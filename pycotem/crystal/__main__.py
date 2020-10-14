from __future__ import division
import numpy as np
from PyQt5 import QtWidgets
import sys
import os
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib import pyplot as plt
from . import crystalUI


def Rot(th, a, b, c):

    aa = a / np.linalg.norm([a, b, c])
    bb = b / np.linalg.norm([a, b, c])
    cc = c / np.linalg.norm([a, b, c])
    c1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    c2 = np.array([[aa**2, aa * bb, aa * cc], [bb * aa, bb**2, bb * cc], [cc * aa,
                                                                          cc * bb, cc**2]], float)
    c3 = np.array([[0, -cc, bb], [cc, 0, -aa], [-bb, aa, 0]], float)
    R = np.cos(th) * c1 + (1 - np.cos(th)) * c2 + np.sin(th) * c3

    return R


def coord_ortho(P, plan):
    global Dstar, varname
    ref = np.array([0, 0, 1])
    planN = np.dot(Dstar, plan)
    the = np.arccos(np.dot(planN, ref) / (np.linalg.norm(ref) * np.linalg.norm(planN)))
    if plan[0] == 0 and plan[1] == 0:
        axe = ref
    else:
        axe = np.cross(planN, ref)

    M = np.dot(Rot(the, axe[0], axe[1], axe[2]), P)
    return M


def unique(a):
    a = np.sort(a)
    b = np.diff(a)
    b = np.r_[1, b]
    return a[b != 0]


def calcul():
    global vec, varname, atom0, Dstar, taille, zoom, EL, Dz, dsc_cond

    if varname != 0:
        f_space = open(varname[0], "r")
    else:
        varname = getFileName()
        f_space = open(varname, "r")

    crist = []

    for line in f_space:
        crist.append(list(map(str, line.split())))

    f_space.close()
    vec = []
    atom0 = []
    for i in range(0, np.size(crist)):
        if np.size(crist[i]) == 3:
            vec.append(crist[i])
        else:
            atom0.append(crist[i])

    vec = np.array(vec, float)
    E = np.array([0, 0, 0, 0])
    EL = np.array([0, 0, 0, 0, 0])
    H = np.array([[0]])
    Dz = np.array([0, 0, 0])

    if ui.dsc_box.isChecked() is False:
        atom0 = np.array(atom0, float)
        maxi = np.int(atom0[np.shape(atom0)[0] - 1, 0])
        for h in range(1, maxi + 1):
            Ca = calcul_atom(atom0[atom0[:, 0] == h])

            E = np.vstack((E, Ca[0]))
            Dz = np.vstack((Dz, Ca[1]))
            H = np.vstack((H, h * np.ones((np.shape(Ca[0])[0], 1))))

    if ui.dsc_box.isChecked():
        atom0 = np.array([1, 0, 0, 0])
        h = 1
        Ca = calcul_atom(atom0)
        E = np.vstack((E, Ca[0]))
        Dz = np.vstack((Dz, Ca[1]))
        H = np.vstack((H, h * np.ones((np.shape(Ca[0])[0], 1))))

    EL = np.append(E, H, axis=1)
    EL = np.delete(EL, (0), axis=0)
    Dz = np.delete(Dz, (0), axis=0)


def trace():
    global vec, varname, atom0, Dstar, taille, zoom, EL, Dz

    fi = figure.add_subplot(111)
    fi.figure.clear()
    fi = figure.add_subplot(111)
    sim = int(ui.markers_entry.text())
    if ui.dsc_box.isChecked() is False:
        if ui.square_box.isChecked() is False:
            fi.scatter(EL[:, 0], EL[:, 1], s=sim, c=EL[:, 3], marker='o')
        else:
            fi.scatter(EL[:, 0], EL[:, 1], s=sim, c=EL[:, 3], marker='s')

        if ui.atoms_box.isChecked():
            for k in range(0, np.shape(EL)[0]):
                fi.annotate(str(int(EL[k, 4])), (EL[k, 0], EL[k, 1]))

        if ui.labels_box.isChecked():
            for q in range(0, np.shape(EL)[0]):
                at = Dz[q, :]
                at = np.dot(at, Dstar)

                vector = str(np.around(at[0], decimals=3)) + ',' + str(np.around(at[1], decimals=3)) + ',' + str(np.around(at[2], decimals=3))
                fi.annotate(vector, (EL[q, 0], EL[q, 1]))

    if ui.dsc_box.isChecked():
        theta = float(ui.angle_entry.text())
        theta = theta * np.pi / 180
        ELr = np.dot(EL[:, 0:3], Rot(theta, 0, 0, 1))
        M = unique(EL[:, 3])

        Ma0 = []
        Ma1 = []
        Ma2 = []
        Ma = []
        m = ('o', 's', '^', '*', 'h')
        abc = ui.abc_entry.text().split(",")
        a = np.float(abc[0])
        b = np.float(abc[1])
        c = np.float(abc[2])
        ee = np.float(ui.precision_entry.text())
        if ui.coincidence_box.isChecked():
            coi = np.array([0])
            ep = 0.01 * ee * np.max([a, b, c])
            for i in range(0, np.shape(EL[:, 3])[0]):
                for j in range(0, np.shape(ELr[:, 0])[0]):
                    if np.linalg.norm(EL[i, 0:3] - ELr[j, :]) < ep:
                        coi = np.hstack((coi, i))
            coi = np.unique(coi)
        for t in range(0, np.shape(M)[0]):
            for i in range(0, np.shape(EL[:, 3])[0]):
                if EL[i, 3] == M[t]:

                    if t > np.size(m) - 1:
                        Ma0.append('o')
                        Ma1.append('o')

                    else:
                        Ma0.append(m[t])
                        Ma1.append(m[t])
                    if ui.layers_box.isChecked():
                        Ma2.append(str(t))
        else:
            Ma.append('D')

        for y in range(0, np.shape(EL[:, 3])[0]):
            fi.scatter(ELr[y, 0], ELr[y, 1], s=sim, marker=Ma0[y], color='white', edgecolor='black')
            fi.scatter(EL[y, 0], EL[y, 1], s=sim, marker=Ma1[y], color='black', edgecolor='black')
            if ui.layers_box.isChecked():
                fi.text(EL[y, 0], EL[y, 1], Ma2[y])
                fi.text(ELr[y, 0], ELr[y, 1], Ma2[y])
        if ui.coincidence_box.isChecked():
            for z in coi:
                fi.scatter(EL[z, 0], EL[z, 1], s=sim * 1.5, marker=Ma0[z], color='blue', edgecolor='black')

        if ui.labels_box.isChecked():
            for q in range(0, np.shape(EL)[0]):
                at = Dz[q, :]
                at = np.dot(at, Dstar)

                vector = str(np.around(at[0], decimals=1)) + ',' + str(np.around(at[1], decimals=1)) + ',' + str(np.around(at[2], decimals=1))
                fi.annotate(vector, (EL[q, 0], EL[q, 1]))
    fi.axis('off')
    fi.axis('equal')
    fi.figure.canvas.draw()


def rep():
    global varname, vec, E, C, Dz, atom0
    fi = figure.add_subplot(111)
    fi.figure.clear()
    fi = figure.add_subplot(111, projection='3d')
    sim = int(ui.markers_entry.text())
    if varname != 0:
        f_space = open(varname, "r")
    else:
        varname = getFileName()
        f_space = open(varname, "r")

    crist = []

    for line in f_space:
        crist.append(list(map(str, line.split())))

    f_space.close()
    vec = []
    atom0 = []
    for i in range(0, np.size(crist)):
        if np.size(crist[i]) == 3:
            vec.append(crist[i])
        else:
            atom0.append(crist[i])

    vec = np.array(vec, float)
    atom0 = np.array(atom0, float)

    maxi = np.int(atom0[np.shape(atom0)[0] - 1, 0])

    for h in range(1, maxi + 1):

        E = calcul_rep(atom0[atom0[:, 0] == h])

        fi.scatter(E[:, 0], E[:, 1], E[:, 2], s=sim, c=str(h / maxi))

    fi.axis('off')
    fi.figure.canvas.draw()


def calcul_rep(atom):
    global Dstar, varname, C, D0, Dz, planN, plan, vec, c
    abc = ui.abc_entry.text().split(",")
    a = np.float(abc[0])
    b = np.float(abc[1])
    c = np.float(abc[2])
    alphabetagamma = ui.alphabetagamma_entry.text().split(",")
    alpha = np.float(alphabetagamma[0])
    beta = np.float(alphabetagamma[1])
    gamma = np.float(alphabetagamma[2])
    alp = alpha * np.pi / 180
    bet = beta * np.pi / 180
    gam = gamma * np.pi / 180

    V = a * b * c * np.sqrt(1 - (np.cos(alp)**2) - (np.cos(bet))**2 - (np.cos(gam))**2 + 2 * b * c * np.cos(alp) * np.cos(bet) * np.cos(gam))
    D = np.array([[a, b * np.cos(gam), c * np.cos(bet)], [0, b * np.sin(gam), c * (np.cos(alp) - np.cos(bet) * np.cos(gam)) / np.sin(gam)], [0, 0, V / (a * b * np.sin(gam))]])
    Dstar = np.transpose(np.linalg.inv(D))

    n = ui.size_entry.text().split(",")
    na_rep = np.int(n[0])
    nb_rep = np.int(n[1])
    nc_rep = np.int(n[2])

    A = np.zeros((np.shape(atom)[0], np.shape(atom)[1] - 1))
    w = 0

    for v in range(0, np.shape(atom)[0]):

        A[w, :] = np.dot(D, np.array([atom[v, 1], atom[v, 2], atom[v, 3]]))
        w = w + 1

    atom_pos = np.array(A[0, :])
    for f in range(0, np.shape(A)[0]):
        for i in range(-na_rep, na_rep + 1):
            for j in range(-nb_rep, nb_rep + 1):
                for k in range(-nc_rep, nc_rep + 1):

                    atom_pos = np.vstack((atom_pos, A[f, :] + i * a * vec[0, :] + j * b * vec[1, :] + k * c * vec[2, :]))

    return atom_pos


def calcul_atom(atom):
    global Dstar, varname, C, D0, planN, plan, vec, atom_pos

    abc = ui.abc_entry.text().split(",")
    a = np.float(abc[0])
    b = np.float(abc[1])
    c = np.float(abc[2])
    alphabetagamma = ui.alphabetagamma_entry.text().split(",")
    alpha = np.float(alphabetagamma[0])
    beta = np.float(alphabetagamma[1])
    gamma = np.float(alphabetagamma[2])
    alp = alpha * np.pi / 180
    bet = beta * np.pi / 180
    gam = gamma * np.pi / 180

    V = a * b * c * np.sqrt(1 - (np.cos(alp)**2) - (np.cos(bet))**2 - (np.cos(gam))**2 + 2 * b * c * np.cos(alp) * np.cos(bet) * np.cos(gam))
    D = np.array([[a, b * np.cos(gam), c * np.cos(bet)], [0, b * np.sin(gam), c * (np.cos(alp) - np.cos(bet) * np.cos(gam)) / np.sin(gam)], [0, 0, V / (a * b * np.sin(gam))]])
    Dstar = np.transpose(np.linalg.inv(D))

    n = ui.size_entry.text().split(",")
    na = np.int(n[0])
    nb = np.int(n[1])
    nc = np.int(n[2])

    if ui.dsc_box.isChecked() is False:
        A = np.zeros((np.shape(atom)[0], np.shape(atom)[1] - 1))
        w = 0
        for v in range(0, np.shape(atom)[0]):
            A[w, :] = np.dot(D, np.array([atom[v, 1], atom[v, 2], atom[v, 3]]))
            w = w + 1
        atom_pos = np.array(A[0, :])
        for f in range(0, np.shape(A)[0]):
            for i in range(-na, na + 1):
                for j in range(-nb, nb + 1):
                    for k in range(-nc, nc + 1):
                        atom_pos = np.vstack((atom_pos, A[f, :] + i * a * vec[0, :] + j * b * vec[1, :] + k * c * vec[2, :]))

    if ui.dsc_box.isChecked():
        atom_pos = np.array([0, 0, 0])
        for i in range(-na, na + 1):
            for j in range(-nb, nb + 1):
                for k in range(-nc, nc + 1):
                    atom_pos = np.vstack((atom_pos, i * a * vec[0, :] + j * b * vec[1, :] + k * c * vec[2, :]))
    pl = ui.plane_entry.text().split(",")
    h = np.float(pl[0])
    k = np.float(pl[1])
    l = np.float(pl[2])

    plan = np.array([h, k, l])
    planN = np.dot(Dstar, plan)
    Dz = np.array([0, 0, 0])
    D0 = np.array([0])
    C = np.array([0, 0, 0])

    L = np.zeros(np.shape(atom_pos)[0])
    tt = 0
    for t in range(0, np.shape(atom_pos)[0]):
        L[tt] = np.around(np.dot(planN, atom_pos[t]), decimals=4)
        tt = tt + 1

    Le = unique(np.abs(L))
    lay = ui.layers_entry.text().split(",")
    cc = np.int(lay[0])
    dd = np.int(lay[1])

    for y in range(cc, dd):
        for i in range(0, np.shape(atom_pos)[0]):

            if (np.around(np.dot(planN, atom_pos[i]), decimals=4)) == Le[y]:

                Dz = np.vstack((Dz, atom_pos[i]))

    for j in range(1, np.shape(Dz)[0]):
        C = np.vstack((C, coord_ortho(Dz[j, :], plan)))
        D0 = np.vstack((D0, 1 + np.abs(np.around(np.dot(planN, Dz[j]), decimals=4))))
    C = np.delete(C, (0), axis=0)
    D0 = np.delete(D0, (0), axis=0)
    Dz = np.delete(Dz, (0), axis=0)
    F = np.append(C, D0, axis=1)

    return F, Dz, atom_pos


def getFileName():
    global varname
    varname = QtWidgets.QFileDialog.getOpenFileName(None, 'OpenFile')
    return varname

#######################################
#
# Save image as png
#
#########################################


def image_save():
    filename = QtWidgets.QFileDialog.getSaveFileName(Index, "Save file", "", ".png")
    f = str(filename[0]) + ".png"
    canvas.print_figure(f)


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
###########################################################
#
# Structure
#
##############################################################


def structure(item):
    global x0, var_hexa, d_label_var, e_entry

    ui.abc_entry.setText(str(item[1]) + ',' + str(item[2]) + ',' + str(item[3]))
    ui.alphabetagamma_entry.setText(str(item[4]) + ',' + str(item[5]) + ',' + str(item[6]))


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    sys.excepthook = except_hook
    QtWidgets.qApp.setApplicationName("Crystal")
    Index = QtWidgets.QMainWindow()
    ui = crystalUI.Ui_Crystal()
    ui.setupUi(Index)
    figure = plt.figure()
    canvas = FigureCanvas(figure)
    ui.mplvl.addWidget(canvas)
    toolbar = NavigationToolbar(canvas, canvas)
    toolbar.setMinimumWidth(601)


# Read structure file

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

    ui.actionSave_figure.triggered.connect(image_save)
    varname = 0
    ui.calculate_button.clicked.connect(calcul)
    ui.draw_button.clicked.connect(trace)
    ui.changeStruct_button.clicked.connect(getFileName)
    ui.abc_entry.setText('1,1,1')
    ui.alphabetagamma_entry.setText('90,90,90')
    ui.plane_entry.setText('1,1,1')
    ui.size_entry.setText('2,2,2')
    ui.layers_entry.setText('0,2')
    ui.markers_entry.setText('30')
    ui.precision_entry.setText('5')
    Index.show()
    sys.exit(app.exec_())
