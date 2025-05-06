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
    global vec, varname, atom0, Dstar, taille, zoom, EL, Dz, dsc_cond, V, T, coi, Ma, Ma0, Ma1, Ma2, ELr, theta
    ui.v_listBox.clear()
    if varname != 0:
        f_space = open(varname[0], "r")
    else:
        varname = getFileName()
        f_space = open(varname[0], "r")

    crist = []

    for line in f_space:
        crist.append(list(map(str, line.split())))

    f_space.close()
    vec = []
    atom0 = []
    for i in range(0, len(crist)):
        if np.size(crist[i]) == 3:
            vec.append(crist[i])
        else:
            atom0.append(crist[i])

    vec = np.array(vec, float)
    E = np.array([0, 0, 0, 0])
    EL = np.array([0, 0, 0, 0, 0])
    H = np.array([[0]])
    Dz = np.array([0, 0, 0])
    V = np.zeros((1, 10))
    if ui.dsc_box.isChecked() is False:
        atom0 = np.array(atom0, float)
        maxi = int(atom0[np.shape(atom0)[0] - 1, 0])
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
    if ui.dsc_box.isChecked():
        theta = np.float64(ui.angle_entry.text())
        theta = theta * np.pi / 180
        ELr = np.dot(EL[:, 0:3], Rot(theta, 0, 0, 1))
        M = unique(EL[:, 3])
        Ma0 = []
        Ma1 = []
        Ma2 = []
        m = (0, 1, 2, 3, 4)
        abc = ui.abc_entry.text().split(",")
        a = np.float64(abc[0])
        b = np.float64(abc[1])
        c = np.float64(abc[2])
        ee = np.float64(ui.precision_entry.text())
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
                        Ma0.append(0)
                        Ma1.append(0)
                    else:
                        Ma0.append(m[t])
                        Ma1.append(m[t])
                    if ui.layers_box.isChecked():
                        Ma2.append(t)
        Ma1 = np.array(Ma1)
        Ma2 = np.array(Ma2)
        Ma0 = np.array(Ma0)


def draw():
    global fi
    minx = np.min(EL[:, 0])
    maxx = np.max(EL[:, 0])
    miny = np.min(EL[:, 1])
    maxy = np.max(EL[:, 1])
    fi.axis([minx, maxx, miny, maxy])
    trace()


def trace():
    global vec, varname, atom0, Dstar, D, taille, zoom, EL, Dz, fi, V, hexa, plan, planN
    minx, maxx = fi.get_xlim()
    miny, maxy = fi.get_ylim()
    fi.clear()
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
                if hexa == 1:
                    i1 = (2 * at[0] - at[1]) / 3
                    i2 = (2 * at[1] - at[0]) / 3
                    i3 = at[2]
                else:
                    i1, i2, i3 = at[0], at[1], at[2]

                vector = str(np.around(i1, decimals=3)) + ',' + str(np.around(i2, decimals=3)) + ',' + str(np.around(i3, decimals=3))
                fi.annotate(vector, (EL[q, 0], EL[q, 1]))
    m = ['o', 's', '^', '*', 'h']
    if ui.dsc_box.isChecked():
        for i in range(0, 5):
            fi.scatter(ELr[Ma0 == i, 0], ELr[Ma0 == i, 1], s=sim, marker=m[i], color='white', edgecolor='black')
            fi.scatter(EL[Ma1 == i, 0], EL[Ma1 == i, 1], s=sim, marker=m[i], color='black', edgecolor='black')
        if ui.coincidence_box.isChecked():
            for z in coi:
                fi.scatter(EL[z, 0], EL[z, 1], s=sim * 1.5, marker=m[Ma0[z]], color='blue', edgecolor='black')

        if ui.layers_box.isChecked():
            for y in range(0, EL.shape[0]):
                fi.text(EL[y, 0], EL[y, 1], Ma2[y])
                fi.text(ELr[y, 0], ELr[y, 1], Ma2[y])

        if ui.labels_box.isChecked():
            for q in range(0, np.shape(EL)[0]):
                at = Dz[q, :]
                at = np.dot(at, Dstar)
                if hexa == 1:
                    i1 = (2 * at[0] - at[1]) / 3
                    i2 = (2 * at[1] - at[0]) / 3
                    i3 = at[2]
                else:
                    i1, i2, i3 = at[0], at[1], at[2]
                vector = str(np.around(i1, decimals=3)) + ',' + str(np.around(i2, decimals=3)) + ',' + str(np.around(i3, decimals=3))
                if ui.G2_checkBox.isChecked():
                    fi.annotate(vector, (ELr[q, 0], ELr[q, 1]))
                else:
                    fi.annotate(vector, (EL[q, 0], EL[q, 1]))

    if V.shape[0] > 0:
        for i in range(1, V.shape[0]):
            if V[i, 8] == 1:
                v = np.dot(Dstar, V[i, 0:3])
                v = v / np.linalg.norm(v) / V[i, 3]
            else:
                v = np.dot(D, V[i, 0:3]) / V[i, 3]
            t = np.dot(D, V[i, 5:8])
            L = np.dot(v, planN)
            Lt = np.dot(t, planN)
            vec_p = v - L * planN
            vec_t = t - Lt * planN
            if V[i, 4] == 2:
                proj = np.dot(Rot(-theta, 0, 0, 1), coord_ortho(vec_p, plan))
                proj_t = np.dot(Rot(-theta, 0, 0, 1), coord_ortho(vec_t, plan))
                c = 'red'
            else:
                proj = coord_ortho(vec_p, plan)
                proj_t = coord_ortho(vec_t, plan)
                c = 'blue'

            fi.arrow(proj_t[0], proj_t[1], proj[0], proj[1], width=0.01, head_width=0.03, head_length=0.03, length_includes_head=True, color=c)
            if ui.label_checkBox.isChecked():
                if hexa == 1 and V[i, 8] == 0:
                    sa = '[' + str((2 * V[i, 0] - V[i, 1]) / 3) + ',' + str((2 * V[i, 1] - V[i, 0]) / 3) + ',' + str(V[i, 2]) + ']' + '_' + str(int(V[i, 4]))
                else:
                    if V[i, 8] == 0:
                        sa = '[' + str(V[i, 0]) + ',' + str(V[i, 1]) + ',' + str(V[i, 2]) + ']' + '_' + str(int(V[i, 4]))
                    else:
                        sa = '(' + str(V[i, 0]) + ',' + str(V[i, 1]) + ',' + str(V[i, 2]) + ')' + '_' + str(int(V[i, 4]))
                fi.annotate(sa, (proj[0] + proj_t[0], proj[1] + proj_t[1]), color=c)
        fi.axis('off')
        fi.axis('equal')
        fi.axis([minx, maxx, miny, maxy])
        fi.figure.canvas.draw()


def add_vec():
    global V, hexa
    if ui.G2_draw_checkBox.isChecked() and ui.dsc_box.isChecked():
        g = 2
    else:
        g = 1
    v0 = np.float64(ui.v_entry.text().split(","))
    t0 = np.float64(ui.t_entry.text().split(","))
    den_v = np.float64(ui.v_denom_entry.text())
    if ui.normal_checkBox.isChecked() is False:
        if hexa == 1:
            i1 = 2 * v0[0] + v0[1]
            i2 = 2 * v0[1] + v0[0]
            i3 = v0[2]
            t1 = 2 * t0[0] + t0[1]
            t2 = 2 * t0[1] + t0[0]
            t3 = t0[2]
        else:
            i1, i2, i3 = v0[0], v0[1], v0[2]
            t1, t2, t3 = t0[0], t0[1], t0[2]
        v = np.dot(D, np.array([i1, i2, i3])) / den_v
        L = np.around(np.dot(v, planN), decimals=3)
        V = np.vstack((V, np.array([i1, i2, i3, den_v, g, t1, t2, t3, 0, L])))
        lp, rp = '[', ']'
    else:
        v = np.dot(Dstar, v0) / den_v
        L = np.around(np.dot(v, planN), decimals=3)
        V = np.vstack((V, np.array([v0[0], v0[1], v0[2], den_v, g, t0[0], t0[1], t0[2], 1, L])))
        lp, rp = '(', ')'

    s = str('1/' + str(int(den_v)) + lp + str(v0[0]) + ',' + str(v0[1]) + ',' + str(v0[2]) + rp + str(g) + '   ' + lp + str(t0[0]) + ',' + str(t0[1]) + ',' + str(t0[2]) + rp + ',' + str(L))
    ui.v_listBox.addItem(s)
    cur = ui.v_listBox.currentRow()
    ui.v_listBox.setCurrentRow(cur + 1)
    trace()


def remove_vec():
    global V
    if ui.v_listBox.currentRow() == -1:
        cr = 1
    else:
        cr = ui.v_listBox.currentRow() + 1
    if V.shape[0] > 1:
        V = np.delete(V, cr, axis=0)
        ui.v_listBox.takeItem(cr - 1)
    trace()


def calcul_atom(atom):
    global Dstar, D, varname, C, D0, planN, plan, vec, atom_pos, hexa

    abc = ui.abc_entry.text().split(",")
    a = np.float64(abc[0])
    b = np.float64(abc[1])
    c = np.float64(abc[2])
    alphabetagamma = ui.alphabetagamma_entry.text().split(",")
    alpha = np.float64(alphabetagamma[0])
    beta = np.float64(alphabetagamma[1])
    gamma = np.float64(alphabetagamma[2])
    alp = alpha * np.pi / 180
    bet = beta * np.pi / 180
    gam = gamma * np.pi / 180
    if a == b != c and alpha == beta == 90 and gamma == 120:
        hexa = 1
    else:
        hexa = 0

    V = a * b * c * np.sqrt(1 - (np.cos(alp)**2) - (np.cos(bet))**2 - (np.cos(gam))**2 + 2 * b * c * np.cos(alp) * np.cos(bet) * np.cos(gam))
    D = np.array([[a, b * np.cos(gam), c * np.cos(bet)], [0, b * np.sin(gam), c * (np.cos(alp) - np.cos(bet) * np.cos(gam)) / np.sin(gam)], [0, 0, V / (a * b * np.sin(gam))]])
    Dstar = np.transpose(np.linalg.inv(D))

    n = ui.size_entry.text().split(",")
    na = int(n[0])
    nb = int(n[1])
    nc = int(n[2])

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
    h = np.float64(pl[0])
    k = np.float64(pl[1])
    l = np.float64(pl[2])

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
    cc = int(lay[0])
    dd = int(lay[1])

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
    global x0, d_label_var, e_entry

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
    fi = figure.add_subplot(111)
    fi.axis("off")
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
    ui.draw_button.clicked.connect(draw)
    ui.changeStruct_button.clicked.connect(getFileName)
    ui.add_v_Button.clicked.connect(add_vec)
    ui.remove_v_Button.clicked.connect(remove_vec)
    ui.abc_entry.setText('1,1,1')
    ui.alphabetagamma_entry.setText('90,90,90')
    ui.plane_entry.setText('1,1,1')
    ui.size_entry.setText('2,2,2')
    ui.layers_entry.setText('0,2')
    ui.markers_entry.setText('30')
    ui.precision_entry.setText('5')
    ui.t_entry.setText('0,0,0')
    Index.show()
    sys.exit(app.exec_())
