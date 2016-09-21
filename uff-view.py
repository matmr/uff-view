"""
Created on 13. maj 2014

@author: Matjaz


TODO:
    -- types of fields
    -- order of columns
    -- connecting coor. sys. with nodes
    -- drawing coor. sys.
    -- basic elements on graphs

"""
import sys

import colorsys

import pandas as pd

from PySide import QtGui, QtCore

import pyqtgraph as pg

import pyqtgraph.dockarea as da

import pyqtgraph.opengl as gl

import numpy as np


import lib.modaldata as uff

# backend = 'pyside'
# vapp = vv.use(backend)

# TODO: deleting from table jumps elsewhere.
# TODO: Adding does not refresh.

# -- CONSTANTS
# .. Which fields to show for measurements index.
MAIN_WINDOW_TITLE = 'UFF View'
MEASUREMENT_INDEX_SELECT = ['model_id', 'uffid', 'ref_node', 'rsp_node', 'field_type', 'num_pts']
LINES_SELECT = ['model_id', 'uffid', 'field_type', 'trace_num', 'color', 'n_nodes']
FONT_TABLE_FAMILY = 'Consolas'
FONT_TABLE_SIZE = 13
ABOUT_DIALOG_TITLE = "About Uff View"
ABOUT_DIALOG = ("<p>The <b>Uff View</b> -- explore content of .uff/.unv files</p>"
                "<p>Some stuff to try:<br />"
                "-- File->Open uff file<br />"
                "-- Select one or more entries and press show</p>"
                "...")
OVERVIEW_LIST_FONT_FAMILY = 'Consolas'
OVERVIEW_LIST_FONT_SIZE = 10

_CUBE = np.array([[[-1.0, -1.0, -1.0],
                  [-1.0, -1.0, 1.0],
                  [-1.0, 1.0, 1.0]],
                 [[1.0, 1.0, -1.0],
                  [-1.0, -1.0, -1.0],
                  [-1.0, 1.0, -1.0]],
                 [[1.0, -1.0, 1.0],
                  [-1.0, -1.0, -1.0],
                  [1.0, -1.0, -1.0]],
                 [[1.0, 1.0, -1.0],
                  [1.0, -1.0, -1.0],
                  [-1.0, -1.0, -1.0]],
                 [[-1.0, -1.0, -1.0],
                  [-1.0, 1.0, 1.0],
                  [-1.0, 1.0, -1.0]],
                 [[1.0, -1.0, 1.0],
                  [-1.0, -1.0, 1.0],
                  [-1.0, -1.0, -1.0]],
                 [[-1.0, 1.0, 1.0],
                  [-1.0, -1.0, 1.0],
                  [1.0, -1.0, 1.0]],
                 [[1.0, 1.0, 1.0],
                  [1.0, -1.0, -1.0],
                  [1.0, 1.0, -1.0]],
                 [[1.0, -1.0, -1.0],
                  [1.0, 1.0, 1.0],
                  [1.0, -1.0, 1.0]],
                 [[1.0, 1.0, 1.0],
                  [1.0, 1.0, -1.0],
                  [-1.0, 1.0, -1.0]],
                 [[1.0, 1.0, 1.0],
                  [-1.0, 1.0, -1.0],
                  [-1.0, 1.0, 1.0]],
                 [[1.0, 1.0, 1.0],
                  [-1.0, 1.0, 1.0],
                  [1.0, -1.0, 1.0]]])


def colorize_overview(item):
    '''Return a color based on uff field type.'''
    if '151' in item or '164' in item:
        return QtGui.QColor('yellow')
    elif '15' in item or '18' in item or '2420' in item or '2411' in item:
        return QtGui.QColor('green')
    elif '58' in item:
        return QtGui.QColor('pink')
    elif '82' in item:
        return QtGui.QColor('cyan')
    elif '55' in item:
        return QtGui.QColor('red')


class MainWindow(QtGui.QMainWindow):
    '''Main GUI class, all QT elements are here (except data visualization).'''

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.init_dock()
        self.init_main_widget()
        self.setCentralWidget(self.main_widget_obj)

        self.init_menu_bar()

        self.statusBar().showMessage('Ready')

        self.setGeometry(50, 50, 1350, 500)
        self.setWindowTitle(self.tr(MAIN_WINDOW_TITLE))
        self.show()

    def init_menu_bar(self):
        '''Initialize menu bar.'''
        # .. Add exit option.
        exit_action = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)

        # .. Add open file option.
        open_action = QtGui.QAction(QtGui.QIcon('open.png'), '&Open...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.setStatusTip('Open file ...')
        open_action.triggered.connect(self.getFiles)

        menubar = self.menuBar()
        file_menu = menubar.addMenu('&File')
        file_menu.addAction(open_action)
        file_menu.addAction(exit_action)

        # .. Add about.
        about_action = QtGui.QAction(QtGui.QIcon('about.png'), '&About', self)
        about_action.setStatusTip('About')
        about_action.triggered.connect(self.about)

        help_menu = menubar.addMenu('&Help')
        help_menu.addAction(about_action)

    def init_dock(self):
        '''Initialize the dock window.'''
        dock = QtGui.QDockWidget('Graphs', self)
        dock.setAllowedAreas(
            QtCore.Qt.BottomDockWidgetArea | QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)

        # .. Prepare the DockArea to be used with pyqtgraph.
        self.dock_area = da.DockArea()
        self.dock_area.setMinimumSize(QtCore.QSize(600, 500))

        dock.setWidget(self.dock_area)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)

    def getFiles(self):
        '''File dialog for opening uff files. Updates data object, also updates status bar and window title appropriately.'''
        self.file_name, filtr = QtGui.QFileDialog.getOpenFileName(self, self.tr("Open File"), "/.",
                                                                  self.tr("Universal File Format (*.uff *.unv *.txt)"))
        self.statusBar().showMessage('Reading file ...')
        self.update_table()
        self.setWindowTitle(self.tr("%s -- %s" % (MAIN_WINDOW_TITLE, self.file_name,)))
        self.statusBar().showMessage('Ready')

    def about(self):
        '''About dialog.'''
        QtGui.QMessageBox.about(self, ABOUT_DIALOG_TITLE, ABOUT_DIALOG)

    def init_main_widget(self):
        '''Initialize the center widget.'''
        self.main_widget_obj = QtGui.QWidget()

        # .. Send DockArea handler to the DrawSelection class,
        #    in charge for plotting.
        self.draw_selection = DrawSelection(self.dock_area)

        # .. Put up list widget.
        self.list_widget = QtGui.QListView(self.main_widget_obj)
        self.list_model = QtGui.QStandardItemModel(self.list_widget)
        self.list_widget.setModel(self.list_model)
        font = QtGui.QFont(OVERVIEW_LIST_FONT_FAMILY, OVERVIEW_LIST_FONT_SIZE)
        self.list_widget.setFont(font)
        self.list_widget.setMinimumSize(20, 10)
        self.list_widget.setMaximumWidth(200)
        self.list_widget.setEnabled(False)

        # .. Initialize tabs, using the factory class, which puts in one
        #    table view with model for reading from pandas.
        self.geometry_table_view, self.geometry_table_model = self.tab_factory()
        self.analysis_table_view, self.analysis_table_model = self.tab_factory()
        self.lines_table_view, self.lines_table_model = self.tab_factory()
        self.info_table_view, self.info_table_model = self.tab_factory()
        self.measurement_table_view, self.measurement_table_model = self.tab_factory()

        # .. Put up tab widget.
        self.tabWidget = QtGui.QTabWidget()

        # .. Add all tabs.
        self.tabWidget.addTab(self.geometry_table_view, self.tr("Geometry"))
        self.tabWidget.addTab(self.measurement_table_view, self.tr("Measurements"))
        self.tabWidget.addTab(self.analysis_table_view, self.tr("Analyses"))
        self.tabWidget.addTab(self.lines_table_view, self.tr("Lines"))
        self.tabWidget.addTab(self.info_table_view, self.tr("Info"))

        self.tabWidget.setMinimumSize(900, 350)

        # .. Move data to the plotting object DrawSelection instance.
        def draw_selected_rows():
            self.draw_selection.dataadd(self.data)
            if self.tabWidget.currentIndex() == 1:
                self.draw_selection.containeradd(self.measurement_table_view.selectionModel())
                self.draw_selection.draw()
            elif self.tabWidget.currentIndex() == 0:
                self.draw_selection.containeradd(self.geometry_table_view.selectionModel())
                self.draw_selection.drawgeom(self.cube_scale)

        showButton = QtGui.QPushButton(self.tr('Show'))

        # .. Signal for plotting. (see above)
        showButton.pressed.connect(draw_selected_rows)

        # .. Prepare layout.
        buttonLayout = QtGui.QHBoxLayout()
        buttonLayout.addStretch(1)
        buttonLayout.addWidget(showButton)

        widgetLayout = QtGui.QHBoxLayout()
        widgetLayout.addWidget(self.list_widget)
        widgetLayout.addWidget(self.tabWidget)

        mainLayout = QtGui.QVBoxLayout()
        mainLayout.addLayout(widgetLayout)
        mainLayout.addLayout(buttonLayout)
        self.main_widget_obj.setLayout(mainLayout)

    def tab_factory(self):
        '''Set up a tab with QTableView and TableModel. Each tab
        window is made the same.'''
        cdf = pd.DataFrame(columns=['None'])
        table_model = TableModel(self)
        table_model.update(cdf)
        table_view = QtGui.QTableView()
        table_view.setModel(table_model)

        table_view.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        font = QtGui.QFont(FONT_TABLE_FAMILY, FONT_TABLE_SIZE)
        font1 = QtGui.QFont(FONT_TABLE_FAMILY, FONT_TABLE_SIZE, QtGui.QFont.Bold)
        table_view.horizontalHeader().setFont(font1)
        table_view.setFont(font)
        table_view.setAlternatingRowColors(True)
        table_view.setSortingEnabled(True)

        return table_view, table_model

    def update_table(self):
        '''Read data from uff file, using ModalData object, then populate
        tables and list.'''
        # .. Get uffdata object with data from uff, written into pandas tables.
        df = uff.ModalData()
        df.import_uff(self.file_name)
        self.data = df

        # .. Update list widget (file structure).
        # TODO: Make abstract model for this list!!
        self.list_model.clear()
        for item in df.file_structure:
            qitem = QtGui.QStandardItem(item)
            qitem.setSelectable(False)
            qitem.setEditable(False)
            qitem.setBackground(colorize_overview(item))
            self.list_model.appendRow(qitem)
        self.list_widget.setEnabled(True)

        # .. For every table that is populated (True), fill the appropriate
        #    table view, otherwise grey it out.
        # TODO: This thing below could fit in one function?
        if 'geometry' in df.uff_import_tables:
            # Calcuate mean distance between points. Useful for determining size of cubes.
            x_mean = df.tables['geometry'].x.mean()
            y_mean = df.tables['geometry'].y.mean()
            z_mean = df.tables['geometry'].z.mean()
            self.cube_scale = (x_mean + y_mean + z_mean) / 3 / 50

            # Position the geometry model in the centre of the coor. sys.
            df.tables['geometry'].x = df.tables['geometry'].x - x_mean
            df.tables['geometry'].y = df.tables['geometry'].y - y_mean
            df.tables['geometry'].z = df.tables['geometry'].z - z_mean

            self.geometry_table_model.update(df.tables['geometry'])
            self.geometry_table_view.resizeColumnsToContents()
            self.tabWidget.setTabEnabled(0, True)
        else:
            self.tabWidget.setTabEnabled(0, False)

        if 'measurement_index' in df.uff_import_tables:
            self.measurement_table_model.update(df.tables['measurement_index'][MEASUREMENT_INDEX_SELECT])
            self.measurement_table_view.resizeColumnsToContents()
            self.tabWidget.setTabEnabled(1, True)
        else:
            self.tabWidget.setTabEnabled(1, False)

        if 'analysis_index' in df.uff_import_tables:
            self.analysis_table_model.update(df.tables['analysis_index'])
            self.analysis_table_view.resizeColumnsToContents()
            self.tabWidget.setTabEnabled(2, True)
        else:
            self.tabWidget.setTabEnabled(2, False)

        if 'lines' in df.uff_import_tables:
            grouped = df.tables['lines'].groupby(['trace_id'])
            self.lines_table_model.update(grouped.first()[LINES_SELECT])
            self.lines_table_view.resizeColumnsToContents()
            self.tabWidget.setTabEnabled(3, True)
        else:
            self.tabWidget.setTabEnabled(3, False)

        if 'info' in df.uff_import_tables:
            self.info_table_model.update(df.tables['info'])
            self.info_table_view.resizeColumnsToContents()
            self.tabWidget.setTabEnabled(4, True)
        else:
            self.tabWidget.setTabEnabled(4, False)


class DrawSelection(object):
    """Class for data visualization. It draws selected data (rows),
    which can be real/img response or points in 3D."""

    def __init__(self, dock_area):
        super(DrawSelection, self).__init__()

        self.container = None
        self.dock_area = dock_area
        self.dock_mag = da.Dock('')
        self.dock_area.addDock(self.dock_mag, 'above')
        self.dock_geom = da.Dock('')
        self.dock_area.addDock(self.dock_geom, 'above')

    def containeradd(self, smodel):
        self.rownr = [row.row() for row in smodel.selectedRows()]

    def dataadd(self, data):
        self.data = data

    def drawgeom(self, cube_scale=0.01):
        xyz = self.data.tables['geometry'][['x', 'y', 'z']].iloc[self.rownr, :].values

        self.dock_geom.close()
        self.dock_geom = da.Dock('Geometry')
        self.dock_area.addDock(self.dock_geom, 'above')

        glview = gl.GLViewWidget()
        self.dock_geom.addWidget(glview)

        cube = _CUBE * cube_scale
        cube = np.tile(cube, (xyz.shape[0], 1, 1))
        xyz = xyz.repeat(36, axis=0).reshape(cube.shape[0], 3, 3)

        m1 = gl.GLMeshItem(vertexes=(cube + xyz))
        glview.addItem(m1)

    #
    def draw(self):
        # .. Set colors.
        N = len(self.rownr)

        i = 0
        self.dock_mag.close()
        self.dock_mag = da.Dock('Magnitude and Phase')
        self.dock_area.addDock(self.dock_mag, 'above')

        graphics_view = pg.GraphicsView()

        layout = QtGui.QGridLayout()

        fig_mag = pg.PlotWidget(name='Magnitude')
        fig_mag.setMinimumHeight(325)
        fig_mag.setLabel('left', 'Magnitude dB')
        fig_mag.setLabel('bottom', 'Frequency', units='Hz')
        fig_mag.setLogMode(x=False, y=True)

        fig_ang = pg.PlotWidget(name='Angle')
        fig_ang.setMinimumHeight(125)
        fig_ang.setMaximumHeight(300)
        fig_ang.setLabel('left', 'Angle', units='rad')
        fig_ang.setLabel('bottom', 'Frequency', units='Hz')
        fig_ang.setXLink('Magnitude')

        layout.addWidget(fig_mag, 0, 0, 10, 1)
        layout.addWidget(fig_ang, 10, 0, 11, 1)
        graphics_view.setLayout(layout)

        self.dock_mag.addWidget(graphics_view)
        # .. Prepare data.
        # TODO: optimize reading here
        self.fig_container = []
        for row in self.rownr:
            uffid = self.data.tables['measurement_index']['uffid'][row]
            xy = self.data.tables['measurement_values'][['frq', 'amp']][
                self.data.tables['measurement_values']['uffid'] == uffid].values.astype('complex')

            y = np.abs(xy[:, 1])
            x = xy[:, 0].real

            mask = y > 1e-8

            # v = np.log10(list(y[mask]))
            fig_mag.plot(x[mask], y[mask], pen=pg.colorTuple(pg.intColor(i, hues=N)))

            fig_ang.plot(xy[:, 0][mask].real, np.angle(xy[:, 1][mask]), pen=pg.colorTuple(pg.intColor(i, hues=N)))

            #
            i += 1


class TableModel(QtCore.QAbstractTableModel):
    '''Table model that suits all tables (for now). It specifies
    access to data and some other stuff.'''

    def __init__(self, parent, *args):
        super(TableModel, self).__init__(parent, *args)
        self.datatable = None

    def update(self, dataIn):
        self.emit(QtCore.SIGNAL("layoutAboutToBeChanged()"))
        self.datatable = dataIn
        self.dataChanged.emit(0, 0)
        self.emit(QtCore.SIGNAL("layoutChanged()"))

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.datatable.index)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.datatable.columns.values)

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self.datatable.columns[col]
        return None

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        elif role != QtCore.Qt.DisplayRole:
            return None

        i = index.row()
        j = index.column()
        return '{0}'.format(self.datatable.iat[i, j])

    def sort(self, col, order):
        """sort table by given column number col"""
        self.emit(QtCore.SIGNAL("layoutAboutToBeChanged()"))
        if order == QtCore.Qt.DescendingOrder:
            self.datatable = self.datatable.sort_values(by=self.datatable.columns[col], ascending=0)
        else:
            self.datatable = self.datatable.sort_values(by=self.datatable.columns[col])
        self.emit(QtCore.SIGNAL("layoutChanged()"))


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    tabwindow = MainWindow()
    tabwindow.show()
    sys.exit(app.exec_())