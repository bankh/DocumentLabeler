# Copyright (C) <2022-present> Sinan Bank, Craftnetics Inc., Colorado State University, Fort Collins
# Copyright (C) <2015-2022> Tzutalin
# Copyright (C) <2013-2015> MIT, Computer Science and Artificial Intelligence Laboratory. Bryan Russell, 
# Antonio Torralba, William T. Freeman. Permission is hereby granted, free of charge, to any person obtaining 
# a copy of this software and associated documentation files (the "Software"), to deal in the Software without 
# restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, 
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do 
# so, subject to the following conditions: 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO 
# EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN 
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
# OTHER DEALINGS IN THE SOFTWARE.

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# pyrcc5 -o libs/resources.py resources.qrc

# ToDo: (Higher Priority)
    # Correct the duplication bug on the predefined key (label) list

    # Generate a console in a tab to access the variables, execute methods, 
    # and send variables [Implementation passed, details are needed]

    # Add comparison of the estimate for the results of the inference  could 
    # be added to a tab
        # Show inference results
        # Show the estimates for each token
        # Heatmap based on the estimates (colder colors are incorrect ones and
        # warmer colors are correct ones )
# ToDo: (Medium Priority)
    # Add backend code of the front-end menu [Python notebook algos passed 
    # - WiP]

    # Add label statistics per page

# ToDo: (Lower Priority - Future Work)
    # Add ruler on the canvas and it needs to be able to toggled via UI

    # Add a gridline from ruler extension on canvas connected to a radio 
    # button to toggle on/off the gridline

    # Add a tab to include images processing tools (Second tab)
        # Adding vision tools with another class
        # https://github.com/huyhoang1905/Image-Processing-with-OpenCV-Python-and-PyQt5
        # https://pyshine.com/Make-GUI-for-OpenCv-And-PyQt5/

    # Add a tab to include pre-processing tool to import file from pdf and 
    # (First tab)
        # Adding a pdf processing tools with another class 

import argparse
import ast
import codecs
import json
import os.path
import platform
import subprocess
import sys
import xlrd
from functools import partial
from copy import deepcopy
import os
import shutil
from PIL import Image

from PyQt5.QtCore import QSize, Qt, QPoint, QByteArray, QTimer, QFileInfo, QPointF, QProcess, QMargins, QRect
from PyQt5.QtGui import QImage, QCursor, QPixmap, QImageReader
from PyQt5.QtWidgets import QMainWindow, QListWidget, QVBoxLayout, QToolButton, QHBoxLayout, QDockWidget, QWidget, \
                            QSlider, QGraphicsOpacityEffect, QMessageBox, QListView, QScrollArea, QWidgetAction, \
                            QApplication, QLabel, QGridLayout, QFileDialog, QListWidgetItem, QComboBox, QDialog, \
                            QAbstractItemView, QTabWidget, QVBoxLayout, QSizePolicy, QSpacerItem, QProgressBar

# Embed Python Console for Active  Changes
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager
from IPython.lib import guisupport

# To import packages setting the environment 
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../DocumentLabel')))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from models.modelRunner import Inference

from libs.constants import *
from libs.utils import *
from libs.labelColor import label_colormap
from libs.settings import Settings
from libs.shape import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR, DEFAULT_LOCK_COLOR
from libs.stringBundle import StringBundle
from libs.canvas import Canvas
from libs.zoomWidget import ZoomWidget
from libs.autoDialog import AutoDialog
from libs.ocrDialog import OCRDialog
from libs.colorDialog import ColorDialog
from libs.ustr import ustr
from libs.hashableQListWidgetItem import HashableQListWidgetItem
from libs.editinlist import EditInList
from libs.unique_label_qlist_widget import UniqueLabelQListWidget
from libs.keyDialog import KeyDialog
from libs.consoleTab import ConsoleTab
from libs.progress_bar import ExportProgress

__appname__ = 'Document_Labeler'

LABEL_COLORMAP = label_colormap()

class MainWindow(QMainWindow):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self,
                 lang="en",
                 gpu=False,
                 kie_mode=False,
                 default_filename=None,
                 default_predefined_class_file=None,
                 default_save_dir=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)
        self.setWindowState(Qt.WindowMaximized)  # Set window max
        self.activateWindow()                    # DocumentLabeler goes to the front when activate

        # Load setting in the main thread
        self.settings = Settings()
        self.settings.load()
        settings = self.settings
        self.lang = lang

        # Load string bundle for i18n
        if lang not in ['ch', 'en']:
            lang = 'en'
        self.stringBundle = StringBundle.getBundle(localeStr='zh-CN' if lang == 'ch' else 'en')  # 'en'
        getStr = lambda strId: self.stringBundle.getString(strId)

        # KIE setting
        self.kie_mode = kie_mode
        self.key_previous_text = ""
        self.existed_key_cls_set = set()
        self.key_dialog_tip = getStr('keyDialogTip')
        self.defaultSaveDir = default_save_dir

        # For loading all image under a directory
        self.mImgList = []
        self.mImgList5 = []
        self.dirname = None
        self.textHist = []
        self.lastOpenDir = None
        self.result_dic = []
        self.result_dic_locked = []
        self.changeFileFolder = False
        self.haveAutoReced = False
        self.labelFile = None
        self.currIndex = 0

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False
        self._beginner = True
        self.screencastViewer = self.getAvailableScreencastViewer()
        self.screencast = "https://github.com/bankh/DocumentLabeler/DocumentLabeler"

        # Load predefined classes to the list
        self.loadPredefinedClasses(default_predefined_class_file)

        # Main widgets and related state.
        self.ocrDialog = OCRDialog(parent=self, listItem=self.textHist)
        self.autoDialog = AutoDialog(parent=self)

        self.itemsToShapes = {}
        self.shapesToItems = {}
        self.itemsToShapesbox = {}
        self.shapesToItemsbox = {}
        self.prevText = getStr('tempLabel')
        self.noLabelText = getStr('nullLabel')
        self.model = 'pick'                                 # Default model we will change this
        self.PPreader = None
        self.autoSaveNum = 5

        # [Part of ToDo] Add the model loading here
        # Inference(--ckpt      model path, 
        #           --bt        boxes_and_transcripts path, 
        #           --impt      image_path, 
        #           --bs        batch_size, 
        #           --output_path,
        #           --config_file)   
        self.inference = Inference(self.model,
                                   checkpoint_path='/mnt/data_drive/CSU_PhD/research/software/DocumentEngineering/DocumentLabeler_FinalDev/models/pick/saved/PICK-default/models/SROIE_test_0521_015255/model_best.pth',
                                   bt_path='/mnt/data/Data-Document/GeneralDocument/SROIE_PICK/boxes_and_transcripts',
                                   impt_path='/mnt/data/Data-Document/GeneralDocument/SROIE_PICK/images',
                                   batch_size=2,
                                   output_path='/mnt/data_drive/CSU_PhD/research/software/DocumentEngineering/DocumentLabeler_FinalDev/output',
                                   config_file='/mnt/data_drive/CSU_PhD/research/software/DocumentEngineering/DocumentLabeler_FinalDev/models/configs/pick/pick_config.yaml')

        # self.ocr = PaddleOCR(use_pdserving=False,
        #                      use_angle_cls=True,
        #                      det=True,
        #                      cls=True,
        #                      use_gpu=gpu,
        #                      lang=lang,
        #                      show_log=False)
        # self.table_ocr = PPStructure(use_pdserving=False,
        #                              use_gpu=gpu,
        #                              lang=lang,
        #                              layout=False,
        #                              show_log=False)

        # if os.path.exists('./data/paddle.png'):
        #     result = self.ocr.ocr('./data/paddle.png', cls=True, det=True)
        #     result = self.table_ocr('./data/paddle.png', return_ocr_result_in_table=True)

        #  ================== File List (left) ==================
        filelistLayout = QVBoxLayout()
        filelistLayout.setContentsMargins(0, 0, 0, 0)

        self.fileListWidget = QListWidget()
        self.fileListWidget.itemClicked.connect(self.fileitemDoubleClicked)
        self.fileListWidget.setIconSize(QSize(25, 25))
        filelistLayout.addWidget(self.fileListWidget)

        fileListContainer = QWidget()
        fileListContainer.setLayout(filelistLayout)
        self.fileListName = getStr('fileList')
        self.fileDock = QDockWidget(self.fileListName, self)
        self.fileDock.setObjectName(getStr('files'))
        self.fileDock.setWidget(fileListContainer)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.fileDock)

        #  ================== Key List (left bottom if enabled) ==================
        if self.kie_mode:
            self.keyList = UniqueLabelQListWidget()

            # set key list height
            key_list_height = int(QApplication.desktop().height() // 4)
            if key_list_height < 50:
                key_list_height = 50
            self.keyList.setMaximumHeight(key_list_height)

            self.keyListDockName = getStr('labelListTitle')
            self.keyListDock = QDockWidget(self.keyListDockName, self)
            self.keyListDock.setWidget(self.keyList)
            self.keyListDock.setFeatures(QDockWidget.NoDockWidgetFeatures)
            filelistLayout.addWidget(self.keyListDock)

        self.AutoRecognition = QToolButton()
        self.AutoRecognition.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.AutoRecognition.setIcon(newIcon('Auto'))

        autoRecLayout = QHBoxLayout()
        autoRecLayout.setContentsMargins(0, 0, 0, 0)
        autoRecLayout.addWidget(self.AutoRecognition)

        autoRecContainer = QWidget()
        autoRecContainer.setLayout(autoRecLayout)
        filelistLayout.addWidget(autoRecContainer)
        
        #  ================== Right Area  ==================
        listLayout = QVBoxLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)

        # Buttons
        self.editButton = QToolButton()

        self.reRecogButton = QToolButton()
        self.reRecogButton.setIcon(newIcon('reRec', 30))
        self.reRecogButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.reRecogButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.tableRecButton = QToolButton()
        self.tableRecButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.tableRecButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.newButton = QToolButton()
        self.newButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.newButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.createpolyButton = QToolButton()
        self.createpolyButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.createpolyButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.labelGroupButton = QToolButton()
        self.labelGroupButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.labelGroupButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.deleteGroupButton = QToolButton()
        self.deleteGroupButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.deleteGroupButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.mergeGroupButton = QToolButton()
        self.mergeGroupButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.mergeGroupButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.SaveButton = QToolButton()
        self.SaveButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        self.DelButton = QToolButton()
        self.DelButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # ================== Right Top Tool Box  ==================
        rightTopToolBox = QGridLayout()
        rightTopToolBox.addWidget(self.newButton, 0, 0, 2, 1)
        rightTopToolBox.addWidget(self.createpolyButton, 2, 0, 2, 1)
        rightTopToolBox.addWidget(self.labelGroupButton, 0, 2, 2, 1)
        rightTopToolBox.addWidget(self.reRecogButton,0, 1, 2, 1) 
        rightTopToolBox.addWidget(self.tableRecButton, 2, 1, 2, 1)
        rightTopToolBox.addWidget(self.deleteGroupButton, 2, 2, 2, 1)
        rightTopToolBox.addWidget(self.mergeGroupButton, 4, 2, 2, 1)

        rightTopToolBoxContainer = QWidget()
        rightTopToolBoxContainer.setLayout(rightTopToolBox)
        listLayout.addWidget(rightTopToolBoxContainer)
        
        #  ================== OCR Text List  ==================
        textIndexListlBox = QHBoxLayout()

        # Create and add a widget for showing current label item index
        self.indexList = QListWidget()
        self.indexList.setMaximumSize(30, 16777215)                         # limit max width
        self.indexList.setEditTriggers(QAbstractItemView.NoEditTriggers)    # no editable
        self.indexList.itemSelectionChanged.connect(self.indexSelectionChanged)
        self.indexList.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)    # no scroll Bar

        self.indexListDock = QDockWidget('No.', self)
        self.indexListDock.setWidget(self.indexList)
        self.indexListDock.setFeatures(QDockWidget.NoDockWidgetFeatures)

        textIndexListlBox.addWidget(self.indexListDock, 1)
        self.indexList.setViewportMargins(QMargins(0, 0, 0, 0))

        # no margin between two boxes
        textIndexListlBox.setSpacing(0)

        # Create and add a widget for showing current label items
        self.textList = EditInList()
        textListContainer = QWidget()
        textListContainer.setLayout(listLayout)
        self.textList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.textList.clicked.connect(self.textList.item_clicked)

        # Connect to itemChanged to detect checkbox changes.
        self.textList.itemChanged.connect(self.labelItemChanged)
        self.textList.setViewportMargins(QMargins(0, 0, 0, 0))
        self.textListDockName = getStr('recognitionResult')
        self.textListDock = QDockWidget(self.textListDockName, self)
        self.textListDock.setWidget(self.textList)
        self.textListDock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        textIndexListlBox.addWidget(self.textListDock, 10) # label list is wider than index list

        # enable textList drag_drop to adjust bbox order
        # Set the selection mode to single selection
        self.textList.setSelectionMode(QAbstractItemView.SingleSelection)
        # Enable drag and drop
        self.textList.setDragEnabled(True)
        # Set to accept drag and drop
        self.textList.viewport().setAcceptDrops(True)
        # Sets where the display will be placed
        self.textList.setDropIndicatorShown(True)
        # Set the drag and drop mode to move items, if not set, the default is to copy items
        self.textList.setDragDropMode(QAbstractItemView.InternalMove) 
        # trigger placement
        self.textList.model().rowsMoved.connect(self.drag_drop_happened)
        labelIndexListContainer = QWidget()
        labelIndexListContainer.setLayout(textIndexListlBox)
        listLayout.addWidget(labelIndexListContainer)
        # textList indexList 同步滚动 (synchronous scrolling)
        self.textListBar = self.textList.verticalScrollBar()
        self.indexListBar = self.indexList.verticalScrollBar()
        self.textListBar.valueChanged.connect(self.move_scrollbar)
        self.indexListBar.valueChanged.connect(self.move_scrollbar)

        #  ================== Detection Box  ==================
        self.BoxList = QListWidget()
        self.BoxList.itemSelectionChanged.connect(self.boxSelectionChanged)
        self.BoxList.itemDoubleClicked.connect(self.editBox)
        # Connect to itemChanged to detect checkbox changes.
        self.BoxList.itemChanged.connect(self.boxItemChanged)
        self.BoxListDockName = getStr('detectionBoxposition')
        self.BoxListDock = QDockWidget(self.BoxListDockName, self)
        self.BoxListDock.setWidget(self.BoxList)
        self.BoxListDock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        listLayout.addWidget(self.BoxListDock)

        # ================== Lower Right Area  ==================
        leftbtmtoolbox = QHBoxLayout()
        leftbtmtoolbox.addWidget(self.SaveButton)
        leftbtmtoolbox.addWidget(self.DelButton)
        leftbtmtoolboxcontainer = QWidget()
        leftbtmtoolboxcontainer.setLayout(leftbtmtoolbox)
        listLayout.addWidget(leftbtmtoolboxcontainer)
        self.dock = QDockWidget(getStr('boxText'), self)   # CHanged it to empty in string
        self.dock.setObjectName(getStr('labels'))
        self.dock.setWidget(textListContainer)

        # ================== Tabs Right  ======================
        tabs_right = QTabWidget()

        # =================== Tab1: Preprocess Tab ===================
        # Future work
        # Preprocess Tab
        preprocess_widget = QWidget()
        tabs_right.addTab(preprocess_widget,"Preprocess")
        
        # =================== Tab2: Computer Vision ==================
        # Future work
        # Computer Vision Tab
        CVedit_widget = QWidget()
        tabs_right.addTab(CVedit_widget,"CV Edit")

        # =================== Tab3: Boxes/OCR Edit ===================
        # Add the Pages tab to the tabs widget
        self.dock.setWidget(tabs_right)
        editBoxContainer = QWidget()
        editBoxContainer.setLayout(listLayout)
        tabs_right.addTab(editBoxContainer, "Boxes/OCR Edit")
        # tabs_right.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        # ================== Zoom Bar  ==================
        self.imageSlider = QSlider(Qt.Horizontal)
        self.imageSlider.valueChanged.connect(self.CanvasSizeChange)
        self.imageSlider.setMinimum(-9)
        self.imageSlider.setMaximum(510)
        self.imageSlider.setSingleStep(1)
        self.imageSlider.setTickPosition(QSlider.TicksBelow)
        self.imageSlider.setTickInterval(1)
        op = QGraphicsOpacityEffect()
        op.setOpacity(0.2)
        self.imageSlider.setGraphicsEffect(op)
        self.imageSlider.setStyleSheet("background-color:transparent")
        self.imageSliderDock = QDockWidget(getStr('ImageResize'), self)
        self.imageSliderDock.setObjectName(getStr('IR'))
        self.imageSliderDock.setWidget(self.imageSlider)
        self.imageSliderDock.setFeatures(QDockWidget.DockWidgetFloatable)
        self.imageSliderDock.setAttribute(Qt.WA_TranslucentBackground)
        self.addDockWidget(Qt.RightDockWidgetArea, self.imageSliderDock)
        self.zoomWidget = ZoomWidget()
        self.colorDialog = ColorDialog(parent=self)
        self.zoomWidgetValue = self.zoomWidget.value()
        self.msgBox = QMessageBox()

        # =================== Thumbnail ==================
        hlayout = QHBoxLayout()
        m = (0, 0, 0, 0)
        hlayout.setSpacing(0)
        hlayout.setContentsMargins(*m)
        self.preButton = QToolButton()
        self.preButton.setIcon(newIcon("prev", 40))
        self.preButton.setIconSize(QSize(40, 100))
        self.preButton.clicked.connect(self.openPrevImg)
        self.preButton.setStyleSheet('border: none;')
        self.preButton.setShortcut('a')
        self.iconlist = QListWidget()
        self.iconlist.setViewMode(QListView.IconMode)
        self.iconlist.setFlow(QListView.TopToBottom)
        self.iconlist.setSpacing(10)
        self.iconlist.setIconSize(QSize(50, 50))
        self.iconlist.setMovement(QListView.Static)
        self.iconlist.setResizeMode(QListView.Adjust)
        self.iconlist.itemClicked.connect(self.iconitemDoubleClicked)
        self.iconlist.setStyleSheet("QListWidget{ background-color:transparent; border: none;}")
        self.iconlist.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.nextButton = QToolButton()
        self.nextButton.setIcon(newIcon("next", 40))
        self.nextButton.setIconSize(QSize(40, 100))
        self.nextButton.setStyleSheet('border: none;')
        self.nextButton.clicked.connect(self.openNextImg)
        self.nextButton.setShortcut('d')

        hlayout.addWidget(self.preButton)
        hlayout.addWidget(self.iconlist)
        hlayout.addWidget(self.nextButton)
        self.iconlist.setFixedHeight(100) 
        hlayout.setAlignment(self.iconlist, Qt.AlignCenter)
        tab_layout = QVBoxLayout()
        tab_layout.addLayout(hlayout)
        iconListContainer = QWidget()
        iconListContainer.setLayout(tab_layout)

        # Create an instance of the ConsoleTab class
        console_tab = ConsoleTab(self)
        tabs = QTabWidget()
        # Add the Pages tab to the tabs widget
        tabs.addTab(iconListContainer, "Pages")
        # Add the console tab to the tabs widget
        tabs.addTab(console_tab.console_widget, "Python Console")
        tabs.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        
        # =================== Canvas ==================
        self.canvas = Canvas(parent=self)
        self.canvas.zoomRequest.connect(self.zoomRequest)
        self.canvas.setDrawingShapeToSquare(settings.get(SETTING_DRAW_SQUARE, False))
        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.scrollArea = scroll
        self.canvas.scrollRequest.connect(self.scrollRequest)
        self.canvas.newShape.connect(partial(self.newShape, False))
        self.canvas.shapeMoved.connect(self.updateBoxlist)  # self.setDirty
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)
        centerLayout = QVBoxLayout()
        centerLayout.setContentsMargins(0, 0, 0, 0)
        centerLayout.addWidget(scroll)
        # tabs.setFixedWidth(self.canvas.size().width())
        tabs.setFixedWidth(scroll.size().width())
        # tabs.setFixedWidth(self.scrollArea.width())
        tabs.setFixedHeight(200)
        centerLayout.addWidget(tabs,0,Qt.AlignCenter)
        # centerLayout.addWidget(iconListContainer, 0, Qt.AlignCenter)
        centerContainer = QWidget()
        centerContainer.setLayout(centerLayout)
        self.setCentralWidget(centerContainer)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable)
        self.fileDock.setFeatures(QDockWidget.NoDockWidgetFeatures)

        # =================== Actions ==================
        action = partial(newAction, self)
        quit = action(getStr('quit'), 
                      self.close,
                      'Ctrl+Q', 'quit', 
                      getStr('quitApp'))
        opendir = action(getStr('openDir'), 
                         self.openDirDialog,
                         'Ctrl+u', 'open', 
                         getStr('openDir'))
        import_dataset = action(getStr('importDatasetDir'),
                                self.importDatasetDirDialog,
                                'Ctrl+i', 'import',
                                getStr('importDatasetDir'))
        save = action(getStr('save'), 
                      self.saveFile,
                      'Ctrl+V', 'verify', 
                      getStr('saveDetail'), 
                      enabled=False)
        alcm = action(getStr('choosemodel'), 
                      self.autolcm,
                      'Ctrl+M', 'next', 
                      getStr('tipchoosemodel'))
        deleteImg = action(getStr('deleteImg'), 
                           self.deleteImg, 
                           'Ctrl+Shift+D', 'close', 
                           getStr('deleteImgDetail'),
                           enabled=False)
        resetAll = action(getStr('resetAll'), 
                          self.resetAll, None, 
                          'resetall', 
                          getStr('resetAllDetail'))
        color1 = action(getStr('boxLineColor'), 
                        self.chooseColor,
                        'Ctrl+L', 'color_line', 
                        getStr('boxLineColorDetail'))
        createMode = action(getStr('crtBox'), 
                            self.setCreateMode,
                            'w', 'new', 
                            getStr('crtBoxDetail'),
                            enabled=False)
        editMode = action('&Edit\nRectBox', 
                          self.setEditMode,
                          'Ctrl+J', 'edit', 
                          u'Move and edit Boxes', 
                          enabled=False)
        create = action(getStr('crtBox'), 
                        self.createShape,
                        'w', 'objects', 
                        getStr('crtBoxDetail'),
                        enabled=False)
        delete = action(getStr('delBox'), 
                        self.deleteSelectedShape,
                        'backspace', 'delete', 
                        getStr('delBoxDetail'), 
                        enabled=False)
        copy = action(getStr('dupBox'), 
                      self.copySelectedShape,
                      'Ctrl+C', 'copy', 
                      getStr('dupBoxDetail'),
                      enabled=False)
        hideAll = action(getStr('hideBox'), 
                         partial(self.togglePolygons, False),
                         'Ctrl+H', 'hide', 
                         getStr('hideAllBoxDetail'),
                         enabled=False)
        showAll = action(getStr('showBox'), 
                         partial(self.togglePolygons, True),
                         'Ctrl+A', 'hide', 
                         getStr('showAllBoxDetail'),
                         enabled=False)
        showInfo = action(getStr('info'), 
                          self.showInfoDialog, 
                          None, 'help', 
                          getStr('info'))
        showSteps = action(getStr('steps'), 
                           self.showStepsDialog, 
                           None, 'help', 
                           getStr('steps'))
        showKeys = action(getStr('keys'), 
                          self.showKeysDialog, 
                          None, 'help', 
                          getStr('keys'))

        zoom = QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(u"Zoom in or out of the image. Also accessible with"
                                      " %s and %s from the canvas." % (fmtShortcut("Ctrl+[-+]"),
                                      fmtShortcut("Ctrl+Wheel")))
        self.zoomWidget.setEnabled(False)
        zoomIn = action(getStr('zoomin'), 
                        partial(self.addZoom, 10),
                        'Ctrl++', 'zoom-in',
                        getStr('zoominDetail'), 
                        enabled=False)
        zoomOut = action(getStr('zoomout'), 
                         partial(self.addZoom, -10),
                         'Ctrl+-',
                         'zoom-out',
                         getStr('zoomoutDetail'),
                         enabled=False)
        zoomOrg = action(getStr('originalsize'),
                         partial(self.setZoom, 100),
                         'Ctrl+=', 
                         'zoom', 
                         getStr('originalsizeDetail'), 
                         enabled=False)
        fitWindow = action(getStr('fitWin'), 
                           self.setFitWindow,
                           'Ctrl+F', 
                           'fit-window', 
                           getStr('fitWinDetail'),
                           checkable=True, 
                           enabled=False)
        fitWidth = action(getStr('fitWidth'), 
                          self.setFitWidth,
                          'Ctrl+Shift+F', 
                          'fit-width', 
                          getStr('fitWidthDetail'),
                          checkable=True, 
                          enabled=False)
        
        # Group zoom controls into a list for easier toggling.
        zoomActions = (self.zoomWidget, 
                       zoomIn, 
                       zoomOut,
                       zoomOrg, 
                       fitWindow, 
                       fitWidth)
        self.zoomMode = self.MANUAL_ZOOM
        self.scalers = {self.FIT_WINDOW: self.scaleFitWindow,
                        self.FIT_WIDTH: self.scaleFitWidth,
                        # Set to one to scale to 100% when loading files.
                        self.MANUAL_ZOOM: lambda: 1,}

        # ================== New Actions ==================
        edit = action(getStr('editLabel'), 
                      self.editLabel,
                      'Ctrl+E', 'edit', 
                      getStr('editLabelDetail'), 
                      enabled=False)

        AutoRec = action(getStr('autoRecognition'), 
                         self.autoRecognition,
                         '', 'Auto', 
                         getStr('autoRecognition'), 
                         enabled=False)

        reRec = action(getStr('reRecognition'), 
                       self.reRecognition,
                       'Ctrl+Shift+R', 'reRec', 
                       getStr('reRecognition'), 
                       enabled=False)

        singleRere = action(getStr('singleRe'), 
                            self.singleRerecognition,
                            'Ctrl+R', 'reRec', 
                            getStr('singleRe'), 
                            enabled=False)

        createpoly = action(getStr('creatPolygon'), 
                            self.createPolygon,
                            'q', 'new', 
                            getStr('creatPolygon'), 
                            enabled=False)

        deletegroup = action(getStr('deleteGroup'),
                            self.deleteGroup,
                            'Ctrl+D', 'delete',
                            getStr('deleteGroupDetail'),
                            enabled=False)

        labelgroup = action(getStr('labelGroup'),
                            self.labelGroup,
                            'r', 'labelGroup',
                            getStr('labelGroupDetail'),
                            enabled=False)

        mergegroup = action(getStr('mergeGroup'),
                            self.mergeGroup,
                            'e', 'mergeGroup',
                            getStr('mergeGroupDetail'),
                            enabled=False)

        tableRec = action(getStr('TableRecognition'), 
                          self.TableRecognition,
                          '', 'Auto', 
                          getStr('TableRecognition'), 
                          enabled=False)

        cellreRec = action(getStr('cellreRecognition'), 
                           self.cellreRecognition,
                           '', 'reRec', 
                           getStr('cellreRecognition'), 
                           enabled=False)

        saveRec = action(getStr('saveRec'), 
                         self.saveRecResult,
                         '', 'save', 
                         getStr('saveRec'), 
                         enabled=False)

        saveLabel = action(getStr('saveLabel'), 
                           self.saveLabelFile,  #
                           'Ctrl+S', 'save', 
                           getStr('saveLabel'), 
                           enabled=False)

        exportPubTabNet = action(getStr('exportPubTabNet'), 
                            self.exportPubTabNet,
                            '', 'save', 
                            getStr('exportPubTabNet'), 
                            enabled=False)

        exportPICK = action(getStr('exportPICK'), 
                            self.exportPICK,
                            '', 'save', 
                            getStr('exportPICK'), 
                            enabled=False)
        
        exportFUNSD = action(getStr('exportFUNSD'), 
                            self.exportFUNSD,
                            '', 'save', 
                            getStr('exportFUNSD'), 
                            enabled=False)
        
        exportXFUND = action(getStr('exportXFUND'), 
                            self.exportXFUND,
                            '', 'save', 
                            getStr('exportXFUND'), 
                            enabled=False)

        undoLastPoint = action(getStr("undoLastPoint"), 
                               self.canvas.undoLastPoint,
                               'Ctrl+Z', "undo", 
                               getStr("undoLastPoint"), 
                               enabled=False)

        rotateLeft = action(getStr("rotateLeft"), 
                            partial(self.rotateImgAction, 1),
                            'Ctrl+Alt+L', "rotateLeft", 
                            getStr("rotateLeft"), 
                            enabled=False)

        rotateRight = action(getStr("rotateRight"), 
                             partial(self.rotateImgAction, -1),
                             'Ctrl+Alt+R', "rotateRight", 
                             getStr("rotateRight"), 
                             enabled=False)

        undo = action(getStr("undo"), 
                      self.undoShapeEdit,
                      'Ctrl+Z', "undo", 
                      getStr("undo"), 
                      enabled=False)

        change_cls = action(getStr("labelChange"), 
                            self.change_box_key, 
                            'Ctrl+X', "edit", 
                            getStr("labelChange"), 
                            enabled=False)

        lock = action(getStr("lockBox"), 
                      self.lockSelectedShape,
                      None, "lock", 
                      getStr("lockBoxDetail"), 
                      enabled=False)

        # Button
        self.editButton.setDefaultAction(edit)
        self.newButton.setDefaultAction(create)
        self.labelGroupButton.setDefaultAction(labelgroup)   ## Added by HSB
        self.deleteGroupButton.setDefaultAction(deletegroup) ## Added by HSB
        self.mergeGroupButton.setDefaultAction(mergegroup)   ## Added by HSB
        self.createpolyButton.setDefaultAction(createpoly)
        self.DelButton.setDefaultAction(deleteImg)
        self.SaveButton.setDefaultAction(save)
        self.AutoRecognition.setDefaultAction(AutoRec)
        self.reRecogButton.setDefaultAction(reRec)
        self.tableRecButton.setDefaultAction(tableRec)
        # self.preButton.setDefaultAction(openPrevImg)
        # self.nextButton.setDefaultAction(openNextImg)

        # ================== Zoom layout ==================
        zoomLayout = QHBoxLayout()
        zoomLayout.addStretch()

        self.zoominButton = QToolButton()
        self.zoominButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.zoominButton.setDefaultAction(zoomIn)

        self.zoomoutButton = QToolButton()
        self.zoomoutButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.zoomoutButton.setDefaultAction(zoomOut)

        self.zoomorgButton = QToolButton()
        self.zoomorgButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.zoomorgButton.setDefaultAction(zoomOrg)

        zoomLayout.addWidget(self.zoominButton)
        zoomLayout.addWidget(self.zoomorgButton)
        zoomLayout.addWidget(self.zoomoutButton)

        zoomContainer = QWidget()
        zoomContainer.setLayout(zoomLayout)
        zoomContainer.setGeometry(0, 0, 30, 150)

        shapeLineColor = action(getStr('shapeLineColor'), 
                                self.chshapeLineColor,
                                icon='color_line', 
                                tip=getStr('shapeLineColorDetail'),
                                enabled=False)
        shapeFillColor = action(getStr('shapeFillColor'), 
                                self.chshapeFillColor,
                                icon='color', 
                                tip=getStr('shapeFillColorDetail'),
                                enabled=False)

        # Label list context menu.
        labelMenu = QMenu()
        addActions(labelMenu, (edit, delete))

        self.textList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.textList.customContextMenuRequested.connect(self.popLabelListMenu)

        # Draw squares/rectangles
        self.drawSquaresOption = QAction(getStr('drawSquares'), self)
        self.drawSquaresOption.setCheckable(True)
        self.drawSquaresOption.setChecked(settings.get(SETTING_DRAW_SQUARE, False))
        self.drawSquaresOption.triggered.connect(self.toogleDrawSquare)

        # Store actions for further handling.
        self.actions = struct(save=save, resetAll=resetAll, deleteImg=deleteImg,
                              lineColor=color1, create=create, createpoly=createpoly, tableRec=tableRec, delete=delete, 
                              edit=edit, copy=copy,
                              saveRec=saveRec, singleRere=singleRere, AutoRec=AutoRec, reRec=reRec, cellreRec=cellreRec,
                              createMode=createMode, labelgroup=labelgroup, deletegroup=deletegroup, mergegroup=mergegroup,
                              editMode=editMode, shapeLineColor=shapeLineColor, shapeFillColor=shapeFillColor,
                              zoom=zoom, zoomIn=zoomIn, zoomOut=zoomOut, zoomOrg=zoomOrg,
                              fitWindow=fitWindow, fitWidth=fitWidth,
                              zoomActions=zoomActions, saveLabel=saveLabel, change_cls=change_cls,
                              undo=undo, undoLastPoint=undoLastPoint, import_dataset=import_dataset,
                              rotateLeft=rotateLeft, rotateRight=rotateRight, lock=lock, exportPubTabNet=exportPubTabNet,
                              exportPICK=exportPICK, exportFUNSD=exportFUNSD, exportXFUND=exportXFUND,
                              fileMenuActions=(opendir, import_dataset, saveLabel, exportPubTabNet, exportPICK, exportFUNSD, 
                                               exportXFUND, resetAll, quit),
                              beginner=(), advanced=(),
                              editMenu=(createpoly, edit, copy, delete, singleRere, cellreRec, mergegroup, labelgroup, deletegroup, None, 
                                        undo, undoLastPoint, None, 
                                        rotateLeft, rotateRight, None, color1, self.drawSquaresOption, lock,
                                        None, change_cls),
                              beginnerContext=(create, createpoly, edit, copy, delete, singleRere, cellreRec, rotateLeft, 
                                               rotateRight, lock, change_cls),
                              advancedContext=(createMode, editMode, edit, copy,
                                               delete, shapeLineColor, shapeFillColor),
                              onLoadActive=(create, createpoly, createMode, editMode),
                              onShapesPresent=(hideAll, showAll))

        # Menus
        self.menus = struct(file=self.menu('&' + getStr('mfile')),
                            edit=self.menu('&' + getStr('medit')),
                            view=self.menu('&' + getStr('mview')),
                            autolabel=self.menu('&Inference'),   ## Added by HSB to generalize the items later PyTorch will be added
                            help=self.menu('&' + getStr('mhelp')),
                            recentFiles=QMenu('Open &Recent'),
                            textList=labelMenu)
        self.lastOCR = None

        # Add option to enable/disable labels being displayed at the top of bounding boxes
        self.displayLabelOption = QAction(getStr('displayLabel'), self)
        self.displayLabelOption.setShortcut("Ctrl+Shift+P")
        self.displayLabelOption.setCheckable(True)
        self.displayLabelOption.setChecked(settings.get(SETTING_PAINT_LABEL, False))
        self.displayLabelOption.triggered.connect(self.togglePaintLabelsOption)
        # Add option to enable/disable box index being displayed at the top of bounding boxes
        self.displayIndexOption = QAction(getStr('displayIndex'), self)
        self.displayIndexOption.setCheckable(True)
        self.displayIndexOption.setChecked(settings.get(SETTING_PAINT_INDEX, False))
        self.displayIndexOption.triggered.connect(self.togglePaintIndexOption)
        self.ocrDialogOption = QAction(getStr('ocrDialogOption'), self)
        self.ocrDialogOption.setShortcut("Ctrl+Shift+L")
        self.ocrDialogOption.setCheckable(True)
        self.ocrDialogOption.setChecked(settings.get(SETTING_PAINT_LABEL, False))
        self.displayIndexOption.setChecked(settings.get(SETTING_PAINT_INDEX, False))
        self.ocrDialogOption.triggered.connect(self.speedChoose)
        self.autoSaveOption = QAction(getStr('autoSaveMode'), self)
        self.autoSaveOption.setCheckable(True)
        self.autoSaveOption.setChecked(settings.get(SETTING_PAINT_LABEL, False))
        self.displayIndexOption.setChecked(settings.get(SETTING_PAINT_INDEX, False))
        self.autoSaveOption.triggered.connect(self.autoSaveFunc)
        # Create the side menu
        self.sideMenu = QMenu("Export", self)
        self.sideMenu.setEnabled(False)

        exportPubTabNet = QAction("Export PubTabNet",self)
        exportPubTabNet.triggered.connect(self.exportPubTabNet)
        self.sideMenu.addAction(exportPubTabNet)

        exportPICK= QAction("Export PICK",self)
        exportPICK.triggered.connect(self.exportPICK)
        self.sideMenu.addAction(exportPICK)

        self.sideMenu.addAction("exportFUNSD")
        self.sideMenu.addAction("exportXFUND")

        # Top menu
        addActions(self.menus.file,
                   (opendir, 
                   import_dataset, 
                   None, # None adds horizontal line
                   saveLabel, 
                   saveRec, 
                   self.sideMenu, 
                   self.autoSaveOption, 
                   None, # None adds horizontal line
                   resetAll, 
                   deleteImg,
                   quit))
        addActions(self.menus.help, 
                  (showKeys, 
                   showSteps, 
                   showInfo))
        addActions(self.menus.view, 
                  (self.displayLabelOption, 
                   self.displayIndexOption, 
                   self.ocrDialogOption,
                   None, # None adds horizontal line
                   hideAll, showAll, None, # None adds horizontal line
                   zoomIn, zoomOut, zoomOrg, None, # None adds horizontal line
                   fitWindow, 
                   fitWidth))
        addActions(self.menus.autolabel, 
                  (AutoRec, 
                   reRec, 
                   cellreRec, 
                   alcm, 
                   None, # None adds horizontal line
                   #help))
                  ))
        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        addActions(self.canvas.menus[0], 
                   self.actions.beginnerContext)
        self.statusBar().showMessage('%s started.' % __appname__)
        self.statusBar().show()

        # Application state.
        self.image = QImage()
        self.filePath = ustr(default_filename)
        self.lastOpenDir = None
        self.recentFiles = []
        self.maxRecent = 7
        self.lineColor = None
        self.fillColor = None
        self.zoom_level = 100
        self.fit_window = False
        self.difficult = False

        # Fix the compatible issue for qt4 and qt5. Convert the QStringList to python list
        if settings.get(SETTING_RECENT_FILES):
            if have_qstring():
                recentFileQStringList = settings.get(SETTING_RECENT_FILES)
                self.recentFiles = [ustr(i) for i in recentFileQStringList]
            else:
                self.recentFiles = recentFileQStringList = settings.get(SETTING_RECENT_FILES)

        size = settings.get(SETTING_WIN_SIZE, QSize(1200, 800))

        position = QPoint(0, 0)
        saved_position = settings.get(SETTING_WIN_POSE, position)
        # Fix the multiple monitors issue
        for i in range(QApplication.desktop().screenCount()):
            if QApplication.desktop().availableGeometry(i).contains(saved_position):
                position = saved_position
                break
        self.resize(size)
        self.move(position)
        saveDir = ustr(settings.get(SETTING_SAVE_DIR, None))
        self.lastOpenDir = ustr(settings.get(SETTING_LAST_OPEN_DIR, None))

        self.restoreState(settings.get(SETTING_WIN_STATE, QByteArray()))
        Shape.line_color = self.lineColor = QColor(settings.get(SETTING_LINE_COLOR, DEFAULT_LINE_COLOR))
        Shape.fill_color = self.fillColor = QColor(settings.get(SETTING_FILL_COLOR, DEFAULT_FILL_COLOR))
        self.canvas.setDrawingColor(self.lineColor)

        Shape.difficult = self.difficult

        # ADD:
        # Populate the File menu dynamically.
        self.updateFileMenu()

        # Since loading the file may take some time, make sure it runs in the background.
        if self.filePath and os.path.isdir(self.filePath):
            self.queueEvent(partial(self.importDirImages, self.filePath or ""))
        elif self.filePath:
            self.queueEvent(partial(self.loadFile, self.filePath or ""))

        self.keyDialog = None

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)
        self.populateModeActions()

        # Display cursor coordinates at the right of status bar
        self.labelCoordinates = QLabel('')
        self.statusBar().addPermanentWidget(self.labelCoordinates)

        # Open Dir if default file
        if self.filePath and os.path.isdir(self.filePath):
            self.openDirDialog(dirpath=self.filePath, silent=True)

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.canvas.setDrawingShapeToSquare(False)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            # Draw rectangle if Ctrl is pressed
            self.canvas.setDrawingShapeToSquare(True)

    def noShapes(self):
        return not self.itemsToShapes

    def populateModeActions(self):
        self.canvas.menus[0].clear()
        addActions(self.canvas.menus[0], self.actions.beginnerContext)
        self.menus.edit.clear()
        actions = (self.actions.create,)  # if self.beginner() else (self.actions.createMode, self.actions.editMode)
        addActions(self.menus.edit, actions + self.actions.editMenu)

    def setDirty(self):
        self.dirty = True
        self.actions.save.setEnabled(True)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)

        self.actions.create.setEnabled(True)
        self.actions.createpoly.setEnabled(True)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.itemsToShapes.clear()
        self.shapesToItems.clear()
        self.itemsToShapesbox.clear()  # ADD
        self.shapesToItemsbox.clear()
        self.textList.clear()
        self.BoxList.clear()
        self.indexList.clear()
        self.filePath = None
        self.imageData = None
        self.labelFile = None
        self.canvas.resetState()
        self.labelCoordinates.clear()
        # self.comboBox.cb.clear()
        self.result_dic = []

    def currentItem(self):
        items = self.textList.selectedItems()
        if items:
            return items[0]
        return None

    def currentBox(self):
        items = self.BoxList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filePath):
        if filePath in self.recentFiles:
            self.recentFiles.remove(filePath)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filePath)

    def beginner(self):
        return self._beginner

    def advanced(self):
        return not self.beginner()

    def getAvailableScreencastViewer(self):
        osName = platform.system()
        if osName == 'Windows':
            return ['C:\\Program Files\\Internet Explorer\\iexplore.exe']
        elif osName == 'Linux':
            return ['xdg-open']
        elif osName == 'Darwin':
            return ['open']

    ## Callbacks ##
    def showInfoDialog(self):
        from libs.__init__ import __version__
        msg = u'Name:{0} \nApp Version:{1} \n{2} '.format(__appname__, __version__, sys.version_info)
        QMessageBox.information(self, u'Information', msg)

    def showStepsDialog(self):
        msg = stepsInfo(self.lang)
        QMessageBox.information(self, u'Information', msg)

    def showKeysDialog(self):
        msg = keysInfo(self.lang)
        QMessageBox.information(self, u'Information', msg)

    # Change the label of the existing labels that intersect with the created rectangle
    def labelGroup(self):
        assert self.beginner()
        self.canvas.setEditing(True)
        self.canvas.changeLabel()                      # Changing the label from the Canvas class
        self.deleteSelectedShape(_value=False)         # Deleting the selected shape but changing the label _value=False
        self.actions.labelgroup.setEnabled(True)       # Disabling the labelgroup button

    # Merge the bounding boxes of the selected labels that intersect with the created rectangle
    def mergeGroup(self):
        assert self.beginner()
        print("Merging group.")
        self.canvas.setEditing(True)
        self.mergeSelectedShapes()                      # Deleting the selected shape
        self.actions.mergegroup.setEnabled(True)        # Disabling the labelgroup button

    # Delete existing labels that intersect with the created rectangle
    def deleteGroup(self):
        assert self.beginner()
        print("Delete group.")
        self.canvas.setEditing(True)
        # self.canvas.deleteLabel()                    # Deleting the label from the Canvas class
        self.deleteSelectedShape(_value=True)          # Deleting the selected shape, label, idx interface between label, idx of UI and canvas functions
        self.actions.deletegroup.setEnabled(True)      # Disabling the deletegroup button

    def createShape(self):
        assert self.beginner()
        self.canvas.setEditing(False)
        self.actions.create.setEnabled(False)
        self.actions.createpoly.setEnabled(False)
        self.canvas.fourpoint = False

    def createPolygon(self):
        assert self.beginner()
        self.canvas.setEditing(False)
        self.canvas.fourpoint = True
        self.actions.create.setEnabled(False)
        self.actions.createpoly.setEnabled(False)
        self.actions.undoLastPoint.setEnabled(True)

    def rotateImg(self, filename, k, _value):
        self.actions.rotateRight.setEnabled(_value)
        pix = cv2.imread(filename)
        pix = np.rot90(pix, k)
        cv2.imwrite(filename, pix)
        self.canvas.update()
        self.loadFile(filename)

    def rotateImgWarn(self):
        if self.lang == 'ch':
            self.msgBox.warning(self, "提示", "\n 该图片已经有标注框,旋转操作会打乱标注,建议清除标注框后旋转。")
        else:
            self.msgBox.warning(self, "Warn", "\n The picture already has a label box, "
                                              "and rotation will disrupt the label. "
                                              "It is recommended to clear the label box and rotate it.")

    def rotateImgAction(self, k=1, _value=False):

        filename = self.mImgList[self.currIndex]

        if os.path.exists(filename):
            if self.itemsToShapesbox:
                self.rotateImgWarn()
            else:
                self.saveFile()
                self.dirty = False
                self.rotateImg(filename=filename, k=k, _value=True)
        else:
            self.rotateImgWarn()
            self.actions.rotateRight.setEnabled(False)
            self.actions.rotateLeft.setEnabled(False)

    def toggleDrawingSensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        self.actions.editMode.setEnabled(not drawing)
        if not drawing and self.beginner():
            # Cancel creation.
            print('Cancel creation.')
            self.canvas.setEditing(True)
            self.canvas.restoreCursor()
            self.actions.create.setEnabled(True)
            self.actions.createpoly.setEnabled(True)

    def toggleDrawMode(self, edit=True):
        self.canvas.setEditing(edit)
        self.actions.createMode.setEnabled(edit)
        self.actions.editMode.setEnabled(not edit)

    def setCreateMode(self):
        assert self.advanced()
        self.toggleDrawMode(False)

    def setEditMode(self):
        assert self.advanced()
        self.toggleDrawMode(True)
        self.labelSelectionChanged()

    def updateFileMenu(self):
        currFilePath = self.filePath

        def exists(filename):
            return os.path.exists(filename)

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f !=
                 currFilePath and exists(f)]
        for i, f in enumerate(files):
            icon = newIcon('labels')
            action = QAction(
                icon, '&%d %s' % (i + 1, QFileInfo(f).fileName()), self)
            action.triggered.connect(partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.textList.exec_(self.textList.mapToGlobal(point))

    def editLabel(self):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:
            return
        text = self.textDialog.popUp(item.text())
        if text is not None:
            item.setText(text)
            # item.setBackground(generateColorByText(text))
            self.setDirty()
            self.updateComboBox()

    # =================== detection box related functions ===================
    def boxItemChanged(self, item):
        shape = self.itemsToShapesbox[item]

        box = ast.literal_eval(item.text())
        # print('shape in labelItemChanged is',shape.points)
        if box != [(int(p.x()), int(p.y())) for p in shape.points]:
            # shape.points = box
            shape.points = [QPointF(p[0], p[1]) for p in box]

            # QPointF(x,y)
            # shape.line_color = generateColorByText(shape.label)
            self.setDirty()
        else:  # User probably changed item visibility
            self.canvas.setShapeVisible(shape, True)  # item.checkState() == Qt.Checked

    def editBox(self):  # ADD
        if not self.canvas.editing():
            return
        item = self.currentBox()
        if not item:
            return
        text = self.textDialog.popUp(item.text())

        imageSize = str(self.image.size())
        width, height = self.image.width(), self.image.height()
        if text:
            try:
                text_list = eval(text)
            except:
                msg_box = QMessageBox(QMessageBox.Warning, 'Warning', 'Please enter the correct format')
                msg_box.exec_()
                return
            if len(text_list) < 4:
                msg_box = QMessageBox(QMessageBox.Warning, 'Warning', 'Please enter the coordinates of 4 points')
                msg_box.exec_()
                return
            for box in text_list:
                if box[0] > width or box[0] < 0 or box[1] > height or box[1] < 0:
                    msg_box = QMessageBox(QMessageBox.Warning, 'Warning', 'Out of picture size')
                    msg_box.exec_()
                    return

            item.setText(text)
            # item.setBackground(generateColorByText(text))
            self.setDirty()
            self.updateComboBox()

    def updateBoxlist(self):
        self.canvas.selectedShapes_hShape = []
        if self.canvas.hShape != None:
            self.canvas.selectedShapes_hShape = self.canvas.selectedShapes + [self.canvas.hShape]
        else:
            self.canvas.selectedShapes_hShape = self.canvas.selectedShapes
        for shape in self.canvas.selectedShapes_hShape:
            if shape in self.shapesToItemsbox.keys():
                item = self.shapesToItemsbox[shape]  # listitem
                text = [(int(p.x()), int(p.y())) for p in shape.points]
                item.setText(str(text))
        self.actions.undo.setEnabled(True)
        self.setDirty()

    def indexTo5Files(self, currIndex):
        if currIndex < 2:
            return self.mImgList[:5]
        elif currIndex > len(self.mImgList) - 3:
            return self.mImgList[-5:]
        else:
            return self.mImgList[currIndex - 2: currIndex + 3]

    # Tzutalin 20160906 : Add file list and dock to move faster
    def fileitemDoubleClicked(self, item=None):
        self.currIndex = self.mImgList.index(ustr(os.path.join(os.path.abspath(self.dirname), item.text())))
        filename = self.mImgList[self.currIndex]
        if filename:
            self.mImgList5 = self.indexTo5Files(self.currIndex)
            # self.additems5(None)
            self.loadFile(filename)

    def iconitemDoubleClicked(self, item=None):
        self.currIndex = self.mImgList.index(ustr(os.path.join(item.toolTip())))
        filename = self.mImgList[self.currIndex]
        if filename:
            self.mImgList5 = self.indexTo5Files(self.currIndex)
            # self.additems5(None)
            self.loadFile(filename)

    def CanvasSizeChange(self):
        if len(self.mImgList) > 0 and self.imageSlider.hasFocus():
            self.zoomWidget.setValue(self.imageSlider.value())

    def shapeSelectionChanged(self, selected_shapes):
        self._noSelectionSlot = True
        for shape in self.canvas.selectedShapes:
            shape.selected = False
        self.textList.clearSelection()
        self.indexList.clearSelection()
        self.canvas.selectedShapes = selected_shapes
        for shape in self.canvas.selectedShapes:
            shape.selected = True
            self.shapesToItems[shape].setSelected(True)
            self.shapesToItemsbox[shape].setSelected(True)
            index = self.textList.indexFromItem(self.shapesToItems[shape]).row()
            self.indexList.item(index).setSelected(True)

        self.textList.scrollToItem(self.currentItem())  # QAbstractItemView.EnsureVisible
        # map current label item to index item and select it
        index = self.textList.indexFromItem(self.currentItem()).row()
        self.indexList.scrollToItem(self.indexList.item(index)) 
        self.BoxList.scrollToItem(self.currentBox())

        if self.kie_mode:
            if len(self.canvas.selectedShapes) == 1 and self.keyList.count() > 0:
                selected_key_item_row = self.keyList.findItemsByLabel(self.canvas.selectedShapes[0].key_cls,
                                                                      get_row=True)
                if isinstance(selected_key_item_row, list) and len(selected_key_item_row) == 0:
                    key_text = self.canvas.selectedShapes[0].key_cls
                    item = self.keyList.createItemFromLabel(key_text)
                    self.keyList.addItem(item)
                    rgb = self._get_rgb_by_label(key_text, self.kie_mode)
                    self.keyList.setItemLabel(item, key_text, rgb)
                    selected_key_item_row = self.keyList.findItemsByLabel(self.canvas.selectedShapes[0].key_cls,
                                                                          get_row=True)

                self.keyList.setCurrentRow(selected_key_item_row)

        self._noSelectionSlot = False
        n_selected = len(selected_shapes)
        self.actions.singleRere.setEnabled(n_selected)
        self.actions.cellreRec.setEnabled(n_selected)
        self.actions.delete.setEnabled(n_selected)
        self.actions.copy.setEnabled(n_selected)
        self.actions.edit.setEnabled(n_selected == 1)
        self.actions.lock.setEnabled(n_selected)
        self.actions.change_cls.setEnabled(n_selected)

    def addTextBox(self, shape):
        shape.paintLabel = self.displayLabelOption.isChecked()
        shape.paintIdx = self.displayIndexOption.isChecked()

        # ADD for text
        item = HashableQListWidgetItem(shape.label)
        self.itemsToShapes[item] = shape
        self.shapesToItems[shape] = item
        current_index = QListWidgetItem(str(self.textList.count()))
        current_index.setTextAlignment(Qt.AlignHCenter)
        self.indexList.addItem(current_index)
        self.textList.addItem(item)

        # ADD for box
        item = HashableQListWidgetItem(str([(int(p.x()), int(p.y())) for p in shape.points]))
        self.itemsToShapesbox[item] = shape
        self.shapesToItemsbox[shape] = item
        self.BoxList.addItem(item)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)
        self.updateComboBox()

        # update show counting
        self.BoxListDock.setWindowTitle(self.BoxListDockName + f" ({self.BoxList.count()})")
        self.textListDock.setWindowTitle(self.textListDockName + f" ({self.textList.count()})")

    # Remove items (text, labels, etc) connected to the selected shape that tend to remove
    def remLabels(self, shapes):
        if shapes is None:
            print('Empty shapes.')
            return
        for shape in shapes:
            item = self.shapesToItems[shape]
            self.textList.takeItem(self.textList.row(item))
            del self.shapesToItems[shape]
            del self.itemsToShapes[item]
            self.updateComboBox()

            item = self.shapesToItemsbox[shape]
            self.BoxList.takeItem(self.BoxList.row(item))
            del self.shapesToItemsbox[shape]
            del self.itemsToShapesbox[item]
            self.updateComboBox()

        self.updateIndexList()

    def mergeLabels(self, shapes):
        if shapes is None:
            print('Empty shapes.')
            return

        # Get the items and bounding boxes of the shapes, excluding the last shape.
        items = [self.shapesToItems[shape] for shape in shapes]
        bounding_boxes = [shape.boundingRect() for shape in shapes]
        # Find the minimum and maximum x and y coordinates of the bounding boxes.
        min_x = int(min(box.x() for box in bounding_boxes[1:]))
        min_y = int(min(box.y() for box in bounding_boxes[1:]))
        max_x = int(max(box.x() + box.width() for box in bounding_boxes[1:]))
        max_y = int(max(box.y() + box.height() for box in bounding_boxes[1:]))
        # Create a new item with the merged text and bounding box
        merged_text = " ".join([item.text() for item in items[1:]]) # set changes the order of the items.
        merged_box = [QPointF(min_x, min_y), QPointF(max_x, min_y), QPointF(max_x, max_y), QPointF(min_x, max_y)]

        # Store the first item and its index before deleting
        first_shape = shapes[0]
        first_item = self.shapesToItems[first_shape]
        first_item_index = self.textList.row(first_item)

        # Merge the items and bounding boxes
        for shape in shapes:
            item = self.shapesToItems[shape]
            # Delete the item and bounding box from the list
            item_index = self.textList.row(item)
            self.textList.takeItem(item_index)
            self.BoxList.takeItem(item_index)
            del self.shapesToItems[shape]
            del self.shapesToItemsbox[shape]
            self.updateComboBox()
            self.updateIndexList()

        # Create a new shape to represent the merged shape
        merged_shape = Shape()
        merged_shape.label = merged_text
        merged_shape.points = merged_box
        merged_shape.difficult = first_shape.difficult
        # Add the new shape to the shapelist
        self.canvas.shapes.append(merged_shape)

        merged_shape.paintLabel = self.displayLabelOption.isChecked()
        merged_shape.paintIdx = self.displayIndexOption.isChecked()

        self.addTextBox(merged_shape)
    
    def loadLabels(self, shapes):
        s = []
        shape_index = 0
        for label, points, line_color, key_cls, difficult in shapes:
            shape = Shape(label=label, line_color=line_color, key_cls=key_cls)
            for x, y in points:

                # Ensure the labels are within the bounds of the image. If not, fix them.
                x, y, snapped = self.canvas.snapPointToCanvas(x, y)
                if snapped:
                    self.setDirty()

                shape.addPoint(QPointF(x, y))
            shape.difficult = difficult
            shape.idx = shape_index
            shape_index += 1
            # shape.locked = False
            shape.close()
            s.append(shape)

            self._update_shape_color(shape)
            self.addTextBox(shape)

        self.updateComboBox()
        self.canvas.loadShapes(s)

    def singleLabel(self, shape):
        if shape is None:
            # print('rm empty label')
            return
        item = self.shapesToItems[shape]
        item.setText(shape.label)
        self.updateComboBox()

        # ADD:
        item = self.shapesToItemsbox[shape]
        item.setText(str([(int(p.x()), int(p.y())) for p in shape.points]))
        self.updateComboBox()

    def updateComboBox(self):
        # Get the unique labels and add them to the Combobox.
        itemsTextList = [str(self.textList.item(i).text()) for i in range(self.textList.count())]

        uniqueTextList = list(set(itemsTextList))
        # Add a null row for showing all the labels
        uniqueTextList.append("")
        uniqueTextList.sort()

    # self.comboBox.update_items(uniqueTextList)
    def updateIndexList(self):
        self.indexList.clear()
        for i in range(self.textList.count()):
            string = QListWidgetItem(str(i))
            string.setTextAlignment(Qt.AlignHCenter)
            self.indexList.addItem(string)

    def saveLabels(self, annotationFilePath, mode='Auto'):
        # Mode is Auto means that labels will be loaded from self.result_dic totally, which is the output of ocr model
        annotationFilePath = ustr(annotationFilePath)

        def format_shape(s):
            # print('s in saveLabels is ',s)
            return dict(label=s.label,  # str
                        line_color=s.line_color.getRgb(),
                        fill_color=s.fill_color.getRgb(),
                        points=[(int(p.x()), int(p.y())) for p in s.points],  # QPointF
                        difficult=s.difficult,
                        key_cls=s.key_cls)  # bool

        if mode == 'Auto':
            shapes = []
        else:
            shapes = [format_shape(shape) for shape in self.canvas.shapes if shape.line_color != DEFAULT_LOCK_COLOR]
        # Can add differrent annotation formats here
        for box in self.result_dic:
            trans_dic = {"label": box[1][0], "points": box[0], "difficult": False}
            if self.kie_mode:
                if len(box) == 3:
                    trans_dic.update({"key_cls": box[2]})
                else:
                    trans_dic.update({"key_cls": "None"})
            if trans_dic["label"] == "" and mode == 'Auto':
                continue
            shapes.append(trans_dic)

        try:
            trans_dic = []
            for box in shapes:
                trans_dict = {"transcription": box['label'], "points": box['points'], "difficult": box['difficult']}
                if self.kie_mode:
                    trans_dict.update({"key_cls": box['key_cls']})
                trans_dic.append(trans_dict)
            self.Doclabel[annotationFilePath] = trans_dic
            if mode == 'Auto':
                self.Cachelabel[annotationFilePath] = trans_dic

            # else:
            #     self.labelFile.save(annotationFilePath, shapes, self.filePath, self.imageData,
            #                         self.lineColor.getRgb(), self.fillColor.getRgb())
            # print('Image:{0} -> Annotation:{1}'.format(self.filePath, annotationFilePath))
            return True
        except:
            self.errorMessage(u'Error saving label data', u'Error saving label data')
            return False

    def copySelectedShape(self):
        for shape in self.canvas.copySelectedShape():
            self.addTextBox(shape)
        # fix copy and delete
        # self.shapeSelectionChanged(True)

    def move_scrollbar(self, value):
        self.textListBar.setValue(value)
        self.indexListBar.setValue(value)

    def labelSelectionChanged(self):
        if self._noSelectionSlot:
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.textList.selectedItems():
                selected_shapes.append(self.itemsToShapes[item])
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def indexSelectionChanged(self):
        if self._noSelectionSlot:
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.indexList.selectedItems():
                # map index item to label item
                index = self.indexList.indexFromItem(item).row()
                item = self.textList.item(index)
                selected_shapes.append(self.itemsToShapes[item])
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def boxSelectionChanged(self):
        if self._noSelectionSlot:
            # self.BoxList.scrollToItem(self.currentBox(), QAbstractItemView.PositionAtCenter)
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.BoxList.selectedItems():
                selected_shapes.append(self.itemsToShapesbox[item])
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def labelItemChanged(self, item):
        # avoid accidentally triggering the itemChanged siganl with unhashable item
        # Unknown trigger condition
        if type(item) == HashableQListWidgetItem:
            shape = self.itemsToShapes[item]
            label = item.text()
            if label != shape.label:
                shape.label = item.text()
                # shape.line_color = generateColorByText(shape.label)
                self.setDirty()
            elif not ((item.checkState() == Qt.Unchecked) ^ (not shape.difficult)):
                shape.difficult = True if item.checkState() == Qt.Unchecked else False
                self.setDirty()
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(shape, True)  # item.checkState() == Qt.Checked
                # self.actions.save.setEnabled(True)
        else:
            print('enter labelItemChanged slot with unhashable item: ', item, item.text())
    
    def drag_drop_happened(self):
        '''
        label list drag drop signal slot
        '''
        # print('___________________drag_drop_happened_______________')
        # should only select single item
        for item in self.textList.selectedItems():
            newIndex = self.textList.indexFromItem(item).row()

        # only support drag_drop one item
        assert len(self.canvas.selectedShapes) > 0
        for shape in self.canvas.selectedShapes:
            selectedShapeIndex = shape.idx
        
        if newIndex == selectedShapeIndex:
            return

        # move corresponding item in shape list
        shape = self.canvas.shapes.pop(selectedShapeIndex)
        self.canvas.shapes.insert(newIndex, shape)
            
        # update bbox index
        self.canvas.updateShapeIndex()

        # boxList update simultaneously
        item = self.BoxList.takeItem(selectedShapeIndex)
        self.BoxList.insertItem(newIndex, item)

        # changes happen
        self.setDirty()

    # Callback functions:
    def newShape(self, value=True):
        """
        Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        if len(self.textHist) > 0:
            self.textDialog = OCRDialog(parent=self, listItem=self.textHist)

        if value:
            text = self.textDialog.popUp(text=self.prevText)
            self.lastOCR = text
        else:
            text = self.prevText

        if text is not None:
            self.prevText = self.stringBundle.getString('tempLabel')

            shape = self.canvas.setLastLabel(text, None, None, None)  # generate_color, generate_color
            if self.kie_mode:
                key_text, _ = self.keyDialog.popUp(self.key_previous_text)
                if key_text is not None:
                    shape = self.canvas.setLastLabel(text, None, None, key_text)  # generate_color, generate_color
                    self.key_previous_text = key_text
                    if not self.keyList.findItemsByLabel(key_text):
                        item = self.keyList.createItemFromLabel(key_text)
                        self.keyList.addItem(item)
                        rgb = self._get_rgb_by_label(key_text, self.kie_mode)
                        self.keyList.setItemLabel(item, key_text, rgb)

                    self._update_shape_color(shape)
                    self.keyDialog.addLabelHistory(key_text)

            self.addTextBox(shape)
            if self.beginner():  # Switch to edit mode.
                self.canvas.setEditing(True)
                self.actions.create.setEnabled(True)
                self.actions.createpoly.setEnabled(True)
                self.actions.undoLastPoint.setEnabled(False)
                self.actions.undo.setEnabled(True)
            else:
                self.actions.editMode.setEnabled(True)
            self.setDirty()

        else:
            # self.canvas.undoLastLine()
            self.canvas.resetAllLines()

    def _update_shape_color(self, shape):
        r, g, b = self._get_rgb_by_label(shape.key_cls, self.kie_mode)
        shape.line_color = QColor(r, g, b)
        shape.vertex_fill_color = QColor(r, g, b)
        shape.hvertex_fill_color = QColor(255, 255, 255)
        shape.fill_color = QColor(r, g, b, 128)
        shape.select_line_color = QColor(255, 255, 255)
        shape.select_fill_color = QColor(r, g, b, 155)

    def _get_rgb_by_label(self, label, kie_mode):
        shift_auto_shape_color = 2  # use for random color
        if kie_mode and label != "None":
            item = self.keyList.findItemsByLabel(label)[0]
            label_id = self.keyList.indexFromItem(item).row() + 1
            label_id += shift_auto_shape_color
            return LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]
        else:
            return (0, 255, 0)

    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + increment)
        self.imageSlider.setValue(self.zoomWidget.value() + increment)  # set zoom slider value

    def zoomRequest(self, delta):
        # get the current scrollbar positions
        # calculate the percentages ~ coordinates
        h_bar = self.scrollBars[Qt.Horizontal]
        v_bar = self.scrollBars[Qt.Vertical]

        # get the current maximum, to know the difference after zooming
        h_bar_max = h_bar.maximum()
        v_bar_max = v_bar.maximum()

        # get the cursor position and canvas size
        # calculate the desired movement from 0 to 1
        # where 0 = move left
        #       1 = move right
        # up and down analogous
        cursor = QCursor()
        pos = cursor.pos()
        relative_pos = QWidget.mapFromGlobal(self, pos)

        cursor_x = relative_pos.x()
        cursor_y = relative_pos.y()

        w = self.scrollArea.width()
        h = self.scrollArea.height()

        # the scaling from 0 to 1 has some padding
        # you don't have to hit the very leftmost pixel for a maximum-left movement
        margin = 0.1
        move_x = (cursor_x - margin * w) / (w - 2 * margin * w)
        move_y = (cursor_y - margin * h) / (h - 2 * margin * h)

        # clamp the values from 0 to 1
        move_x = min(max(move_x, 0), 1)
        move_y = min(max(move_y, 0), 1)

        # zoom in
        units = delta / (8 * 15)
        scale = 10
        self.addZoom(scale * units)

        # get the difference in scrollbar values
        # this is how far we can move
        d_h_bar_max = h_bar.maximum() - h_bar_max
        d_v_bar_max = v_bar.maximum() - v_bar_max

        # get the new scrollbar values
        new_h_bar_value = h_bar.value() + move_x * d_h_bar_max
        new_v_bar_value = v_bar.value() + move_y * d_v_bar_max

        h_bar.setValue(new_h_bar_value)
        v_bar.setValue(new_v_bar_value)

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def togglePolygons(self, value):
        for item, shape in self.itemsToShapes.items():
            self.canvas.setShapeVisible(shape, value)

    def loadFile(self, filePath=None):
        """Load the specified file, or the last opened file if None."""
        if self.dirty:
            self.mayContinue()
        self.resetState()
        self.canvas.setEnabled(False)
        if filePath is None:
            filePath = self.settings.get(SETTING_FILENAME)

        # Make sure that filePath is a regular python string, rather than QString
        filePath = ustr(filePath)
        # Fix bug: An index error after select a directory when open a new file.
        unicodeFilePath = ustr(filePath)
        # unicodeFilePath = os.path.abspath(unicodeFilePath)
        # Tzutalin 20160906 : Add file list and dock to move faster
        # Highlight the file item

        if unicodeFilePath and self.fileListWidget.count() > 0:
            if unicodeFilePath in self.mImgList:
                index = self.mImgList.index(unicodeFilePath)
                fileWidgetItem = self.fileListWidget.item(index)
                print('unicodeFilePath is', unicodeFilePath)
                fileWidgetItem.setSelected(True)
                self.iconlist.clear()
                self.additems5(None)

                for i in range(5):
                    item_tooltip = self.iconlist.item(i).toolTip()
                    # print(i,"---",item_tooltip)
                    if item_tooltip == ustr(filePath):
                        titem = self.iconlist.item(i)
                        titem.setSelected(True)
                        self.iconlist.scrollToItem(titem)
                        break
            else:
                self.fileListWidget.clear()
                self.mImgList.clear()
                self.iconlist.clear()

        # if unicodeFilePath and self.iconList.count() > 0:
        #     if unicodeFilePath in self.mImgList:

        if unicodeFilePath and os.path.exists(unicodeFilePath):
            self.canvas.verified = False
            cvimg = cv2.imdecode(np.fromfile(unicodeFilePath, dtype=np.uint8), 1)
            height, width, depth = cvimg.shape
            cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
            image = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)

            if image.isNull():
                self.errorMessage(u'Error opening file',
                                  u"<p>Make sure <i>%s</i> is a valid image file." % unicodeFilePath)
                self.status("Error reading %s" % unicodeFilePath)
                return False
            self.status("Loaded %s" % os.path.basename(unicodeFilePath))
            self.image = image
            self.filePath = unicodeFilePath
            self.canvas.loadPixmap(QPixmap.fromImage(image))

            if self.validFilestate(filePath) is True:
                self.setClean()
            else:
                self.dirty = False
                self.actions.save.setEnabled(True)
            if len(self.canvas.lockedShapes) != 0:
                self.actions.save.setEnabled(True)
                self.setDirty()
            self.canvas.setEnabled(True)
            self.adjustScale(initial=True)
            self.paintCanvas()
            self.addRecentFile(self.filePath)
            self.toggleActions(True)

            self.showBoundingBoxFromDoclabel(filePath)

            self.setWindowTitle(__appname__ + ' ' + filePath)

            # Default : select last item if there is at least one item
            if self.textList.count():
                self.textList.setCurrentItem(self.textList.item(self.textList.count() - 1))
                self.textList.item(self.textList.count() - 1).setSelected(True)
                self.indexList.item(self.textList.count() - 1).setSelected(True)

            # show file list image count
            select_indexes = self.fileListWidget.selectedIndexes()
            if len(select_indexes) > 0:
                self.fileDock.setWindowTitle(self.fileListName + f" ({select_indexes[0].row() + 1}"
                                                                 f"/{self.fileListWidget.count()})")
            # update show counting
            self.BoxListDock.setWindowTitle(self.BoxListDockName + f" ({self.BoxList.count()})")
            self.textListDock.setWindowTitle(self.textListDockName + f" ({self.textList.count()})")

            self.canvas.setFocus(True)
            return True
        return False

    def showBoundingBoxFromDoclabel(self, filePath):
        width, height = self.image.width(), self.image.height()
        imgidx = self.getImglabelidx(filePath)
        shapes = []
        # box['ratio'] of the shapes saved in lockedShapes contains the ratio of the
        # four corner coordinates of the shapes to the height and width of the image
        for box in self.canvas.lockedShapes:
            key_cls = 'None' if not self.kie_mode else box['key_cls']
            if self.canvas.isInTheSameImage:
                shapes.append((box['transcription'], [[s[0] * width, s[1] * height] for s in box['ratio']],
                               DEFAULT_LOCK_COLOR, key_cls, box['difficult']))
            else:
                shapes.append(('锁定框：待检测', [[s[0] * width, s[1] * height] for s in box['ratio']],
                               DEFAULT_LOCK_COLOR, key_cls, box['difficult']))
        if imgidx in self.Doclabel.keys():
            for box in self.Doclabel[imgidx]:
                key_cls = 'None' if not self.kie_mode else box.get('key_cls', 'None')
                shapes.append((box['transcription'], box['points'], None, key_cls, box.get('difficult', False)))

        if shapes != []:
            self.loadLabels(shapes)
            self.canvas.verified = False

    def validFilestate(self, filePath):
        if filePath not in self.fileStatedict.keys():
            return None
        elif self.fileStatedict[filePath] == 1:
            return True
        else:
            return False

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull() \
                and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))
        self.imageSlider.setValue(self.zoomWidget.value())  # set zoom slider value

    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e - 110
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        else:
            settings = self.settings
            # If it loads images from dir, don't load it at the beginning
            if self.dirname is None:
                settings[SETTING_FILENAME] = self.filePath if self.filePath else ''
            else:
                settings[SETTING_FILENAME] = ''

            settings[SETTING_WIN_SIZE] = self.size()
            settings[SETTING_WIN_POSE] = self.pos()
            settings[SETTING_WIN_STATE] = self.saveState()
            settings[SETTING_LINE_COLOR] = self.lineColor
            settings[SETTING_FILL_COLOR] = self.fillColor
            settings[SETTING_RECENT_FILES] = self.recentFiles
            settings[SETTING_ADVANCE_MODE] = not self._beginner
            if self.defaultSaveDir and os.path.exists(self.defaultSaveDir):
                settings[SETTING_SAVE_DIR] = ustr(self.defaultSaveDir)
            else:
                settings[SETTING_SAVE_DIR] = ''

            if self.lastOpenDir and os.path.exists(self.lastOpenDir):
                settings[SETTING_LAST_OPEN_DIR] = self.lastOpenDir
            else:
                settings[SETTING_LAST_OPEN_DIR] = ''

            settings[SETTING_PAINT_LABEL] = self.displayLabelOption.isChecked()
            settings[SETTING_PAINT_INDEX] = self.displayIndexOption.isChecked()
            settings[SETTING_DRAW_SQUARE] = self.drawSquaresOption.isChecked()
            settings.save()
            try:
                self.saveLabelFile()
            except:
                pass

    def loadRecent(self, filename):
        if self.mayContinue():
            print(filename, "======")
            self.loadFile(filename)

    def scanAllImages(self, folderPath):
        extensions = ['.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        images = []

        for file in os.listdir(folderPath):
            if file.lower().endswith(tuple(extensions)):
                relativePath = os.path.join(folderPath, file)
                path = ustr(os.path.abspath(relativePath))
                images.append(path)
        natural_sort(images, key=lambda x: x.lower())
        return images

    # Dialog menu for open
    def openDirDialog(self, _value=False, dirpath=None, silent=False):
        if not self.mayContinue():
            return

        defaultOpenDirPath = dirpath if dirpath else '.'
        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = os.path.dirname(self.filePath) if self.filePath else '.'
        if silent != True:
            targetDirPath = ustr(QFileDialog.getExistingDirectory(self,
                                                                  '%s - Open Directory' % __appname__,
                                                                  defaultOpenDirPath,
                                                                  QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        else:
            targetDirPath = ustr(defaultOpenDirPath)
        
        if targetDirPath:
            self.lastOpenDir = targetDirPath

            # Check if self.comboBox is populated
            if hasattr(self, 'comboBox') and self.comboBox.count() > 0:
                selected_import_type = self.comboBox.currentText()  # Get the selected import data type

                if selected_import_type == 'PICK':
                    self.importDirImagesPICK(targetDirPath)
                elif selected_import_type == 'DocBank':
                    self.importDirImagesDocBank(targetDirPath)
                elif selected_import_type == 'FUNSD':
                    self.importDirImagesFUNSD(targetDirPath)
                elif selected_import_type == 'XFUND':
                    self.importDirImagesXFUND(targetDirPath)
            else:
                # Handle the case when self.comboBox is not populated yet
                self.importDirImages(targetDirPath)
      
    def importDatasetDirDialog(self):
        # Import different datasets into the DocumentLabeler
        # More details are provided in the readMe file and the documentation
        # DocBank, FUNSD, XFUND
        vbox = QVBoxLayout()

        self.panel = QLabel()
        self.panel.setText(self.stringBundle.getString('datasetImport'))
        self.panel.setAlignment(Qt.AlignLeft)

        # Select Type of the Dataset
        self.comboBox = QComboBox()
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItems(['PICK', 'DocBank', 'FUNSD', 'XFUND'])

        vbox.addWidget(self.panel)
        vbox.addWidget(self.comboBox)

        self.dialog = QDialog()
        self.dialog.setWindowTitle(self.stringBundle.getString('datasetImport'))

        hbox = QHBoxLayout()
        hbox.addWidget(self.panel)
        hbox.addWidget(self.comboBox)
        hbox.addStretch(1)

        vbox.addLayout(hbox)

        self.browseBtn = QPushButton(self.stringBundle.getString('browse'))
        self.browseBtn.clicked.connect(self.openDirDialog)

        vbox.addWidget(self.browseBtn)

        self.dialog.setLayout(vbox)
        self.dialog.setWindowModality(Qt.ApplicationModal)
        self.dialog.exec_()

    # This menu contains the import related and the state of the buttons once we import a file     
    def importDirImages(self, dirpath, isDelete=False):
        if not self.mayContinue() or not dirpath:
            return

        if self.defaultSaveDir and self.defaultSaveDir != dirpath:
            self.saveLabelFile()

        if not isDelete:
            self.loadFilestate(dirpath)
            self.Doclabelpath = dirpath + '/Label.txt'
            self.Doclabel = self.loadLabelFile(self.Doclabelpath)
            self.Cachelabelpath = dirpath + '/Cache.cach'
            self.Cachelabel = self.loadLabelFile(self.Cachelabelpath)
            self.keyListPath = dirpath + '/keyList.txt'
            self.keyListItems = self.loadKeyFile(self.keyListPath)
            if self.Cachelabel:
                self.Doclabel = dict(self.Cachelabel, **self.Doclabel)

            self.init_label_list(self.Doclabel)
            self.init_key_list(self.keyListItems)

        self.lastOpenDir = dirpath
        self.dirname = dirpath

        self.defaultSaveDir = dirpath
        self.statusBar().showMessage('%s started. Annotation will be saved to %s' %
                                     (__appname__, self.defaultSaveDir))
        self.statusBar().show()

        self.filePath = None
        self.fileListWidget.clear()
        self.mImgList = self.scanAllImages(dirpath)
        self.mImgList5 = self.mImgList[:5]
        self.openNextImg()
        doneicon = newIcon('done')
        closeicon = newIcon('close')
        for imgPath in self.mImgList:
            filename = os.path.basename(imgPath)
            if self.validFilestate(imgPath) is True:
                item = QListWidgetItem(doneicon, filename)
            else:
                item = QListWidgetItem(closeicon, filename)
            self.fileListWidget.addItem(item)

        print('DirPath in importDirImages is', dirpath)
        self.iconlist.clear()
        self.additems5(dirpath)
        self.changeFileFolder = True
        self.haveAutoReced = False
        self.AutoRecognition.setEnabled(True)
        self.reRecogButton.setEnabled(True)
        self.tableRecButton.setEnabled(True)
        self.labelGroupButton.setEnabled(True)
        self.deleteGroupButton.setEnabled(True)
        self.mergeGroupButton.setEnabled(True)
        self.actions.AutoRec.setEnabled(True)
        self.actions.reRec.setEnabled(True)
        self.actions.labelgroup.setEnabled(True)
        self.actions.deletegroup.setEnabled(True)
        self.actions.mergegroup.setEnabled(True)
        self.actions.tableRec.setEnabled(True)
        self.actions.deleteImg.setEnabled(True)
        self.sideMenu.setEnabled(True)
        self.actions.rotateLeft.setEnabled(True)
        self.actions.rotateRight.setEnabled(True)

        self.fileListWidget.setCurrentRow(0)  # set list index to first
        self.fileDock.setWindowTitle(self.fileListName + f" (1/{self.fileListWidget.count()})")  # show image count

    def convert_string(self,s):
        try:
            return int(s)           # Try to convert the string to an integer
        except ValueError:
            try:
                return float(s)     # Try to convert the string to a float
            except ValueError:
                return s            # If both conversions failed, return the original string
                
    def importDirImagesPICK(self, dirpath):
        # Translate PICK dataset structure to DocumentLabeler format
        print(dirpath)
        dirpath_PICK = dirpath
        dirpath_PICK_doclabeler = dirpath_PICK + '/DocumentLabeler'
        dirpath_PICK_txt = dirpath_PICK + '/boxes_and_transcripts/'
        dirpath_PICK_images =dirpath_PICK + '/images/'

        dirpath_label_txt = dirpath_PICK_doclabeler + '/Label.txt'
        dirpath_fileState_txt = dirpath_PICK_doclabeler + '/fileState.txt'

        if not os.path.exists(dirpath_PICK_doclabeler):
            try:
                os.makedirs(dirpath_PICK_doclabeler)
                print(f"Directory '{dirpath_PICK_doclabeler}' created successfully.")
            except FileExistsError:
                print(f"Directory '{dirpath_PICK_doclabeler}' already exists.")

        # Open boxes_and_transcripts to generate Label.txt
        # Iterate over the files in the target folder
        with open(dirpath_label_txt, 'w') as label_txt:
            for root, dirs, files in os.walk(dirpath_PICK_txt):
                for file in files:
                    if file.endswith('.tsv'):  # Assuming the data is stored in .txt files
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as f:
                            data = f.read()
                            lines = data.split('\n')

                            # Function to parse the boxes and transcripts data
                            parsed_data = []
                            for line in lines:
                                if line.strip() != '':
                                    line = re.sub(r',+', ',', line)  # Remove multiple commas
                                    items = re.split(r'(?<!\\),', line)  # Split using comma while excluding escaped commas
                                    # items = [item.replace("\\,", "\t").split('\t') if "\\," in item else [item] for item in items]
                                    # items = [item for sublist in items for item in sublist]  # Flatten the list
                                                                    # Ensure items has at least 10 elements before the label
                                    # Pad items with empty strings if necessary
                                    required_length = 11  # Adjust based on your data structure; 10 for 9 cells + 1 label
                                    while len(items) < required_length:
                                        print('Length of items:', len(items))
                                        items.append('')

                                    print(items)
                                    try:
                                        line_index = int(items[0])
                                        x1, y1 = int(self.convert_string(items[1])), int(self.convert_string(items[2]))
                                        x2, y2 = int(self.convert_string(items[3])), int(self.convert_string(items[4]))
                                        x3, y3 = int(self.convert_string(items[5])), int(self.convert_string(items[6]))
                                        x4, y4 = int(self.convert_string(items[7])), int(self.convert_string(items[8]))
                                        if "\\" in items[9]:
                                            transcripts = items[9].replace('\\', '')  # Remove all occurrences of \
                                        else:
                                            transcripts = items[9]
                                        label = items[10]
                                        parsed_data.append({
                                            'line_index': line_index,
                                            'box_coordinates': [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                                            'transcripts': transcripts,
                                            'label': label
                                        })
                                    except ValueError:
                                        print('Value error')
                                        parsed_data.append({
                                            'line_index': -1,
                                            'box_coordinates': [[-1, -1], [-1, -1], [-1, -1], [-1, -1]],
                                            'transcripts': items[9].replace('\\', ''),  # Remove all occurrences of \
                                            'label': items[10]
                                        })

                            # Copy the corresponding image file
                            # Check for .jpg or .png files when generating image_file
                            image_file = os.path.splitext(file)[0]
                            if os.path.exists(os.path.join(dirpath_PICK_images, f"{image_file}.jpg")):
                                image_file += ".jpg"
                            elif os.path.exists(os.path.join(dirpath_PICK_images, f"{image_file}.png")):
                                image_file += ".png"
                            else:
                                continue  # Skip the file if neither .jpg nor .png exists

                            image_path = os.path.join(dirpath_PICK_images, image_file)
                            if os.path.exists(image_path):
                                image_file_name = os.path.basename(image_path)
                                new_image_path = os.path.join(dirpath_PICK_doclabeler, image_file_name)
                                shutil.copy(image_path, new_image_path)

                            # Function to generate the Label.txt content
                            label_content = f"DocumentLabeler/{image_file}\t"
                            label_data = []
                            id_counter = 0                              # Initialize id counter
                            for data in parsed_data:
                                label_item = {
                                    'transcription': data['transcripts'],
                                    'difficult': 'false',
                                    'points': [
                                        data['box_coordinates'][0],     # Left-top
                                        data['box_coordinates'][1],     # Right-top
                                        data['box_coordinates'][2],     # Right-bottom
                                        data['box_coordinates'][3]      # Left-bottom
                                    ],
                                    'key_cls': data['label'],
                                    'linking': [],
                                    'id': len(label_data)
                                }
                                id_counter += 1                         # Increment id counter by 1 for each data item
                                label_data.append(label_item)
                            label_content += json.dumps(label_data)
                            label_txt.write(label_content + '\n')

        # Create fileState.txt
        with open(dirpath_fileState_txt, 'w') as fileState_txt:
            file_state_content = []
            image_files = [file for file in os.listdir(dirpath_PICK_doclabeler) if file.lower().endswith(('.jpg', '.png'))]
            for image_file in image_files:
                file_state_content.append(f"/DocumentLabeler/{image_file}\t0")

            fileState_txt.write('\n'.join(file_state_content))

        # Call the OpenDirDialog method with the translated data
        self.importDirImages(dirpath_PICK_doclabeler)
 
    def importDirImagesDocBank(self, dirpath):
        # Translate DocBank data to DocumentLabeler format
        print(dirpath)
        dirpath_DocBank = dirpath
        dirpath_DocBank_txt = dirpath_DocBank + '/txt/'
        dirpath_DocBank_images =dirpath_DocBank + '/imgs/'

        dirpath_DocBank_doclabeler = dirpath_DocBank + '/DocumentLabeler'
        dirpath_label_txt = dirpath_DocBank_doclabeler + '/Label.txt'
        dirpath_fileState_txt = dirpath_DocBank_doclabeler + '/fileState.txt'

        if not os.path.exists(dirpath_DocBank_doclabeler):
            try:
                os.makedirs(dirpath_DocBank_doclabeler)
                print(f"Directory '{dirpath_DocBank_doclabeler}' created successfully.")
            except FileExistsError:
                print(f"Directory '{dirpath_DocBank_doclabeler}' already exists.")

        # Create Label.txt
        label_data = []
        image_files = [file for file in os.listdir(dirpath_DocBank_images) if file.lower().endswith(('.jpg', '.png'))]
        for image_file in image_files:
            image_filename = os.path.join(dirpath_DocBank_images, image_file)
            base_filename = os.path.splitext(image_file)[0]
            if base_filename.endswith('_ori'):
                base_filename = base_filename[:-4]  # Remove '_ori' suffix
            
            with Image.open(image_filename) as img:
                width, height = img.size
                print(f'Image file: {image_filename}, Width: {width}, Height: {height}')
 
            txt_filename = os.path.join(dirpath_DocBank_txt, base_filename + '.txt')

            with open(txt_filename, 'r') as txt_file:
                lines = txt_file.readlines()

            bounding_boxes = []
            for line in lines:
                line_data = line.strip().split('\t')
                x0, y0, x1, y1 = line_data[1:5]
                x0 = int(x0)*width/1000
                y0 = int(y0)*height/1000
                x1 = int(x1)*width/1000
                y1 = int(y1)*height/1000
                bounding_boxes.append([int(x0), int(y0), int(x1), int(y1)])

            transcription_data = []
            for i, line in enumerate(lines):
                line_data = line.strip().split('\t')
                content = line_data[0]
                x0, y0, x1, y1 = bounding_boxes[i]
                label = line_data[-1]  # Retrieve the label from the last item
                transcription_data.append({
                    "transcription": content,
                    "difficult": "False",
                    "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                    "key_cls": label,  # Use the label as "key_cls"
                    "linking": [],
                    "id": i
                })

            label_entry = {
                "image_path": f"DocumentLabeler/{image_file}",
                "transcriptions": transcription_data
            }
            label_data.append(label_entry)

            # Copy image file to destination folder
            dest_filename = os.path.join(dirpath_DocBank_doclabeler, image_file)
            shutil.copy2(image_filename, dest_filename)

        # Write label data to Label.txt
        with open(dirpath_label_txt, 'w') as label_file:
            for label_entry in label_data:
                image_path = label_entry["image_path"]
                transcriptions = label_entry["transcriptions"]

                label_line = f"{image_path}\t{transcriptions}"
                label_file.write(label_line + '\n')

        # Create fileState.txt
        with open(dirpath_fileState_txt, 'w') as fileState_txt:
            file_state_content = []
            image_files = [file for file in os.listdir(dirpath_DocBank_doclabeler) if file.lower().endswith(('.jpg', '.png'))]
            for image_file in image_files:
                file_state_content.append(f"DocumentLabeler/{image_file}\t0")

            fileState_txt.write('\n'.join(file_state_content))

        # Call the openDirDialog method with the translated data
        self.importDirImages(dirpath_DocBank_doclabeler)

    def trans_poly_to_bbox(self, poly): # as used in importDirImagesFUNSD method
        x1 = np.min([p[0] for p in poly])
        x2 = np.max([p[0] for p in poly])
        x3 = np.min([p[1] for p in poly])
        x4 = np.max([p[1] for p in poly])
        return [x1, x2, x3, x4]

    def get_outer_poly(self, bbox_list): # as used in importDirImagesFUNSD method
        x1 = min([bbox[0] for bbox in bbox_list])
        y1 = min([bbox[1] for bbox in bbox_list])
        x2 = max([bbox[2] for bbox in bbox_list])
        y2 = max([bbox[3] for bbox in bbox_list])
        return [[x1,y1], [x2,y1], [x2, y2], [x1, y2]]

    def importDirImagesFUNSD(self, dirpath):
        # Translate FUNSD dataset structure to DocumentLabeler format
        print('importDirImagesFUNSD:' + dirpath)
        dirpath_funsd = dirpath
        dirpath_funsd_doclabeler = dirpath_funsd + '/DocumentLabeler'
        dirpath_img_dir = dirpath + '/images/'         # Add the name of the path suffix to documentation (readMe)
        dirpath_anno_dir = dirpath + '/annotations/'   # Add the name of the path suffix to documentation (readMe)

        dirpath_label_txt = dirpath_funsd_doclabeler + '/Label.txt'
        dirpath_fileState_txt = dirpath_funsd_doclabeler + '/fileState.txt'
        dirpath_keyList_txt = dirpath_funsd_doclabeler + '/keyList.txt'

        if not os.path.exists(dirpath_funsd_doclabeler):
            try:
                os.makedirs(dirpath_funsd_doclabeler)
                print(f"Directory '{dirpath_funsd_doclabeler}' created successfully.")
            except FileExistsError:
                print(f"Directory '{dirpath_funsd_doclabeler}' already exists.")

        imgs = os.listdir(dirpath_img_dir)
        annos = os.listdir(dirpath_anno_dir)

        imgs = [img.replace(".png", "") for img in imgs]
        annos = [anno.replace(".json", "") for anno in annos]

        fn_info_map = dict()
        key_list = set()

        for anno_fn in annos:
            print('Annotation files:' + anno_fn)
            res = []
            with open(os.path.join(dirpath_anno_dir, anno_fn + ".json"), "r") as fin:
                infos = json.load(fin)
                infos = infos["form"]
                old_id2new_id_map = dict()
                global_new_id = 0
                for info in infos:
                    if info["text"] is None:
                        continue
                    words = info["words"]
                    if len(words) <= 0:
                        continue
                    word_idx = 1
                    curr_bboxes = [words[0]["box"]]
                    curr_texts = [words[0]["text"]]
                    while word_idx < len(words):
                        # switch to a new link
                        if words[word_idx]["box"][0] + 10 <= words[word_idx - 1]["box"][2]:
                            if len("".join(curr_texts[0])) > 0:
                                res.append({
                                    "transcription": " ".join(curr_texts),
                                    "difficult": "false",
                                    "points": self.get_outer_poly(curr_bboxes),
                                    "key_cls": info["label"],
                                    "linking": info["linking"],
                                    "id": global_new_id,
                                })
                                if info["id"] not in old_id2new_id_map:
                                    old_id2new_id_map[info["id"]] = []
                                old_id2new_id_map[info["id"]].append(global_new_id)
                                global_new_id += 1
                            curr_bboxes = [words[word_idx]["box"]]
                            curr_texts = [words[word_idx]["text"]]
                        else:
                            curr_bboxes.append(words[word_idx]["box"])
                            curr_texts.append(words[word_idx]["text"])
                        word_idx += 1
                    if len("".join(curr_texts[0])) > 0:
                        res.append({
                            "transcription": " ".join(curr_texts),
                            "difficult": "false",
                            "points": self.get_outer_poly(curr_bboxes),
                            "key_cls": info["label"],
                            "linking": info["linking"],
                            "id": global_new_id,
                        })
                        if info["id"] not in old_id2new_id_map:
                            old_id2new_id_map[info["id"]] = []
                        old_id2new_id_map[info["id"]].append(global_new_id)
                        global_new_id += 1
                res = sorted(res, key=lambda r: (r["points"][0][1], r["points"][0][0]))
                for i in range(len(res) - 1):
                    for j in range(i, 0, -1):
                        if abs(res[j + 1]["points"][0][1] - res[j]["points"][0][1]) < 20 and \
                                (res[j + 1]["points"][0][0] < res[j]["points"][0][0]):
                            tmp = deepcopy(res[j])
                            res[j] = deepcopy(res[j + 1])
                            res[j + 1] = deepcopy(tmp)
                        else:
                            break

                # re-generate unique ids
                for idx, r in enumerate(res):
                    new_links = []
                    for link in r["linking"]:
                        # illegal links will be removed
                        if link[0] not in old_id2new_id_map or link[1] not in old_id2new_id_map:
                            continue
                        for src in old_id2new_id_map[link[0]]:
                            for dst in old_id2new_id_map[link[1]]:
                                new_links.append([src, dst])
                    res[idx]["linking"] = deepcopy(new_links)

                fn_info_map[anno_fn] = res

                # Collect unique labels
                for item in res:
                    key_list.add(item["key_cls"])

        # Write Label.txt
        with open(dirpath_label_txt, "w") as fout:
            for fn in fn_info_map:
                fout.write('DocumentLabeler/' + fn + ".png" + "\t" + json.dumps(fn_info_map[fn], ensure_ascii=False) + "\n")

        # Write fileState.txt
        with open(dirpath_fileState_txt, "w") as fout:
            for img_file in imgs:
                file_state = 'DocumentLabeler/' + img_file + ".png" + '\t' + "0\n"
                fout.write(file_state)

        # Write keyList.txt
        with open(dirpath_keyList_txt, "w") as fout:
            for label in key_list:
                fout.write(label + "\n")

        # Copy image files
        image_files = [img for img in imgs if img in annos]
        for image_file in image_files:
            src_path = os.path.join(dirpath_img_dir, image_file + ".png")
            dest_path = os.path.join(dirpath_funsd_doclabeler, image_file + ".png")
            shutil.copy2(src_path, dest_path)

        # Call the openDirDialog method with the translated data
        self.importDirImages(dirpath_funsd_doclabeler)

    def importDirImagesXFUND(self,dirpath):
        # Translate XFUND dataset structure to DocumentLabeler format
        print(dirpath)
        dirpath_xfund = dirpath
        dirpath_xfund_doclabeler = dirpath_xfund + '/DocumentLabeler'

        dirpath_label_txt = dirpath_xfund_doclabeler + '/Label.txt'
        dirpath_fileState_txt = dirpath_xfund_doclabeler + '/fileState.txt'

        if not os.path.exists(dirpath_xfund_doclabeler):
            try:
                os.makedirs(dirpath_xfund_doclabeler)
                print(f"Directory '{dirpath_xfund_doclabeler}' created successfully.")
            except FileExistsError:
                print(f"Directory '{dirpath_xfund_doclabeler}' already exists.")

        # Create fileState.txt
        with open(dirpath_fileState_txt, 'w') as fileState_txt:
            file_state_content = []
            image_files = [file for file in os.listdir(dirpath_xfund_doclabeler) if file.lower().endswith(('.jpg', '.png'))]
            for image_file in image_files:
                file_state_content.append(f"/DocumentLabeler/{image_file}\t0")

            fileState_txt.write('\n'.join(file_state_content))

        # Call the openDirDialog method with the translated data
        self.importDirImages(dirpath_xfund_doclabeler)

    def exportPICK(self):
        '''
            export the document format for PICK 
            Code for the paper "PICK: Processing Key Information Extraction from 
            Documents using Improved Graph Learning-Convolutional Networks" (ICPR 2020)
            https://arxiv.org/abs/2004.07464
        '''
        # Get export dir
        export_dir = QFileDialog.getExistingDirectory(self)

        # Load labels using the corrected method
        label_file = os.path.join(self.lastOpenDir, "Label.txt")
        label_dict = self.loadLabelFile(label_file)

        # Progress dialog
        progress = ExportProgress()
        progress_dialog = QDialog(self)
        layout = QVBoxLayout()
        bar = QProgressBar()
        total_operations = 2 * len(label_dict)  # Covering both copying images and translating labels
        bar.setRange(0, total_operations)
        layout.addWidget(bar)
        progress_dialog.setLayout(layout)

        # Connect progress signal
        progress.progress.connect(bar.setValue)

        # Output dirs
        boxes_dir = os.path.join(export_dir, 'boxes_and_transcripts')
        os.makedirs(boxes_dir, exist_ok=True)

        # Ensure the destination directory exists
        directory = os.path.join(export_dir, 'images')
        if not os.path.exists(directory):
            os.makedirs(directory)

        # List to store image names for train_file_name.csv
        image_names = []

        # Copy images
        for img_path in label_dict:
            img_name = os.path.basename(img_path)
            src = os.path.join(self.lastOpenDir, img_name)
            dst = os.path.join(export_dir, 'images', img_name)
            if img_name and img_name != '':
                shutil.copy2(src, dst)
                image_names.append(img_name)
            else:
                print(f"Skipping {img_name} as it is not a valid file.")
            
            # Update progress after copying each image
            progress.update_progress()

        # Set to store unique labels for label_list.txt
        unique_labels = set()

        # Translate labels
        for i, (img_path, labels) in enumerate(label_dict.items()):
            if labels is None:
                print(f"Warning: No labels found for image path {img_path}. Skipping...")
                continue

            name = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(boxes_dir, f'{name}.tsv')

            with open(out_path, 'w') as f:
                for j, label in enumerate(labels):
                    x1, y1 = label['points'][0]
                    x2, y2 = label['points'][1]
                    x3, y3 = label['points'][2]
                    x4, y4 = label['points'][3]
                    token = label['transcription']
                    cat = label['key_cls']

                    # Add the category to the set
                    unique_labels.add(cat)

                    # Write TSV row format is depending on the PICK
                    row = f'{j},{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{token},{cat}\n'
                    f.write(row)

            # Update progress after translating labels for each image
            progress.update_progress()

        # Write train_file_name.csv
        with open(os.path.join(export_dir, 'train_file_name.csv'), 'w') as f:
            for idx, img_name in enumerate(image_names):
                f.write(f"{idx},document,{img_name}\n")

        # Write label_list.txt
        with open(os.path.join(export_dir, 'label_list.txt'), 'w') as f:
            for label in unique_labels:
                f.write(f"{label}\n")

        # Finish
        progress_dialog.exec_()
        QMessageBox.information(self, 'Done', 'Export complete!')

    def exportXFUND(self):
        '''
            export the output format for XFUND
            Xu, Yiheng, et al. "Layoutxlm: Multimodal pre-training for multilingual
            visually-rich document understanding." arXiv preprint arXiv:2104.08836 (2021).        
        '''
        pass

        msg = 'The output XFUND is sucessfully saved in {}'.format(self.lastOpenDir)
        QMessageBox.information(self, "Information", msg)
    
    def exportFUNSD(self):
        '''
            export the document format for FUNSD
            Jaume, Guillaume, Hazim Kemal Ekenel, and Jean-Philippe Thiran. "Funsd: 
            A dataset for form understanding in noisy scanned documents." 2019 International 
            Conference on Document Analysis and Recognition Workshops (ICDARW). Vol. 2. IEEE, 2019.
        '''
        pass

        msg = 'The output FUNSD is sucessfully saved in {}'.format(self.lastOpenDir)
        QMessageBox.information(self, "Information", msg)

    def exportPubTabNet(self):
        '''
            export Doclabel and CSV to JSON (PubTabNet)
        '''
        import pandas as pd

        # automatically save annotations
        self.saveFilestate()
        self.saveDoclabel(mode='auto')

        # load box annotations
        labeldict = {}
        if not os.path.exists(self.Doclabelpath):
            msg = 'ERROR, Can not find Label.txt'
            QMessageBox.information(self, "Information", msg)
            return
        else:
            with open(self.Doclabelpath, 'r', encoding='utf-8') as f:
                data = f.readlines()
                for each in data:
                    file, label = each.split('\t')
                    if label:
                        label = label.replace('false', 'False')
                        label = label.replace('true', 'True')
                        labeldict[file] = eval(label)
                    else:
                        labeldict[file] = []
        
        # read table recognition output
        TableRec_excel_dir = os.path.join(self.lastOpenDir, 'tableRec_excel_output')

        # save txt
        fid = open("{}/gt.txt".format(self.lastOpenDir), "w", encoding='utf-8')
        for image_path in labeldict.keys():
            # load csv annotations
            filename, _ = os.path.splitext(os.path.basename(image_path))
            csv_path = os.path.join(
                TableRec_excel_dir, filename + '.xlsx')
            if not os.path.exists(csv_path):
                continue

            excel = xlrd.open_workbook(csv_path)
            sheet0 = excel.sheet_by_index(0)  # only sheet 0
            merged_cells = sheet0.merged_cells # (0,1,1,3) start row, end row, start col, end col

            html_list = [['td'] * sheet0.ncols for i in range(sheet0.nrows)]

            for merged in merged_cells:
                html_list = expand_list(merged, html_list)

            token_list = convert_token(html_list)

            # load box annotations
            cells = []
            for anno in labeldict[image_path]:
                tokens = list(anno['transcription'])
                cells.append({
                    'tokens': tokens, 
                    'bbox': anno['points']
                    })

            # 构造标注信息
            html = {
                'structure': {
                    'tokens': token_list
                    }, 
                'cells': cells
                }
            d = {
                'filename': os.path.basename(image_path), 
                'html': html
                }
            # 重构HTML
            d['gt'] = rebuild_html_from_ppstructure_label(d)
            fid.write('{}\n'.format(
                json.dumps(
                    d, ensure_ascii=False)))
                    
        # convert to PP-Structure label format
        fid.close()
        msg = 'JSON sucessfully saved in {}/gt.txt'.format(self.lastOpenDir)
        QMessageBox.information(self, "Information", msg)

    def init_label_list(self, label_dict):
        if not self.kie_mode:
            return
        # load key_cls
        for image, info in label_dict.items():
            for box in info:
                # print("Current box state:", box)
                # Check if 'key_cls' is missing or empty (including strings that are only whitespace)
                if not box.get("key_cls") or box["key_cls"].strip() == "":
                    # print("Current box empty")
                    box["key_cls"] = "None"  # Update to "None" if missing or empty
                # print("Current box state:", box)
                # print("Current box key:", box["key_cls"])
                
                # Now that we've ensured 'key_cls' is neither missing nor empty, add it to the set
                self.existed_key_cls_set.add(box["key_cls"])

        # Update menu of key list
        if len(self.existed_key_cls_set) > 0:
            for key_text in self.existed_key_cls_set:
                print('key_text in init_label_list is ', key_text)
                if not self.keyList.findItemsByLabel(key_text):
                    item = self.keyList.createItemFromLabel(key_text)
                    self.keyList.addItem(item)
                    rgb = self._get_rgb_by_label(key_text, self.kie_mode)
                    self.keyList.setItemLabel(item, key_text, rgb)

        # if self.keyDialog is None:
        #     # key list dialog
        #     self.keyDialog = KeyDialog(
        #         text=self.key_dialog_tip,
        #         parent=self,
        #         labels=self.existed_key_cls_set,
        #         sort_labels=True,
        #         show_text_field=True,
        #         completion="startswith",
        #         fit_to_content={'column': True, 'row': False},
        #         flags=None
        #     )

    def init_key_list(self, keyListItems):
        if not self.kie_mode:
            return
        for list in keyListItems:
            self.existed_key_cls_set.add(list)
        
        # Update menu of key list
        if len(self.existed_key_cls_set) > 0:
            for key_text in self.existed_key_cls_set:
                if not self.keyList.findItemsByLabel(key_text):
                    item = self.keyList.createItemFromLabel(key_text)
                    self.keyList.addItem(item)
                    rgb = self._get_rgb_by_label(key_text, self.kie_mode)
                    self.keyList.setItemLabel(item, key_text, rgb)
        
        if self.keyDialog is None:
            # key list dialog
            self.keyDialog = KeyDialog(
                text=self.key_dialog_tip,
                parent=self,
                labels=self.existed_key_cls_set,
                sort_labels=True,
                show_text_field=True,
                completion="startswith",
                fit_to_content={'column': True, 'row': False},
                flags=None
            )

    def openPrevImg(self, _value=False):
        if len(self.mImgList) <= 0:
            return

        if self.filePath is None:
            return

        currIndex = self.mImgList.index(self.filePath)
        self.mImgList5 = self.mImgList[:5]
        if currIndex - 1 >= 0:
            filename = self.mImgList[currIndex - 1]
            self.mImgList5 = self.indexTo5Files(currIndex - 1)
            if filename:
                self.loadFile(filename)

    def openNextImg(self, _value=False):
        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        filename = None
        if self.filePath is None:
            filename = self.mImgList[0]
            self.mImgList5 = self.mImgList[:5]
        else:
            currIndex = self.mImgList.index(self.filePath)
            if currIndex + 1 < len(self.mImgList):
                filename = self.mImgList[currIndex + 1]
                self.mImgList5 = self.indexTo5Files(currIndex + 1)
            else:
                self.mImgList5 = self.indexTo5Files(currIndex)
        if filename:
            print('file name in openNext is ', filename)
            self.loadFile(filename)

    def updateFileListIcon(self, filename):
        pass

    def saveFile(self, _value=False, mode='Manual'):
        # Manual mode is used for users click "Save" manually,which will change the state of the image
        if self.filePath:
            imgidx = self.getImglabelidx(self.filePath)
            self._saveFile(imgidx, mode=mode)

    def saveLockedShapes(self):
        self.canvas.lockedShapes = []
        self.canvas.selectedShapes = []
        for s in self.canvas.shapes:
            if s.line_color == DEFAULT_LOCK_COLOR:
                self.canvas.selectedShapes.append(s)
        self.lockSelectedShape()
        for s in self.canvas.shapes:
            if s.line_color == DEFAULT_LOCK_COLOR:
                self.canvas.selectedShapes.remove(s)
                self.canvas.shapes.remove(s)

    def _saveFile(self, annotationFilePath, mode='Manual'):
        if len(self.canvas.lockedShapes) != 0:
            self.saveLockedShapes()

        if mode == 'Manual':
            self.result_dic_locked = []
            img = cv2.imread(self.filePath)
            width, height = self.image.width(), self.image.height()
            for shape in self.canvas.lockedShapes:
                box = [[int(p[0] * width), int(p[1] * height)] for p in shape['ratio']]
                # assert len(box) == 4
                result = [(shape['transcription'], 1)]
                result.insert(0, box)
                self.result_dic_locked.append(result)
            self.result_dic += self.result_dic_locked
            self.result_dic_locked = []
            if annotationFilePath and self.saveLabels(annotationFilePath, mode=mode):
                self.setClean()
                self.statusBar().showMessage('Saved to  %s' % annotationFilePath)
                self.statusBar().show()
                currIndex = self.mImgList.index(self.filePath)
                item = self.fileListWidget.item(currIndex)
                item.setIcon(newIcon('done'))

                self.fileStatedict[self.filePath] = 1
                if len(self.fileStatedict) % self.autoSaveNum == 0:
                    self.saveFilestate()
                    self.saveDoclabel(mode='Auto')

                self.fileListWidget.insertItem(int(currIndex), item)
                if not self.canvas.isInTheSameImage:
                    self.openNextImg()
                self.actions.saveRec.setEnabled(True)
                self.actions.saveLabel.setEnabled(True)
                self.actions.exportPubTabNet.setEnabled(True)
                self.actions.exportPICK.setEnabled(True)
                self.actions.exportXFUND.setEnabled(True)
                self.actions.exportFUNSD.setEnabled(True)

        elif mode == 'Auto':
            if annotationFilePath and self.saveLabels(annotationFilePath, mode=mode):
                self.setClean()
                self.statusBar().showMessage('Saved to  %s' % annotationFilePath)
                self.statusBar().show()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def deleteImg(self):
        deletePath = self.filePath
        if deletePath is not None:
            deleteInfo = self.deleteImgDialog()
            if deleteInfo == QMessageBox.Yes:
                if platform.system() == 'Windows':
                    from win32com.shell import shell, shellcon
                    shell.SHFileOperation((0, shellcon.FO_DELETE, deletePath, None,
                                           shellcon.FOF_SILENT | shellcon.FOF_ALLOWUNDO | shellcon.FOF_NOCONFIRMATION,
                                           None, None))
                    # linux
                elif platform.system() == 'Linux':
                    cmd = 'trash ' + deletePath
                    os.system(cmd)
                    # macOS
                elif platform.system() == 'Darwin':
                    import subprocess
                    absPath = os.path.abspath(deletePath).replace('\\', '\\\\').replace('"', '\\"')
                    cmd = ['osascript', '-e',
                           'tell app "Finder" to move {the POSIX file "' + absPath + '"} to trash']
                    print(cmd)
                    subprocess.call(cmd, stdout=open(os.devnull, 'w'))

                if self.filePath in self.fileStatedict.keys():
                    self.fileStatedict.pop(self.filePath)
                imgidx = self.getImglabelidx(self.filePath)
                if imgidx in self.Doclabel.keys():
                    self.Doclabel.pop(imgidx)
                self.openNextImg()
                self.importDirImages(self.lastOpenDir, isDelete=True)

    def deleteImgDialog(self):
        yes, cancel = QMessageBox.Yes, QMessageBox.Cancel
        msg = u'The image will be deleted to the recycle bin'
        return QMessageBox.warning(self, u'Attention', msg, yes | cancel)

    def resetAll(self):
        self.settings.reset()
        self.close()
        proc = QProcess()
        proc.startDetached(os.path.abspath(__file__))

    def mayContinue(self):  #
        if not self.dirty:
            return True
        else:
            discardChanges = self.discardChangesDialog()
            if discardChanges == QMessageBox.No:
                return True
            elif discardChanges == QMessageBox.Yes:
                self.canvas.isInTheSameImage = True
                self.saveFile()
                self.canvas.isInTheSameImage = False
                return True
            else:
                return False

    def discardChangesDialog(self):
        yes, no, cancel = QMessageBox.Yes, QMessageBox.No, QMessageBox.Cancel
        if self.lang == 'ch':
            msg = u'您有未保存的变更, 您想保存再继续吗?\n点击 "No" 丢弃所有未保存的变更.'
        else:
            msg = u'You have unsaved changes, would you like to save them and proceed?\nClick "No" to undo all changes.'
        return QMessageBox.warning(self, u'Attention', msg, yes | no | cancel, defaultButton=yes)

    def errorMessage(self, title, message):
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return os.path.dirname(self.filePath) if self.filePath else '.'

    def chooseColor(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.lineColor = color
            Shape.line_color = color
            self.canvas.setDrawingColor(color)
            self.canvas.update()
            self.setDirty()

    def mergeSelectedShapes(self):
        self.mergeLabels(self.canvas.mergeSelected())
        self.actions.undo.setEnabled(True)
        self.setDirty()
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)
        self.BoxListDock.setWindowTitle(self.BoxListDockName + f" ({self.BoxList.count()})")
        self.textListDock.setWindowTitle(self.textListDockName + f" ({self.textList.count()})")       

    def deleteSelectedShape(self, _value = False):
        self.remLabels(self.canvas.deleteSelected(_value))
        self.actions.undo.setEnabled(True)
        self.setDirty()
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)
        self.BoxListDock.setWindowTitle(self.BoxListDockName + f" ({self.BoxList.count()})")
        self.textListDock.setWindowTitle(self.textListDockName + f" ({self.textList.count()})")

    def chshapeLineColor(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            for shape in self.canvas.selectedShapes: shape.line_color = color
            self.canvas.update()
            self.setDirty()

    def chshapeFillColor(self):
        color = self.colorDialog.getColor(self.fillColor, u'Choose fill color',
                                          default=DEFAULT_FILL_COLOR)
        if color:
            for shape in self.canvas.selectedShapes: shape.fill_color = color
            self.canvas.update()
            self.setDirty()

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.addTextBox(self.canvas.selectedShape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def loadPredefinedClasses(self, predefClassesFile):
        if os.path.exists(predefClassesFile) is True:
            with codecs.open(predefClassesFile, 'r', 'utf8') as f:
                for line in f:
                    line = line.strip()
                    if self.textHist is None:
                        self.textHist = [line]
                    else:
                        self.textHist.append(line)

    def togglePaintLabelsOption(self):
        self.displayIndexOption.setChecked(False)
        for shape in self.canvas.shapes:
            shape.paintLabel = self.displayLabelOption.isChecked()
            shape.paintIdx = self.displayIndexOption.isChecked()
        self.canvas.repaint()

    def togglePaintIndexOption(self):
        self.displayLabelOption.setChecked(False)
        for shape in self.canvas.shapes:
            shape.paintLabel = self.displayLabelOption.isChecked()
            shape.paintIdx = self.displayIndexOption.isChecked()
        self.canvas.repaint()

    def toogleDrawSquare(self):
        self.canvas.setDrawingShapeToSquare(self.drawSquaresOption.isChecked())

    def additems(self, dirpath):
        for file in self.mImgList:
            pix = QPixmap(file)
            _, filename = os.path.split(file)
            filename, _ = os.path.splitext(filename)
            item = QListWidgetItem(QIcon(pix.scaled(100, 100, Qt.IgnoreAspectRatio, Qt.FastTransformation)),
                                   filename[:10])
            item.setToolTip(file)
            self.iconlist.addItem(item)

    def additems5(self, dirpath):
        for file in self.mImgList5:
            pix = QPixmap(file)
            _, filename = os.path.split(file)
            filename, _ = os.path.splitext(filename)
            pfilename = filename[:10]
            if len(pfilename) < 10:
                lentoken = 12 - len(pfilename)
                prelen = lentoken // 2
                bfilename = prelen * " " + pfilename + (lentoken - prelen) * " "
            # item = QListWidgetItem(QIcon(pix.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)),filename[:10])
            item = QListWidgetItem(QIcon(pix.scaled(100, 100, Qt.IgnoreAspectRatio, Qt.FastTransformation)), pfilename)
            # item.setForeground(QBrush(Qt.white))
            item.setToolTip(file)
            self.iconlist.addItem(item)
        owidth = 0
        for index in range(len(self.mImgList5)):
            item = self.iconlist.item(index)
            itemwidget = self.iconlist.visualItemRect(item)
            owidth += itemwidget.width()
        self.iconlist.setMinimumWidth(owidth + 50)

    def gen_quad_from_poly(self, poly):
        """
        Generate min area quad from poly.
        """
        point_num = poly.shape[0]
        min_area_quad = np.zeros((4, 2), dtype=np.float32)
        # (center (x,y), (width, height), angle of rotation)
        rect = cv2.minAreaRect(poly.astype(np.int32))  
        box = np.array(cv2.boxPoints(rect))

        first_point_idx = 0
        min_dist = 1e4
        for i in range(4):
            dist = np.linalg.norm(box[(i + 0) % 4] - poly[0]) + \
                   np.linalg.norm(box[(i + 1) % 4] - poly[point_num // 2 - 1]) + \
                   np.linalg.norm(box[(i + 2) % 4] - poly[point_num // 2]) + \
                   np.linalg.norm(box[(i + 3) % 4] - poly[-1])
            if dist < min_dist:
                min_dist = dist
                first_point_idx = i
        for i in range(4):
            min_area_quad[i] = box[(first_point_idx + i) % 4]

        bbox_new = min_area_quad.tolist()
        bbox = []

        for box in bbox_new:
            box = list(map(int, box))
            bbox.append(box)

        return bbox

    def getImglabelidx(self, filePath):
        if platform.system() == 'Windows':
            spliter = '\\'
        else:
            spliter = '/'
        filepathsplit = filePath.split(spliter)[-2:]
        return filepathsplit[0] + '/' + filepathsplit[1]

    def autoRecognition(self):
        assert self.mImgList is not None
        print('Using model from ', self.model)

        uncheckedList = [i for i in self.mImgList if i not in self.fileStatedict.keys()]
        self.autoDialog = AutoDialog(parent=self, 
                                     ocr=self.ocr, 
                                     mImgList=uncheckedList, 
                                     lenbar=len(uncheckedList))
        self.autoDialog.popUp()
        self.currIndex = len(self.mImgList) - 1
        self.loadFile(self.filePath)  # ADD
        self.haveAutoReced = True
        self.AutoRecognition.setEnabled(False)
        self.actions.AutoRec.setEnabled(False)
        self.setDirty()
        self.saveCacheLabel()

        self.init_key_list(self.Cachelabel)

    def reRecognition(self):
        img = cv2.imdecode(np.fromfile(self.filePath,dtype=np.uint8),1)
        # org_box = [dic['points'] for dic in self.Doclabel[self.getImglabelidx(self.filePath)]]
        if self.canvas.shapes:
            self.result_dic = []
            self.result_dic_locked = []  # result_dic_locked stores the ocr result of self.canvas.lockedShapes
            rec_flag = 0
            for shape in self.canvas.shapes:
                box = [[int(p.x()), int(p.y())] for p in shape.points]
                kie_cls = shape.key_cls

                if len(box) > 4:
                    box = self.gen_quad_from_poly(np.array(box))
                assert len(box) == 4

                img_crop = get_rotate_crop_image(img, np.array(box, np.float32))
                if img_crop is None:
                    msg = 'Cannot recognise the detection box in ' + \
                           self.filePath + \
                           '. Please change manually'
                    QMessageBox.information(self, "Information", msg)
                    return
                result = self.ocr.ocr(img_crop, cls=True, det=False)[0]
                if result[0][0] != '':
                    if shape.line_color == DEFAULT_LOCK_COLOR:
                        shape.label = result[0][0]
                        result.insert(0, box)
                        if self.kie_mode:
                            result.append(kie_cls)
                        self.result_dic_locked.append(result)
                    else:
                        result.insert(0, box)
                        if self.kie_mode:
                            result.append(kie_cls)
                        self.result_dic.append(result)
                else:
                    print('Cannot recognise the box')
                    if shape.line_color == DEFAULT_LOCK_COLOR:
                        shape.label = result[0][0]
                        if self.kie_mode:
                            self.result_dic_locked.append([box, (self.noLabelText, 0), kie_cls])
                        else:
                            self.result_dic_locked.append([box, (self.noLabelText, 0)])
                    else:
                        if self.kie_mode:
                            self.result_dic.append([box, (self.noLabelText, 0), kie_cls])
                        else:
                            self.result_dic.append([box, (self.noLabelText, 0)])
                try:
                    if self.noLabelText == shape.label or result[1][0] == shape.label:
                        print('label no change')
                    else:
                        rec_flag += 1
                except IndexError as e:
                    print('Cannot recognise the box')
            if (len(self.result_dic) > 0 and rec_flag > 0) or self.canvas.lockedShapes:
                self.canvas.isInTheSameImage = True
                self.saveFile(mode='Auto')
                self.loadFile(self.filePath)
                self.canvas.isInTheSameImage = False
                self.setDirty()
            elif len(self.result_dic) == len(self.canvas.shapes) and rec_flag == 0:
                if self.lang == 'ch':
                    QMessageBox.information(self, "Information", "识别结果保持一致！")
                else:
                    QMessageBox.information(self, "Information", "The recognition result remains unchanged!")
            else:
                print('Can not recgonise in ', self.filePath)
        else:
            QMessageBox.information(self, "Information", "Draw a box!")

    def singleRerecognition(self):
        img = cv2.imdecode(np.fromfile(self.filePath,dtype=np.uint8),1)
        for shape in self.canvas.selectedShapes:
            box = [[int(p.x()), int(p.y())] for p in shape.points]
            if len(box) > 4:
                box = self.gen_quad_from_poly(np.array(box))
            assert len(box) == 4
            img_crop = get_rotate_crop_image(img, np.array(box, np.float32))
            if img_crop is None:
                msg = 'Can not recognise the detection box in ' + self.filePath + '. Please change manually'
                QMessageBox.information(self, "Information", msg)
                return
            result = self.ocr.ocr(img_crop, cls=True, det=False)[0]
            if result[0][0] != '':
                result.insert(0, box)
                print('result in reRec is ', result)
                if result[1][0] == shape.label:
                    print('label no change')
                else:
                    shape.label = result[1][0]
            else:
                print('Can not recognise the box')
                if self.noLabelText == shape.label:
                    print('label no change')
                else:
                    shape.label = self.noLabelText
            self.singleLabel(shape)
            self.setDirty()

    def TableRecognition(self):
        '''
            Table Recognition
        '''
        from paddleocr import to_excel

        import time

        start = time.time()
        img = cv2.imread(self.filePath)
        res = self.table_ocr(img, return_ocr_result_in_table=True)

        TableRec_excel_dir = self.lastOpenDir + '/tableRec_excel_output/'
        os.makedirs(TableRec_excel_dir, exist_ok=True)
        filename, _ = os.path.splitext(os.path.basename(self.filePath))

        excel_path = TableRec_excel_dir + '{}.xlsx'.format(filename)
        
        if res is None:
            msg = 'Can not recognise the table in ' + self.filePath + '. Please change manually'
            QMessageBox.information(self, "Information", msg)
            to_excel('', excel_path) # create an empty excel
            return
        
        # save res
        # ONLY SUPPORT ONE TABLE in one image
        hasTable = False
        for region in res:
            if region['type'] == 'table':
                if region['res']['boxes'] is None:
                    msg = 'Can not recognise the detection box in ' + \
                           self.filePath + \
                          '. Please change manually'
                    QMessageBox.information(self, "Information", msg)
                    to_excel('', excel_path) # create an empty excel
                    return
                hasTable = True
                # save table ocr result on PPOCRLabel
                # clear all old annotaions before saving result
                self.itemsToShapes.clear()
                self.shapesToItems.clear()
                self.itemsToShapesbox.clear()  # ADD
                self.shapesToItemsbox.clear()
                self.textList.clear()
                self.indexList.clear()
                self.BoxList.clear()
                self.result_dic = []
                self.result_dic_locked = []

                shapes = []
                result_len = len(region['res']['boxes'])
                order_index = 0
                for i in range(result_len):
                    bbox = np.array(region['res']['boxes'][i])
                    rec_text = region['res']['rec_res'][i][0]

                    rext_bbox = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]

                    # save bbox to shape
                    shape = Shape(label=rec_text, line_color=DEFAULT_LINE_COLOR, key_cls=None)
                    for point in rext_bbox:
                        x, y = point
                        # Ensure the labels are within the bounds of the image. 
                        # If not, fix them.
                        x, y, snapped = self.canvas.snapPointToCanvas(x, y)
                        shape.addPoint(QPointF(x, y))
                    shape.difficult = False
                    shape.idx = order_index
                    order_index += 1
                    # shape.locked = False
                    shape.close()
                    self.addTextBox(shape)
                    shapes.append(shape)
                self.setDirty()
                self.canvas.loadShapes(shapes)
                
                # save HTML result to excel
                try:
                    to_excel(region['res']['html'], excel_path)
                except:
                    print('Can not save excel file, maybe Permission denied (.xlsx is being occupied)')
                break
        
        if not hasTable:
            msg = 'Can not recognise the table in ' + self.filePath + '. Please change manually'
            QMessageBox.information(self, "Information", msg)
            to_excel('', excel_path) # create an empty excel
            return

        # automatically open excel annotation file
        if platform.system() == 'Windows':
            try:
                import win32com.client
            except:
                print("CANNOT OPEN .xlsx. It could be one of the following reasons: " \
                    "Only support Windows | No python win32com")

            try:
                xl = win32com.client.Dispatch("Excel.Application")
                xl.Visible = True
                xl.Workbooks.Open(excel_path)
                # excelEx = "You need to show the excel executable at this point"
                # subprocess.Popen([excelEx, excel_path])

                # os.startfile(excel_path)
            except:
                print("CANNOT OPEN .xlsx. It could be the following reasons: " \
                    ".xlsx is not existed")
        else:
            os.system('open ' + os.path.normpath(excel_path))
                
        print('time cost: ', time.time() - start)

    def cellreRecognition(self):
        '''
            Re-recognise text in a cell
        '''
        img = cv2.imread(self.filePath)
        for shape in self.canvas.selectedShapes:
            box = [[int(p.x()), int(p.y())] for p in shape.points]

            if len(box) > 4:
                box = self.gen_quad_from_poly(np.array(box))
            assert len(box) == 4

            # pad around bbox for better text recognition accuracy
            _box = boxPad(box, img.shape, 6)
            img_crop = get_rotate_crop_image(img, np.array(_box, np.float32))
            if img_crop is None:
                msg = 'Can not recognise the detection box in ' + self.filePath + '. Please change manually'
                QMessageBox.information(self, "Information", msg)
                return

            # merge the text result in the cell
            texts = ''
            probs = 0. # the probability of the cell is avgerage prob of every text box in the cell
            bboxes = self.ocr.ocr(img_crop, det=True, rec=False, cls=False)[0]
            if len(bboxes) > 0:
                bboxes.reverse() # top row text at first
                for _bbox in bboxes:
                    patch = get_rotate_crop_image(img_crop, np.array(_bbox, np.float32))
                    rec_res = self.ocr.ocr(patch, det=False, rec=True, cls=False)[0]
                    text = rec_res[0][0]
                    if text != '':
                        # add space between english word
                        texts += text + ('' if text[0].isalpha() else ' ') 
                        probs += rec_res[0][1]
                probs = probs / len(bboxes)
            result = [(texts.strip(), probs)]

            if result[0][0] != '':
                result.insert(0, box)
                print('result in reRec is ', result)
                if result[1][0] == shape.label:
                    print('label no change')
                else:
                    shape.label = result[1][0]
            else:
                print('Can not recognise the box')
                if self.noLabelText == shape.label:
                    print('label no change')
                else:
                    shape.label = self.noLabelText
            self.singleLabel(shape)
            self.setDirty()

    def autolcm(self):
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()

        self.panel = QLabel()
        self.panel.setText(self.stringBundle.getString('chooseModel'))
        self.panel.setAlignment(Qt.AlignLeft)

        self.comboBox = QComboBox()
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItems(['PyTesseract/OCR-en',
                                'PICK/Token Label',
                                'DeepKE/Token Label',
                                'LiLT/Token Label'])
        
        vbox.addWidget(self.panel)
        vbox.addWidget(self.comboBox)

        self.dialog = QDialog()
        self.dialog.resize(300, 100)
        self.okBtn = QPushButton(self.stringBundle.getString('ok'))
        self.cancelBtn = QPushButton(self.stringBundle.getString('cancel'))

        self.okBtn.clicked.connect(self.modelChoose)
        self.cancelBtn.clicked.connect(self.cancel)
        self.dialog.setWindowTitle(self.stringBundle.getString('chooseModel'))

        hbox.addWidget(self.okBtn)
        hbox.addWidget(self.cancelBtn)

        vbox.addWidget(self.panel)
        vbox.addLayout(hbox)
        self.dialog.setLayout(vbox)
        self.dialog.setWindowModality(Qt.ApplicationModal)
        self.dialog.exec_()

        if self.filePath:
            self.AutoRecognition.setEnabled(True)
            self.actions.AutoRec.setEnabled(True)

    def modelChoose(self):
        
        print(self.comboBox.currentText())
        lg_idx = {'PyTesseract/OCR-en': 'pytesseract_en',
                  'PICK/Token Label': 'pick',
                  'DeepKE/Token Label': 'deepke',
                  'LiLT/Token Label': 'lilt'}    
        # [ToDo] We need to pass the selected inference into the modelRunner
        self.model = lg_idx[self.comboBox.currentText()]

        del self.inference
        # Inference(--ckpt      model path, 
        #           --bt        boxes_and_transcripts path, 
        #           --impt      image_path, 
        #           --bs        batch_size, 
        #           --output_path)
        self.inference = Inference(self.model,
                                   checkpoint_path='/mnt/data_drive/CSU_PhD/research/software/DocumentEngineering/DocumentLabeler_FinalDev/models/pick/saved/PICK_default/models/SROIE_test_0521_015255/checkpoint-epoch100.pth',
                                   bt_path='/mnt/data/Data-Document/GeneralDocument/SROIE_PICK/boxes_and_transcripts',
                                   impt_path='/mnt/data/Data-Document/GeneralDocument/SROIE_PICK/images',
                                   batch_size=2,
                                   output_path='/mnt/data_drive/CSU_PhD/research/software/DocumentEngineering/DocumentLabeler_FinalDev/DocumentLabeler/output',
                                   config_file='/mnt/data_drive/CSU_PhD/research/software/DocumentEngineering/DocumentLabeler_FinalDev/models/configs/pick/pick_config.yaml',
                                   num_workers=2)
        self.inference.perform_inference()
        self.dialog.close()

    def cancel(self):
        self.dialog.close()

    def loadFilestate(self, saveDir):
        self.fileStatepath = saveDir + '/fileState.txt'
        self.fileStatedict = {}
        if not os.path.exists(self.fileStatepath):
            f = open(self.fileStatepath, 'w', encoding='utf-8')
        else:
            with open(self.fileStatepath, 'r', encoding='utf-8') as f:
                states = f.readlines()
                for each in states:
                    file, state = each.split('\t')
                    self.fileStatedict[file] = 1
                self.actions.saveLabel.setEnabled(True)
                self.actions.saveRec.setEnabled(True)
                self.actions.exportPubTabNet.setEnabled(True)
                self.actions.exportPICK.setEnabled(True)
                self.actions.exportXFUND.setEnabled(True)
                self.actions.exportFUNSD.setEnabled(True)

    def saveFilestate(self):
        with open(self.fileStatepath, 'w', encoding='utf-8') as f:
            for key in self.fileStatedict:
                f.write(key + '\t')
                f.write(str(self.fileStatedict[key]) + '\n')

    # def loadLabelFile(self, labelpath):
    #     labeldict = {}
    #     print('Labelpath: '+labelpath)
    #     if not os.path.exists(labelpath):
    #         f = open(labelpath, 'w', encoding='utf-8')

    #     else:
    #         with open(labelpath, 'r', encoding='utf-8') as f:
    #             data = f.readlines()
    #             for each in data:
    #                 file, json_content = each.split('\t')
    #                 if json_content:
    #                     json_content = json_content.replace('false', 'False')
    #                     json_content = json_content.replace('true', 'True')
    #                     labeldict[file] = eval(json_content)
    #                 else:
    #                     labeldict[file] = []
    #     return labeldict
    
    def loadLabelFile(self, labelpath):
        labeldict = {}
        print('Labelpath: ' + labelpath)
        
        if not os.path.exists(labelpath):
            return labeldict  # Return an empty dictionary if the file doesn't exist

        with open(labelpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '\t' not in line:
                    continue  # Skip lines without a tab
                filename, json_content = line.split('\t', 1)  # Split only once
                try:
                    labeldict[filename] = json.loads(json_content)
                except json.JSONDecodeError:
                    labeldict[filename] = []

        return labeldict
    
    def loadKeyFile(self, labelpath):
        keydict = []
        print('Labelpath: '+labelpath)
        if not os.path.exists(labelpath):
            f = open(labelpath, 'w', encoding='utf-8')

        else:
            with open(labelpath, 'r', encoding='utf-8') as f:
                data = f.readlines()
                for each in data:
                    keydict.append(each)
        return keydict
    
    def saveDoclabel(self, mode='Manual'):
        savedfile = [self.getImglabelidx(i) for i in self.fileStatedict.keys()]
        with open(self.Doclabelpath, 'w', encoding='utf-8') as f:
            for key in self.Doclabel:
                if key in savedfile and self.Doclabel[key] != []:
                    f.write(key + '\t')
                    f.write(json.dumps(self.Doclabel[key], ensure_ascii=False) + '\n')

        if mode == 'Manual':
            if self.lang == 'ch':
                msg = '已将检查过的图片标签保存在 ' + self.Doclabelpath + " 文件中"
            else:
                msg = 'Images that have been checked are saved in ' + self.Doclabelpath
            QMessageBox.information(self, "Information", msg)

    def saveCacheLabel(self):
        with open(self.Cachelabelpath, 'w', encoding='utf-8') as f:
            for key in self.Cachelabel:
                f.write(key + '\t')
                f.write(json.dumps(self.Cachelabel[key], ensure_ascii=False) + '\n')

    def saveLabelFile(self):
        self.saveFilestate()
        self.saveDoclabel()

    def saveRecResult(self):
        if {} in [self.Doclabelpath, self.Doclabel, self.fileStatedict]:
            QMessageBox.information(self, "Information", "Check the image first")
            return

        rec_gt_dir = os.path.dirname(self.Doclabelpath) + '/rec_gt.txt'
        crop_img_dir = os.path.dirname(self.Doclabelpath) + '/crop_img/'
        ques_img = []
        if not os.path.exists(crop_img_dir):
            os.mkdir(crop_img_dir)

        with open(rec_gt_dir, 'w', encoding='utf-8') as f:
            for key in self.fileStatedict:
                idx = self.getImglabelidx(key)
                try:
                    img = cv2.imread(key)
                    for i, label in enumerate(self.Doclabel[idx]):
                        if label['difficult']:
                            continue
                        img_crop = get_rotate_crop_image(img, np.array(label['points'], np.float32))
                        img_name = os.path.splitext(os.path.basename(idx))[0] + '_crop_' + str(i) + '.jpg'
                        cv2.imwrite(crop_img_dir + img_name, img_crop)
                        f.write('crop_img/' + img_name + '\t')
                        f.write(label['transcription'] + '\n')
                except Exception as e:
                    ques_img.append(key)
                    print("Can not read image ", e)
        if ques_img:
            QMessageBox.information(self,
                                    "Information",
                                    "The following images can not be saved, please check the image path and labels.\n"
                                    + "".join(str(i) + '\n' for i in ques_img))
        QMessageBox.information(self, "Information", "Cropped images have been saved in " + str(crop_img_dir))

    def speedChoose(self):
        if self.ocrDialogOption.isChecked():
            self.canvas.newShape.disconnect()
            self.canvas.newShape.connect(partial(self.newShape, True))

        else:
            self.canvas.newShape.disconnect()
            self.canvas.newShape.connect(partial(self.newShape, False))

    def autoSaveFunc(self):
        if self.autoSaveOption.isChecked():
            self.autoSaveNum = 1  # Real auto_Save
            try:
                self.saveLabelFile()
            except:
                pass
            print('The program will automatically save once after confirming an image')
        else:
            self.autoSaveNum = 5  # After this number of action the system will save automatically
            print('The program will automatically save once after confirming 5 images (default)')

    def change_box_key(self):
        if not self.kie_mode:
            return
        key_text, _ = self.keyDialog.popUp(self.key_previous_text)
        if key_text is None:
            return
        self.key_previous_text = key_text
        for shape in self.canvas.selectedShapes:
            shape.key_cls = key_text
            if not self.keyList.findItemsByLabel(key_text):
                item = self.keyList.createItemFromLabel(key_text)
                self.keyList.addItem(item)
                rgb = self._get_rgb_by_label(key_text, self.kie_mode)
                self.keyList.setItemLabel(item, key_text, rgb)

            self._update_shape_color(shape)
            self.keyDialog.addLabelHistory(key_text)
            
        # save changed shape
        self.setDirty()

    def undoShapeEdit(self):
        self.canvas.restoreShape()
        self.textList.clear()
        self.indexList.clear()
        self.BoxList.clear()
        self.loadShapes(self.canvas.shapes)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

    def loadShapes(self, shapes, replace=True):
        self._noSelectionSlot = True
        for shape in shapes:
            self.addTextBox(shape)
        self.textList.clearSelection()
        self.indexList.clearSelection()
        self._noSelectionSlot = False
        self.canvas.loadShapes(shapes, replace=replace)
        print("loadShapes")  # 1

    def lockSelectedShape(self):
        """lock the selected shapes.

        Add self.selectedShapes to lock self.canvas.lockedShapes, 
        which holds the ratio of the four coordinates of the locked shapes
        to the width and height of the image
        """
        width, height = self.image.width(), self.image.height()

        def format_shape(s):
            return dict(label=s.label,  # str
                        line_color=s.line_color.getRgb(),
                        fill_color=s.fill_color.getRgb(),
                        ratio=[[int(p.x()) / width, int(p.y()) / height] for p in s.points],  # QPonitF
                        difficult=s.difficult,  # bool
                        key_cls=s.key_cls,  # bool
                        )

        # lock
        if len(self.canvas.lockedShapes) == 0:
            for s in self.canvas.selectedShapes:
                s.line_color = DEFAULT_LOCK_COLOR
                s.locked = True
            shapes = [format_shape(shape) for shape in self.canvas.selectedShapes]
            trans_dic = []
            for box in shapes:
                trans_dict = {"transcription": box['label'], "ratio": box['ratio'], "difficult": box['difficult']}
                if self.kie_mode:
                    trans_dict.update({"key_cls": box["key_cls"]})
                trans_dic.append(trans_dict)
            self.canvas.lockedShapes = trans_dic
            self.actions.save.setEnabled(True)

        # unlock
        else:
            for s in self.canvas.shapes:
                s.line_color = DEFAULT_LINE_COLOR
            self.canvas.lockedShapes = []
            self.result_dic_locked = []
            self.setDirty()
            self.actions.save.setEnabled(True)

    def update_var(self, var_name, new_value):
        setattr(self, var_name, new_value)

def inverted(color):
    return QColor(*[255 - v for v in color.getRgb()])

def read(filename, default=None):
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except:
        return default

def str2bool(v):
    return v.lower() in ("true", "t", "1")

def get_main_app(argv=[]):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_() -- so that we can test the application in one thread
    """
    app = QApplication(argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("app"))
    # Tzutalin 201705+: Accept extra arguments to change predefined class file
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--lang", type=str, default='en', nargs="?")
    arg_parser.add_argument("--gpu", type=str2bool, default=True, nargs="?")
    arg_parser.add_argument("--kie", type=str2bool, default=False, nargs="?")
    arg_parser.add_argument("--predefined_classes_file",
                            default=os.path.join(os.path.dirname(__file__), 
                            "data", "predefined_classes.txt"),
                            nargs="?")
    args = arg_parser.parse_args(argv[1:])

    win = MainWindow(lang=args.lang,
                     gpu=args.gpu,
                     kie_mode=args.kie,
                     default_predefined_class_file=args.predefined_classes_file)
    win.show()
    return app, win

def main():
    """construct main app and run it"""
    app, _win = get_main_app(sys.argv)
    return app.exec_()

if __name__ == '__main__':

    # resource_file = '/home/xlab/CSU_PhD_Research/Research/Software/CatalogBank/DocumentLabel/libs/resources.py'
    resource_file = '/mnt/data_drive/CSU_PhD/research/software/DocumentLabeler/libs/resources.py'
    if not os.path.exists(resource_file):
        output = os.system('pyrcc5 -o libs/resources.py resources.qrc')
#        assert output == 0, "operate the cmd have some problems ,please check  whether there is a in the lib " \
#                            "directory resources.py "

    sys.exit(main())
