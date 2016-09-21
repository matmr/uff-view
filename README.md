# Introduction

This very basic script relies heavily on the [pyuff](https://github.com/openmodal/pyuff) library to read the contents of files written in the **Universal File Format**, offering a simple GUI on top. It is capable of rendering nodes in a 3D view, plotting measured frequency-response functions and listing other contents of the file in a table-like widget.

Fields in the Universal file format can be used for storing measured data and results of structural dynamic analyses. This quite old format is still in use today. A very helpful reference for deciphering the contents can be found [here](http://sdrl.uc.edu/sdrl/referenceinfo/universalfileformats).

This tool was built in the early days of OpenModal project (check [here](https://github.com/openmodal/openmodal) and [here](http://www.openmodal.com)) and had some influence in how its underlying data model was setup. Since lots of improvements were made in the meantime and because uff-view is not actively developed (at the moment), you should probably use [OpenModal](https://github.com/openmodal/openmodal/releases) for opening .uff/.unv files.

# Install & Run

Lots of dependencies will already be installed if you are using a Python distribution, such as [Anaconda](https://www.continuum.io/downloads) (recommended).

```
>>>pip install PySide numpy pandas pyqtgraph
>>>git clone https://github.com/matmr/uff-view
>>>cd uff-view
>>>py uff-view.py
```

## Dependencies
* **PySide**, (should also work with PyQt4)
* **pyqtgraph**, for 3D and 2D plots
* **pandas**, for storing data
* **numpy**, for number crunching
