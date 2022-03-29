import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class Canvas(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100, parent=None):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(Canvas, self).__init__(self.fig)
        self.setParent(parent)