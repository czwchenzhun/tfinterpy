from tfinterpy.gui.tool import *

if __name__ == '__main__':
    app = QApplication(sys.argv)
    exe = Tool()
    exe.showMaximized()
    sys.exit(app.exec_())
