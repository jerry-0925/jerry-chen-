import sys

import numpy as np
import serial
from PyQt5 import QtWidgets # GUI components
import pyqtgraph as pg # data visualization
from pyqtgraph.Qt import QtCore # Qt core functions
import csv
import os
from datetime import datetime

class DynamicPlot(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # set window title and size
        self.setWindowTitle("EMG graph")
        self.setGeometry(100,100,1920,1120)

        # set graphing widget and set as main component
        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)

        # set background color
        self.plot_widget.setBackground("black")

        # initialize data
        self.x_data = np.arange(500)
        self.y_data1 = np.zeros(500)
        self.y_data2 = np.zeros(500)
        self.y_data3 = np.zeros(500)
        self.y_data4 = np.zeros(500)
        self.y_data5 = np.zeros(500)
        self.y_data6 = np.zeros(500)

        # create a curve with colors
        self.curve1 = self.plot_widget.plot(self.x_data,self.y_data1, pen="w") # white
        self.curve2 = self.plot_widget.plot(self.x_data, self.y_data2, pen="g")  # green
        self.curve3 = self.plot_widget.plot(self.x_data, self.y_data3, pen="b")  # blue
        self.curve4 = self.plot_widget.plot(self.x_data, self.y_data4, pen="r")  # red
        self.curve5 = self.plot_widget.plot(self.x_data, self.y_data5, pen="y")  # yellow
        self.curve6 = self.plot_widget.plot(self.x_data, self.y_data6, pen="c")  # cyan

        self.data_buffer = []
        self.buffer_size = 50 # write 50 lines of data at a time
        self.count = 0
        # create data storage directory
        self.DATA_DIR = "emg_data"
        if not os.path.exists(self.DATA_DIR):
            os.makedirs(self.DATA_DIR)

        # create csv file
        self.csv_filename = self.create_csv_file()


        # create a timer to update graph
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1) # trigger every ms

        # set y_axis range
        self.plot_widget.setYRange(0,3500)

    def create_csv_file(self):
        # create file and write head
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.DATA_DIR}/emg_data_{timestamp}.csv"

        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestamp', 'Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5', 'Channel6'])

        print(f"EMG data saved to: {filename}")
        return filename

    def update_csv(self, data):
        with open(self.csv_filename, "a", newline='') as csvfile:
            while self.count < 50:
                writer = csv.writer(csvfile)
                writer.writerow(data[self.count])
                self.count += 1



    def update_plot(self):
        # update graph data
        data = muscle_serial.readline().decode('utf-8')
        s2 = data.split(',')
        print(s2)

        if s2[0] == '' or s2[0] == '\r\n' or s2[0] == '\n':
            return

        # get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        data_row = [timestamp]



        # update curve data
        self.y_data1 = np.roll(self.y_data1, -1)
        self.y_data1[-1] = int(s2[0])
        data_row.append(int(s2[0]))

        self.y_data2 = np.roll(self.y_data2, -1)
        self.y_data2[-1] = int(s2[1]) if len(s2) > 1 else 0
        data_row.append(int(s2[1]))

        self.y_data3 = np.roll(self.y_data3, -1)
        self.y_data3[-1] = int(s2[2]) if len(s2) > 2 else 0
        data_row.append(int(s2[2]))

        self.y_data4 = np.roll(self.y_data4, -1)
        self.y_data4[-1] = int(s2[3]) if len(s2) > 3 else 0
        data_row.append(int(s2[3]))

        self.y_data5 = np.roll(self.y_data5, -1)
        self.y_data5[-1] = int(s2[4]) if len(s2) > 4 else 0
        data_row.append(int(s2[4]))

        self.y_data6 = np.roll(self.y_data6, -1)
        self.y_data6[-1] = int(s2[5]) if len(s2) > 5 else 0
        data_row.append(int(s2[5]))

        # send data to buffer
        self.data_buffer.append(data_row)
        if len(self.data_buffer) >= self.buffer_size:
            self.update_csv(self.data_buffer)
            self.data_buffer = [] # clear buffer
            self.count = 0
        # update curve display
        self.curve1.setData(self.x_data, self.y_data1)
        self.curve2.setData(self.x_data, self.y_data2)
        self.curve3.setData(self.x_data, self.y_data3)
        self.curve4.setData(self.x_data, self.y_data4)
        self.curve5.setData(self.x_data, self.y_data5)
        self.curve6.setData(self.x_data, self.y_data6)




if __name__ == "__main__":
    muscle_serial = serial.Serial('/dev/tty.usbserial-1110', 115200, timeout=1)

    # create Qt app
    app = QtWidgets.QApplication(sys.argv)
    main = DynamicPlot()
    main.show()
    sys.exit(app.exec_()) # start event cycle, end when program exits



