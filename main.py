import sys

from PyQt6.QtWidgets import QApplication

from ingestion.networking import RSSIWorker
from presentation.gui import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("byakugan")

    window = MainWindow()
    window.show()

    rssi_worker = RSSIWorker()
    rssi_worker.rssi_updated.connect(window.update_rssi)
    rssi_worker.source_changed.connect(window.update_source)
    rssi_worker.start()

    app.aboutToQuit.connect(rssi_worker.stop)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
