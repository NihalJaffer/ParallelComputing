import threading
import time
from function import simulate_sensor, process_temperatures, initialize_display, update_display

num_sensors = 3

sensor_threads = []
for i in range(num_sensors):
    sensor_id = f"Sensor-{i+1}"
    sensor_thread = threading.Thread(target=simulate_sensor, args=(sensor_id,), daemon=True)
    sensor_threads.append(sensor_thread)
    sensor_thread.start()


processing_thread = threading.Thread(target=process_temperatures, daemon=True)
processing_thread.start()

display_thread = threading.Thread(target=update_display, daemon=True)
display_thread.start()

initialize_display()
while True:
    time.sleep(1)
