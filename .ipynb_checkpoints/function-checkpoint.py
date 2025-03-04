import random
import time
import threading
import queue
import os

latest_temp = {}
temp_averages = {}

temperature_queue = queue.Queue()


data_lock = threading.RLock() 
condition = threading.Condition()  

def simulate_sensor(sensor_id):
    
    while True:
        temp = random.randint(15, 40)
        
        with data_lock:
            latest_temp[sensor_id] = temp
        
        with condition:
            temperature_queue.put((sensor_id, temp))
            condition.notify()  

        time.sleep(1)

def process_temperatures():
    temp_records = {}

    while True:
        with condition:
            while temperature_queue.empty():
                condition.wait()  

            sensor_id, temp = temperature_queue.get()

        with data_lock:
            if sensor_id not in temp_records:
                temp_records[sensor_id] = []
            
            temp_records[sensor_id].append(temp)
            
            if len(temp_records[sensor_id]) > 10:
                temp_records[sensor_id].pop(0)

            avg_temp = sum(temp_records[sensor_id]) / len(temp_records[sensor_id])
            temp_averages[sensor_id] = round(avg_temp, 2)

def initialize_display():
    os.system("cls" if os.name == "nt" else "clear")
    print("Current temperatures:")
    sensors = [f"Sensor {i}: --°C" for i in range(3)]
    print("Latest Temperatures: " + " ".join(sensors))
    
    for i in range(3):
        print(f"Sensor {i+1} Average: --°C")

def update_display():
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("Current temperatures:")
        
        with data_lock:
            sensor_data = [f"Sensor {i}: {latest_temp.get(f'Sensor-{i+1}', '--')}°C" for i in range(3)]
            print("Latest Temperatures: " + " ".join(sensor_data))

            for i in range(3):
                avg_temp = temp_averages.get(f"Sensor-{i+1}", "--")
                print(f"Sensor {i+1} Average: {avg_temp}°C")

        time.sleep(5)  
