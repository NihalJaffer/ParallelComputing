# Lab4_P2
*Develop a Python program that simulates temperature readings from
multiple sensors, calculates average temperatures, and displays the
information in real-time in the console

# Answers to the questions.
1.We used different synchronization mechanisms to ensure safe access to shared data structures and control thread execution:

 'RLock' : Prevents race conditions when multiple sensors update the "latest_temperatures" dictionary. 
 'Condition':  Allows processing to wait until new sensor data arrives, preventing unnecessary CPU usage. 
'RLock': Ensures safe reading of "latest_temp" and "temp_averages" while multiple threads modify them. 

2.The  main focus is on thread synchronization and safe data access.
- The main goal is to demonstrate concurrent execution.
