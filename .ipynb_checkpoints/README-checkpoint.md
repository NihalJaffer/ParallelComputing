# Lab3_P1
* Build a data parallel model program using threads in Python.
• Build a data parallel model program using processes in Python.
• Understand the basics of parallel programming using Python's threading and multiprocessing modules.

# Questions answered
1.*Sequential Execution: The entire summation is performed in a single loop, which takes a significant amount of time.
  *Threaded Execution: Since Python has the Global Interpreter Lock (GIL), threading does not improve CPU-bound tasks like summation significantly. It may show       some improvements due to parallel memory access but is generally not much faster.
  *Multiprocessing Execution: This method creates separate processes, bypassing the GIL, and enables true parallel execution on multi-core CPUs. This usually         results in a significant speedup compared to sequential and threaded execution.

2.Sequential Execution Time:0.00211 sec

  Thread Execution Time:0.00675 sec
  Speedup: 0.313
  Efficiency: 0.078
  Amdahl's Law Speedup: 3.88
  Gustafson's Law Speedup: 3.96

  Multiprocess Excecution Time:Execution Time: 0.41368 sec
  Speedup: 0.0051
  Efficiency: 0.0013
  Amdahl's Law Speedup: 3.88
  Gustafson's Law Speedup: 3.96

 Sequential execution was the fastest because Python's built-in sum() function is highly optimized.
Threading did not provide significant speedup due to the Global Interpreter Lock (GIL).
Multiprocessing was significantly slower due to the overhead of creating processes and inter-process communication (IPC). For such a simple operation, the overhead outweighs the benefits.
