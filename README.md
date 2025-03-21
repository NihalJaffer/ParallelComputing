# Assignment1-Part1/2
• Develop Python programs that take advantage of python multiprocessing capabilities.


# Questions answered

# Conclusions from performance test.

For **10^6**:
• Sequential processing was the fastest at 0.0589 seconds because it avoids multiprocessing overhead.
• Multiprocessing using Pool.map took 0.1272 seconds, showing an improvement but still with some overhead.
• Concurrent Futures (ProcessPoolExecutor) was extremely slow at 100.1509 seconds, likely due to inefficient task distribution.

For **10^7**:

• Sequential processing took 0.5422 seconds, which scales reasonably.
• Multiprocessing (Pool.map) took 0.8270 seconds, slightly slower than expected but still beneficial.
• Concurrent Futures (ProcessPoolExecutor) was still extremely slow (1013.0337 seconds), indicating a major inefficiency when handling  large lists.

This test confirms that for **10^7** sequential execution scales well.
Multiprocessing improves performance slightly, but Concurrent Futures remains extremely inefficient.
The bottleneck in **ProcessPoolExecutor** suggests that  synchronization overhead is excessive.



## What Happens When More Processes Try to Access the Pool Than Available Connections?
• In the test, 6 processes tried to acquire connections when only 3 were available.
• Three processes acquired connections immediately.
• The remaining three waited until a connection was released, and only then could they proceed.

## How Does the Semaphore Prevent Race Conditions?
• Semaphores act as a counter, ensuring that only a limited number of processes can access a resource.
• If no connections are available, new processes must wait until an existing process releases a connection.
• This prevents simultaneous access to a shared resource, which could cause data corruption or system crashes.

