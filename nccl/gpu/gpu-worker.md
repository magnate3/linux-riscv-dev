

# cuda with cpu worker
```
Create Event: cudaEvent_t myEvent; cudaEventCreate(&myEvent); (on the CPU).

Record Event: In your CUDA code (CPU side, before kernel launch or from callback), you associate the event with a stream: cudaEventRecord(myEvent, stream);.

Worker Executes: The GPU kernel (the "worker") runs asynchronously in the specified stream.

CPU Waits/Checks: On the CPU, you can:

Wait: cudaEventSynchronize(myEvent); (blocks CPU until GPU work is done).

Query: cudaEventQuery(myEvent); (non-blocking check).

Time: cudaEventElapsedTime(&ms, startEvent, stopEvent); (after recording both events).

```

```
To benchmark a kernel execution, we use events. Events are like checkpointing stopwatches in GPU streams.

#### Defining events:

float runtime;
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
Using events:

cudaEventRecord(start, 0);
/* launch kernel */
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&runtime, start, stop);

####  Destroying events:

cudaEventDestroy(stop);
cudaEventDestroy(start);
```