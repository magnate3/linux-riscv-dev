#include "readerwriterqueue.h"
#include <thread>

using namespace moodycamel;

int main() {
    BlockingReaderWriterQueue<int> q;
    
    std::thread reader([&]() {
        int item;
        for (int i = 0; i != 100; ++i) {
            // Fully-blocking:
            q.wait_dequeue(item);
    
            // Blocking with timeout
            if (q.wait_dequeue_timed(item, std::chrono::milliseconds(5)))
                ++i;
        }
    });
    
    std::thread writer([&]() {
        for (int i = 0; i != 100; ++i) {
            q.enqueue(i);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });
    writer.join();
    reader.join();
    
    assert(q.size_approx() == 0);
    return 0;
}
