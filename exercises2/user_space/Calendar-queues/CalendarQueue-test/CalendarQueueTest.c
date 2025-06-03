//#include "CalendarQueue.c"
#include "calqueue.h"
#include<stdlib.h>
#include<math.h>
void queue_init(void) {

	calqueue_init();
}
int main()
{
    double timestamp = 128;
    queue_init();
    //timestamp = 20 * Random();    
    calqueue_put(timestamp, NULL);
    return 0;
}
