//#include "CalendarQueue.c"
#include "calqueue.h"
#include<stdlib.h>
#include<math.h>
static calqueue *fel;
int main()
{
    double timestamp = 128;
    // Allocate and initialize FEL
    fel = malloc(sizeof(calqueue));
    calqueue_init(fel);
    //timestamp = 20 * Random();    
    calqueue_put(fel,timestamp, NULL);
    free(fel);
    return 0;
}
