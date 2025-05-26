#include "linkedlist.c"

#define QSPACE 15000

typedef struct calendar_queue CalendarQueue;

List POOL[QSPACE];

CalendarQueue* initqueue();
void enqueue(struct node* entry, double priority, CalendarQueue* queue);
struct node* dequeue(CalendarQueue* queue);
void display_queue(CalendarQueue* queue);

enum BOOLEAN{FALSE, TRUE};

struct calendar_queue {
    int firstsub;
    List* bucket;
    double buckettop;
    double width;
    int nbuckets;
    int qsize;
    int lastbucket;
    double lastprio;
    int top_threshold;
    int bot_threshold;
    int resizeenabled;
};

int mask_modulo(int m, int n) { return m & (n - 1);}

void localinit(CalendarQueue* q, int qbase, int nbuck,
                        double bwidth, double startprio) {
    int i=0;
    long int n = 0;
    /* Set position and size of new queue. */
    q->firstsub = qbase;
    q->bucket = &POOL[qbase];
    q->width = bwidth;
    q->nbuckets = nbuck;
    //Calculate bit mask for modulo nbuckets operation; ???
    /* Initialize as empty. */
    q->qsize = 0;
    for(i = 0; i < q->nbuckets; ++i) q->bucket[i] = NULL;
    /* Set up initial position in queue. */
    q->lastprio = startprio;
    n = (long int)startprio/ q->width; 
    /* Virtual bucket */
    q->lastbucket = mask_modulo(n, q->nbuckets);
    q->buckettop = (n + 1) * q->width + 0.5 * q->width;
    /* Set up queue size change thresholds. */
    q->bot_threshold = q->nbuckets/2 - 2;
    q->top_threshold = 2 * q->nbuckets;
}

CalendarQueue* initqueue() {
    CalendarQueue* init_q = (CalendarQueue*)malloc(sizeof(CalendarQueue)); 
	localinit(init_q, 0, 2, 1.0, 0.0);
    init_q->resizeenabled = TRUE;
	return init_q;
}

double newwidth(CalendarQueue* q) {
    /* This calculates the width to use for buckets. */
    int nsamples;
    int i = 0;
    /* Decide how many queue elements to sample. */
    if(q->qsize < 2) return(1.0);
    if(q->qsize <= 5)
        nsamples = q->qsize;
    else
        nsamples = 5 + q->qsize/10;
    if(nsamples > 25) nsamples = 25;
    double lastprio = q->lastprio;
    int lastbucket = q->lastbucket;
    double buckettop = q->buckettop;
    
    q->resizeenabled = FALSE;
    List* dqsamples = makelist();
    double* samplesprio = (double*)malloc(sizeof(double) * nsamples);
    
    for(i=0; i < nsamples; ++i) {
        node* n = dequeue(q);
        add(n, dqsamples);
        samplesprio[i] = n->endTime;
    }

    for(i=0; i < nsamples; ++i) {
        node* n = delete(*dqsamples, dqsamples);
        enqueue(n, n->endTime, q);
    }
    destroy(dqsamples);

    q->lastprio = lastprio;
    q->lastbucket = lastbucket;
    q->buckettop = buckettop;
    q->resizeenabled = TRUE;

    double sum = 0;
    for(i=0; i < nsamples - 1; ++i) {
        sum += (samplesprio[i + 1] - samplesprio[i]);
    }
    double avg_seperation = sum / (nsamples - 1);
    int m = 0;
    sum = 0;
    for(i=0; i < nsamples - 1; ++i) {
		double spr = samplesprio[i + 1] - samplesprio[i];
        if(spr < 2*avg_seperation) {
            sum += spr;
            ++m;
        }
    }
    double final_seperation = (sum == 0)? (1.0 / 3) : sum / m;
    free(samplesprio);
    return(3.0 * final_seperation);
}

void resize(CalendarQueue* q, int newsize) 
/* This copies the queue onto a calendar with newsize
buckets. The new bucket array is on the opposite
end of the array a[QSPACE] from the original. */
{
    double bwidth;
    int i;
    int oldnbuckets;
    List * oldbucket;
    if (q->resizeenabled == FALSE) return;
    bwidth = newwidth(q); /* Find new bucket width. */
    /* Save location and size of old calendar for use
    when copying calendar. */
    oldbucket = q->bucket; oldnbuckets = q->nbuckets;
    /* Initialize new calendar. */
    if(q->firstsub == 0)
		localinit(q, QSPACE-newsize, newsize, bwidth, q->lastprio);
    else
        localinit(q, 0, newsize, bwidth, q->lastprio);
	q->resizeenabled = TRUE;
    /* Copy queue elements to new calendar. */
	for (i = oldnbuckets - 1; i >= 0; --i) {
		node* current = oldbucket[i];
		while (current != NULL) {
			node* next = current->next;
			enqueue(current, current->endTime, q);
			current = next;
		}
		//Transfer elements from bucket i to new calendar
		//by enqueueing them;
	}
}

void enqueue(node* entry, double priority, CalendarQueue* q) 
/* This adds one entry to the queue. */
{
    int i;
    /* Calculate the number of the bucket in which to
    place the new entry. */
    i = (long int)priority/ q->width; /* Find virtual bucket.*/
    i = i % q->nbuckets; /* Find actual bucket. */
    add(entry, &(q->bucket[i])); /*Insert entry into bucket i in sorted list. */
    ++(q->qsize); /* Update record of queue size. */
    /* Double the calendar size if needed. */
    if (q->qsize > q->top_threshold) resize(q, 2 * q->nbuckets);
}

struct node* dequeue(CalendarQueue* q)
/* This removes the lowest priority node from the
queue and returns a pointer to the node containing
it. */
{
    register int i;
    if (q->qsize == 0) return(NULL);
    for (i = q->lastbucket; ; ) /* Search buckets */
    {
        /* Check bucket i */
        if (q->bucket[i] != NULL && q->bucket[i]->endTime < q->buckettop)
        {   /* Item to dequeue has been found. */
            node* dequeue_i = delete(q->bucket[i], &(q->bucket[i]));
            /* Update position on calendar. */
            q->lastbucket = i; q->lastprio = dequeue_i->endTime;
            --(q->qsize);
            /* Halve calendar size if needed. */
            if (q->qsize < q->bot_threshold) resize(q, q->nbuckets/2);
            return dequeue_i;
        }
        else{/* Prepare to check next bucket or else go to a direct search. */
            ++i;
            if(i == q->nbuckets) i = 0;
            q->buckettop += q->width;
            if(i == q->lastbucket) break; /* Go to direct search */
        }
    }
    /* Directly search for minimum priority event. */
	int firstprio = TRUE;
    int lowestprio = 0;
    int lowestbucket = 0;
    for(i=0; i < q->nbuckets; ++i) {
		if (q->bucket[i] == NULL) continue;
		if(firstprio) {
			lowestbucket = i;
			lowestprio = q->bucket[i]->endTime;
			firstprio = FALSE;
		}
		if (q->bucket[i]->endTime < lowestprio) {
			lowestbucket = i;
			lowestprio = q->bucket[i]->endTime;
		}
	}
    q->lastbucket = lowestbucket;
    q->lastprio = lowestprio;
    int vbucket = (int)(lowestprio / (q->width));
    q->buckettop = (vbucket + 1) * q->width + 0.5 * q->width;
    return(dequeue(q)); /* Resume search at minnode. */
}

void display_queue(CalendarQueue* q) {
    int i = 0;
    printf("Calendar Queue: \n");
    printf("First subarray: %d; Bucket's width: %.2lf; Number of buckets: %d; \n", q->firstsub,
    q->width, q->nbuckets);
    printf("Queue's size: %d; Top threshold: %d; Bottom threshold: %d;", q->qsize,
    q->top_threshold, q->bot_threshold);
    for(i=0; i < q->nbuckets; ++i) {
        printf("\n  |\n  v\n");
        printf("Bucket [%d]: ", i);
        display(&(q->bucket[i]), 15);
    }
    printf("\n");
}

void insert(CalendarQueue* q, node* n) {
    enqueue(n, n->endTime, q);
}
