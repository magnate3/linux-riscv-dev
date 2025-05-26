#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "node.c"

void insert(node* entry);
node* removeFirst();
double newwidth();
void resize(int newsize);
void localInit(int nbuck, double bwidth, double startprio);
void initqueue();
void enqueue(node* entry);
node* dequeue();
void printBucket(node* n);
void printBuckets();

node** a;
node** buckets;
double width;
int nbuckets;
int firstsub;
int resizeenable;
int qsize;
double lastprio;
int lastbucket;
double buckettop;
int bot_threshold;
int top_threshold;

void insert(node* entry){
    double priority = entry->endTime;

    // i la vi tri bucket ma entry chen vao
    int i;
    i = priority / width;
    i = i % nbuckets;

    // tren i vao vi tri hop li tren buckets[i]
    if(buckets[i] == NULL || buckets[i]->endTime >= priority){
        entry->next = buckets[i];
        buckets[i] = entry;
    } else {
        node* current = buckets[i];
        while(current->next != NULL ){
            if(current->next->endTime < priority)
                current = current->next;
            else break;
        }

        entry->next = current->next;
        current->next = entry;
    }

    if(priority < lastprio){
        int n = priority / width;
        buckettop = (n+1)*width + 0.5*width;
    }

    // cap nhat qsize : so event cua hang doi
    qsize++;

    return;
}

node* removeFirst(){
    int i;
    if(qsize == 0) return NULL;

    i = lastbucket;
    while(1){
        if(buckets[i] != NULL && buckets[i]->endTime < buckettop){
            node* tmp = buckets[i];
            buckets[i] = tmp->next;

            lastbucket = i;
            lastprio = tmp->endTime;
            qsize--;

            return tmp;
        } else {
            i++; if(i==nbuckets) i=0;
            buckettop += width;
            if(i == lastbucket) break;
        }
    }

    // neu khong tim thay gia tri nho nhat trong nam
    // quay lai tim gia tri nho nhat trong tat ca cac gia tri dau cua buckets
    int minbucket;
    double minpri;

    // start : vi tri dau tien buckets[i] != NULL
    int start;
    for(start=0; start<nbuckets; start++)
        if(buckets[start] != NULL){
            lastbucket = start;
            lastprio = buckets[start]->endTime;
            minpri = buckets[start]->endTime;
            minbucket = start;
            break;
        }

    // tim vi tri buckets[i] != NULL ma nho nhat
    for(int i = start+1; i<nbuckets; i++)
        if(buckets[i] != NULL){
            if(buckets[i]->endTime < minpri){
                lastbucket = i;
                lastprio = buckets[i]->endTime;
                minpri = buckets[i]->endTime;
                minbucket = i;
            }
        }

    node* foo = buckets[minbucket];
    buckets[minbucket] = foo->next;

    int n = lastprio / width;
    buckettop = (n+1) * width + 0.5*width;
    qsize--;

    return foo;
}

double newwidth(){
    int nsamples;

    if(qsize < 2) return 1.0;
    if(qsize <= 5)
        nsamples = qsize;
    else
        nsamples = 5 + qsize/10;

    if(nsamples > 25) nsamples = 25;

    double oldlastprio = lastprio;
    int oldlastbucket = lastbucket;
    double oldbuckkettop = buckettop;


    // lay ra nsamples gia tri mau
    // luc lay ra mau ngan chan viec resize, resizeenable = false
    resizeenable = 0;
    node* save = (node*) calloc(nsamples,sizeof(node));
    for(int i=0; i<nsamples; i++){
        node* tmp = removeFirst();
        save[i] = *tmp;
    }
    resizeenable = 1;

    //  tra lai cac gia tri da lay ra trong hang doi
    for(int i=0; i<nsamples; i++){
        insert(&save[i]);
    }
    lastprio = oldlastprio;
    lastbucket = oldlastbucket;
    buckettop = oldbuckkettop;

    // tinh toan gia tri cho new witdh
    double totalSeparation = 0;
    int end = nsamples;
    int cur = 0;
    int next = cur + 1;
    while(next != end){
        totalSeparation += save[next].endTime - save[cur].endTime;
        cur++;
        next++;
    }
    double twiceAvg = totalSeparation / (nsamples - 1) * 2;

    totalSeparation = 0;
    end = nsamples;
    cur = 0;
    next = cur + 1;
    while(next != end){
        double diff = save[next].endTime - save[cur].endTime;
        if(diff <= twiceAvg){
            totalSeparation += diff;
        }
        cur++;
        next++;
    }
    //printf("%.1f\n",totalSeparation);

    // gia tri width moi = 3 lan do phan tach gia tri trung binh
    totalSeparation *= 3;
    totalSeparation = totalSeparation<1.0? 1.0 : totalSeparation;

    return totalSeparation;
}

void resize(int newsize){
    double bwidth;
    int i;
    int oldnbuckets;
    node** oldbuckets;

    if(!resizeenable) return;

    bwidth = newwidth();
    oldbuckets = buckets;
    oldnbuckets = nbuckets;

    localInit(newsize,bwidth,lastprio);

    // them lai cac phan tu vao calendar moi
    for(int i=0; i<oldnbuckets; i++){
        node* foo = oldbuckets[i];
        while(foo!=NULL){ // tranh vien lap vo han
            node* tmp = new_node(foo->type,foo->idElementInGroup,foo->portID,foo->endTime);
            insert(tmp);
            foo = foo->next;
        }
    }

    return;
}

void localInit(int nbuck, double bwidth, double startprio){
    int i;
    long int n;

    // khoi tao cac tham so
    buckets = (node**) calloc(nbuck,sizeof(node));
    //buckets = malloc(sizeof * buckets * nbuckets);
    width = bwidth;
    nbuckets = nbuck;

    // khoi tao cac bucket
    qsize = 0;
    for(int i=0; i<nbuckets; i++){
        buckets[i] = NULL;
    }

    // khoi tao cac chi so ban dau cua bucket dau tien
    lastprio = startprio;
    n = startprio / width;
    lastbucket = n % nbuckets;
    buckettop = (n+1)*width + 0.5*width;

    // khoi tao 2 linh canh dau vao cuoi
    bot_threshold = nbuckets/2 - 2;
    top_threshold = 2*nbuckets;
}

void initqueue(){
    localInit(2,1,0.0);
    resizeenable = 1;
}

// enqueue
void enqueue(node* entry){
    insert(entry);
    //printf("%.1f\n",width);
    // nhan doi so luong calendar neu can
    if(qsize>top_threshold) resize(2*nbuckets);
}

// dequeue
node* dequeue(){
    node* tmp = removeFirst();

    /*thu hep so luong cua calendar neu can*/
    if(qsize < bot_threshold) resize(nbuckets/2);
    return tmp;
}

/*in ra man hinh lich*/
void printBucket(node* n){
    while(n!=NULL){
        printf("%.1f ",n->endTime);
        n = n->next;
    }
    return;
}
void printBuckets(){
    for(int i=0; i<nbuckets; i++){
        printf("Day %d : ",i);
        node* tmp = buckets[i];
        printBucket(tmp);
        printf("\n");
    }
    printf("\nCount of event : %d\n",qsize);
    printf("so luong bucket : %d\n",nbuckets);
    printf("buckettop : %.1f\n",buckettop);
    printf("lastbuckket : %d\n",lastbucket);
    printf("lastprio : %.1f\n",lastprio);
    printf("width : %.1f\n",width);
    printf("bot : %.1d\n",bot_threshold);
    printf("top : %.1d",top_threshold);
}

/*
int main(){
    initqueue();
    long count = 0;
    double currentTime = 0;
    double endTime = 1000*1000;

    int i;
    for(i = 0; i < 6750; i++)
    {
        enqueue(new_node(A, i, 0, i));
    }


    node * ev = dequeue();
    while(currentTime <= endTime && ev->endTime != -1)
    {   //printf("hello world");
        //printBuckets();
        if(ev->endTime == currentTime)
        {
            count++;
            i = ev->idElementInGroup;//Lay id cua host trong danh sach cac hosts
            if(ev->type == A)
            {
                enqueue(new_node(A, i, 0, currentTime + 10000));
                //printf("%.1f\n",currentTime);
                //printf("width : %.1f\n",width);
                enqueue(new_node(B, i, 0, currentTime));
            }
            else if(ev->type == B)
            {
                enqueue(new_node(C, i, 0, currentTime));
            }
            ev->endTime = -1;
            ev = dequeue();

            currentTime = ev->endTime;
        }
    }

    printf("count = %ld\n", count);
    printf("================================\n");

    //enqueue(new_node(A,0,0,16));
    //enqueue(new_node(A,0,0,16.2));
    //enqueue(new_node(A,0,0,17));
    //printf("%.1f \n",dequeue()->endTime);
    //enqueue(new_node(A,0,0,13.7));
    //printf("%.1f \n",dequeue()->endTime);
    //enqueue(new_node(A,0,0,14.5));
    //enqueue(new_node(A,0,0,14.7));
    //enqueue(new_node(A,0,0,14.8));
    //enqueue(new_node(A,0,0,15.7));
    //enqueue(new_node(A,0,0,13.7));
    //enqueue(new_node(A,0,0,16.7));
    //enqueue(new_node(A,0,0,10.7));
    //enqueue(new_node(A,0,0,20.7));
    //if(qsize>top_threshold) resize(2*nbuckets);
    //enqueue(new_node(A,0,0,13.7));
    //resize(nbuckets*2);
    //printf("%.1f \n",newwidth());
    //printf("%.1f \n",dequeue()->endTime);
    //printf("%.1f \n",dequeue()->endTime);
    //printf("%.1f \n",dequeue()->endTime);
    //printf("%.1f \n",dequeue()->endTime);
    //printf("%.1f \n",dequeue()->endTime);
    //printf("%.1f \n",dequeue()->endTime);
    //printBuckets();
}
*/
