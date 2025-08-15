// enum state {Pending, Running, Complete, Purged};
struct request
{
	unsigned int id;  /* request size */
	unsigned int size;  /* request size */
	unsigned int duplicates;    /* request fanout size */
	unsigned int *server_priorities;    /* priorities of replica servers; 0 means not a replica, 1 is the highest priority */
	unsigned int dscp;  /* DSCP of request */
	unsigned int rate;  /* sending rate of request */
	unsigned int sleep_us;  /* sleep time interval */
	struct timeval start_time;  /* start time of request */
	struct timeval stop_time;   /* stop time of request */
	pthread_mutex_t lock;   /* lock for the request, used when multiple flows try to update request variables */
	unsigned int bytes_completed; /* total bytes received from all of the flows in aggregate
																	OR the bytes received from the fastest flow*/
	// enum state status; /*TODO: make use of this, not being used properly atm*/
	bool complete; //todo: verify if this is truly atomic
	// char* buff; // buffer to assemble the response data

};

