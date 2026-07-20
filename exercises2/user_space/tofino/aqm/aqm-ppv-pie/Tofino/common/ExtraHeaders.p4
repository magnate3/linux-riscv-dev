/*******************************************************************************
	Programmable Packet Scheduler. Karlstad University 2021.
	Author: L. Dahlberg
******************************************************************************/

//Start Variables for evaluation

#define COUNTER_SIZE          32w1025	      	// Size of pv_histogram counter in Ingress.
#define CTV_SIZE              32w512    		// Size of GetCTV table in Egress.
#define PROB_SIZE             32w1666        	// size of Probmodification table in Egress.
#define CONTROL_INTERVAL      32w5000000    	// 5ms in nsec. CTV update interval.
#define REFERENCE_DELAY       32w2000000       	// 2ms in nsec. Reference delay.

//End Variables for evaluation

#define N_PORTS               32w70          	// Number of ports pv_ports.
#define STATE_INCREMENT       8w1           	// The increment of the state machine.
#define STATE_INACTIVE        8w0           	// The value to use when state machine is inactive.
#define STATE_MAX             8w8           	// The max value of the states.          
#define STATE_INITIAL         8w5           	// The inital state when the timer is triggered should be last state + 1.
#define STATE_SECOND          8w1           	
#define STATE_THIRD           8w2
#define STATE_FOURTH          8w3 
#define STATE_FIFTH           8w4 
#define RECIRCULATOIN_PORT    9w68
#define MULT_SIZE             32w3328        	// Size of Beta and Alpha Mult tables in Egress.
