/*******************************************************************************
 * A Data Plane native PPV PIE Active Queue Management Scheme using P4 on a Programmable Switching ASIC.
 * Karlstad University 2021.
 * Author: L. Dahlberg
 ******************************************************************************/


#include <core.p4>
#if __TARGET_TOFINO__ == 2
#include <t2na.p4>
#else
#include <tna.p4>
#endif

#include "./common/ExtraHeaders.p4"

struct register_operations_ingress {
	bit<8>     State;
	bit<8>     Active;
}

control IngressProcessing (in bit<32> ingress_tstamp, 
                        in bit<9> egress_port,
                        out bit<8> State,
                        in bit<9> ingress_port)
{
    bit<8> timeToUpdate; 
    bit<32> TIME_BLOCK;                                                                  // Constant used so that the timer does not overflow.

    Register<bit<32>, _>(N_PORTS, 0) update_time;
    RegisterAction<bit<32>, _, bit<8>>(update_time)
        update_status = {
            void apply(inout bit<32> value, out bit<8> output) {
                output = 8w0;
                if(ingress_tstamp >= value || value > TIME_BLOCK){                       // True if CONTORL_INTERVAL time has passed or if time has surrpassed the TIME_BLOCK. 
                    output = 8w1;
                    value = ingress_tstamp + CONTROL_INTERVAL;                           // Store new threshold time.
                }
            }
        };

    Register<register_operations_ingress, _>(N_PORTS) stateMachine;  
    RegisterAction<register_operations_ingress, _, bit<8>>(stateMachine)
        IncrementState = {
            void apply(inout register_operations_ingress value, out bit<8> output) 
            {
                if(value.State < STATE_FIFTH)
                {
                    value.State = value.State + STATE_INCREMENT;
                    output = value.State;
                }
                if(value.State == STATE_FIFTH)
                {
                    value.State = STATE_INACTIVE;
                }
            }
        };

    apply{
        TIME_BLOCK = 4000000000;                                                         
        timeToUpdate = 8w0;
        if(ingress_port != RECIRCULATOIN_PORT)
        {
            timeToUpdate = update_status.execute(egress_port);                           // Checks whether it is time to update CTV or not.
            if(timeToUpdate == 8w1)
            {
                State = STATE_INITIAL;
            }
            else 
                State = STATE_INACTIVE;
        }    
        else                                                                             // Packet is recirculted.
        {
            State = IncrementState.execute(egress_port);
        }
    }
}
