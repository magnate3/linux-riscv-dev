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

#include "./common/util.p4"
#include "./common/headers.p4"
#include "./common/ExtraHeaders.p4"

struct register_operations_32
{
    bit<32> first;
    bit<32> second;
}

control EgressProcessing (in bit<32> delay,
                    in bit<9> egress_port,
                    inout recirculation_h recirculationData)
{

    /**
     * Used by BetaMult.
     *
    **/
    action BetaHit(int<32> result)
	{
		recirculationData.interm_beta = result;
	}

    /**
     * Table that perform the multiplication that uses beta value.  
     *
    **/
	table BetaMult
	{
		key = {
		recirculationData.interm_beta: ternary;
		}
		actions = {
			BetaHit;
		}
        size = MULT_SIZE;
	}

    /**
     * Used by AlphaMult.
     *
    **/
    action AlphaHit(int<32> result)
	{
		recirculationData.interm_alpha = result;
	}

    /**
     * Table that perform the multiplication that uses alpha value.  
     *
    **/
	table AlphaMult
	{
		key = {
		recirculationData.interm_alpha: ternary;
		}
		actions = {
			AlphaHit;
		}
        size = MULT_SIZE;
	}

    /**
     * Used by GetCTV.
     *
    **/
    action CTV(bit<16> result)
	{
		recirculationData.CTV = result;
	}

    /**
     * Table that matches probability to CTV. Filled dynamically by the Control Plane.
     *
    **/ 
	table GetCTV
	{
		key = {
		recirculationData.prob: exact;
		}
		actions = {
			CTV;
		}
        size = CTV_SIZE;
	} 

    /**
     * Used by ProbModification.
     *
    **/
    action Div2Square(bit<32> result)
	{
		recirculationData.prob = result;
	}

    /**
     * Table that performs the square divided by two operation on intermediate probability.
     *
    **/
	table ProbModification
	{
		key = {
		recirculationData.prob_before_table: ternary;
		}
		actions = {
			Div2Square;
		}
        size = PROB_SIZE;
	}

    /**
     *  Used to add the intermediate probability value to the old probablity, return and save it.
     * 
    **/
    Register<register_operations_32, _>(N_PORTS) prev_prob;
    RegisterAction<register_operations_32, _, bit<32>>(prev_prob)
        Get_prob = {
            void apply(inout register_operations_32 value, out bit<32> update_out) {
                value.second = (bit<32>)(recirculationData.interm_prob + (int<32>)value.first);
                update_out = value.second;
            }
        };

    RegisterAction<register_operations_32, _, bit<32>>(prev_prob)
            Store_prob = {
                void apply(inout register_operations_32 value, out bit<32> update_out) {
                    value.first = recirculationData.prob;
                }
            };

    /**
     *  Gets the previous delay and saves the new delay.
     *
    **/
    Register<bit<32>, _>(N_PORTS) prev_delay;
    RegisterAction<bit<32>, _, bit<32>>(prev_delay)
             GetOldAndSetNew_prev_delay = {
                void apply(inout bit<32> value, out bit<32> update_out) {
                    update_out = value;
                    value = (bit<32>)recirculationData.delay;
                }
            }; 

    /**
	 * Temporary register used for debugging purposes.
	 *
	**/
    
    Register<bit<32>, _>(N_PORTS) debug1;
    RegisterAction<bit<32>, _, bit<32>>(debug1)
             debug_1 = {
                void apply(inout bit<32> value, out bit<32> update_out) {
                    value = (bit<32>)recirculationData.CTV;
                }
            }; 
    
    Register<bit<32>, _>(N_PORTS) debug2;
    RegisterAction<bit<32>, _, bit<32>>(debug2)
             debug_2 = {
                void apply(inout bit<32> value, out bit<32> update_out) {
                    value = (bit<32>)recirculationData.interm_beta;
                }
            };
    
    Register<bit<32>, _>(N_PORTS) debug3;
    RegisterAction<bit<32>, _, bit<32>>(debug3)
             debug_3 = {
                void apply(inout bit<32> value, out bit<32> update_out) {
                    value = (bit<32>)recirculationData.interm_alpha;
                }
            }; 
        
        
    Register<bit<32>, _>(N_PORTS) debug4;
    RegisterAction<bit<32>, _, bit<32>>(debug4)
             debug_4 = {
                void apply(inout bit<32> value, out bit<32> update_out) {
                    value = recirculationData.prob;
                }
            };

    Register<bit<32>, _>(N_PORTS) debug5;
    RegisterAction<bit<32>, _, bit<32>>(debug5)
             debug_5 = {
                void apply(inout bit<32> value, out bit<32> update_out) {
                    value = recirculationData.delay;
                }
            };
    
    Register<bit<32>, _>(N_PORTS) debug6;
    RegisterAction<bit<32>, _, bit<32>>(debug6)
             debug_6 = {
                void apply(inout bit<32> value, out bit<32> update_out) {
                    value = (bit<32>)recirculationData.interm_prob;
                }
            };
    Register<bit<32>, _>(N_PORTS) debug7;
    RegisterAction<bit<32>, _, bit<32>>(debug7)
             debug_7 = {
                void apply(inout bit<32> value, out bit<32> update_out) {
                    value = recirculationData.prob_before_table;
                }
            };


    /**
     * End debug.
     *
    **/


    /**
     * Calculation:  p = ( (previous_probability + alpha(Current_delay - Reference_delay) + beta(Current_delay - Old_delay) )^2 )/2
     *
    **/
    apply
    {
        if(recirculationData.State == STATE_INITIAL)                                                                            // Save delay.
        {
            recirculationData.delay = delay;
            debug_5.execute(egress_port);
        }
        else if(recirculationData.State == STATE_SECOND)                                                                        // Calcualte beta.
        {
            recirculationData.delay_old = GetOldAndSetNew_prev_delay.execute(egress_port); 
            recirculationData.interm_beta = (int<32>)recirculationData.delay - (int<32>)recirculationData.delay_old;
            debug_2.execute(egress_port);
            BetaMult.apply();
        }
        else if(recirculationData.State == STATE_THIRD)                                                                         // Calculate alpha.
        {
            recirculationData.interm_alpha = (int<32>)recirculationData.delay - (int<32>)REFERENCE_DELAY;
            debug_3.execute(egress_port);
            AlphaMult.apply();
        }
        else if(recirculationData.State == STATE_FOURTH)                                                                        // ((Alpha + Beta)^2 )/2.
        { 
            recirculationData.interm_prob = (int<32>)recirculationData.interm_alpha + (int<32>)recirculationData.interm_beta;  
            debug_6.execute(egress_port);
            recirculationData.prob_before_table = Get_prob.execute(egress_port);
            debug_7.execute(egress_port); 
            if(ProbModification.apply().hit)
            {
                debug_4.execute(egress_port);
            } 
        }
        else if(recirculationData.State == STATE_FIFTH)                                                                         // Prob -> CTV.
        {
            Store_prob.execute(egress_port);
            if(GetCTV.apply().hit)
            {
                debug_1.execute(egress_port);
            }                                                                                                 
            recirculationData.TimeToUpdateCTV = 8w1; 
        }
    }
}