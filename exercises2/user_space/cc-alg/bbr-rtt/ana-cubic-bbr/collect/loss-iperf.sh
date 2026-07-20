t=10

runtime="10 seconds"

for j in {1..3}  
do  
        for b in {0.001,0.005,0.01,0.05,0.1,0.5,0,4,6,8,10}
        do
               

                for i in {reno,pcc,balia,bbr,lia,olia,wvegas,pcc_loss}
                do
                        echo "$j" | tr '\n' ',' >>  results-trialc.txt
                        echo "$i" | tr '\n' ',' >>  results-trialc.txt
                        echo "$b" | tr '\n' ',' >>  results-trialc.txt

                        endtime=$(date -ud "$runtime" +%s)

                        sudo ssh router1 -f sudo tc qdisc replace dev enp6s0f1 parent 1:3 handle 3: bfifo limit 384000

                        sudo ssh router2 -f sudo tc qdisc replace dev enp6s0f0 parent 1:3 handle 3: bfifo limit 384000

                        sudo ssh -p 22 emulator1 -f sudo tc qdisc replace dev enp6s0f1 root netem loss $b% delay 30ms limit 60000

                        sudo ssh -p 22 emulator2 -f sudo tc qdisc replace dev enp6s0f1 root netem loss $b% delay 30ms limit 60000

                        sudo sysctl -w net.ipv4.tcp_congestion_control=$i

                        if [[ "$i" == "bbr" ]]
                        then
                                sudo sysctl -w net.mptcp.mptcp_scheduler=default_pacing
                        elif [[ "$i" == "pcc" ]]
                        then
                                sudo sysctl -w net.mptcp.mptcp_scheduler=default_pacing
                        elif [[ "$i" == "pcc_loss" ]]
                        then
                                sudo sysctl -w net.mptcp.mptcp_scheduler=default_pacing
                        else
                                sudo sysctl -w net.mptcp.mptcp_scheduler=default
                        fi



                        iperf3 -f m -c 192.168.3.1 -p 5101 -C "$i" -P 1 -i 0.1 -t $t | ts '%.s' | tee $i-$j-$b-trial-iperf.txt > /dev/null & rm -f $i-$j-$b-trial-ss.txt 2>&1 & 
                        iperf3 -f m -c 192.168.3.1 -p 5102 -C "reno" -P 1 -i 0.1 -t $t | ts '%.s' | tee $i-$j-$b-trial-iperf1.txt > /dev/null 2>&1 & 
                        
                        while [[ $(date -u +%s) -le $endtime ]]; do ss --no-header -eipn dst 192.168.3.1 or dst 192.168.4.1 | ts '%.s' | tee -a $i-$j-$b-trial-ss.txt > /dev/null;sleep 0.1; done
                        sleep $t;
                        cat $i-$j-$b-trial-iperf.txt | grep "sender" | awk '{print $8}' | tr '\n' ',' >> results-trialc.txt

                        echo "reno" | tr '\n' ',' >>  results-trialc.txt

                        cat $i-$j-$b-trial-iperf1.txt | grep "sender" | awk '{print $8}' | tr '\n' ',' >> results-trialc.txt
                        sed '/fd=3/d' $i-$j-$b-trial-ss.txt > $i-$j-$b-trial-ss1.txt
                        cat $i-$j-$b-trial-ss1.txt | sed -e ':a; /<->$/ { N; s/<->\n//; ba; }'  | grep "iperf3" > $i-$j-$b-trial-ss-processed.txt
                        echo "latency" | tr '\n' ',' >>  results-trialc.txt
                        cat $i-$j-$b-trial-ss-processed.txt | grep -oP '\brtt:.*?(\s|$)' |  awk -F '[:,]' '{print $2}' | tr -d ' '  | cut -d '/' -f 1   > srtt-$i-$j-$b-trial.txt
                        awk '{ total += $1; count++ } END { print total/count }' srtt-$i-$j-$b-trial.txt >> results-trialc.txt;
                        

                done
        done
done 









t=10

runtime="10 seconds"

for j in {1..3}  
do  
        for b in {0.001,0.005,0.01,0.05,0.1,0.5,0,4,6,8,10}
        do
               

                for i in {reno,pcc,balia,bbr,lia,olia,wvegas,pcc_loss}
                do
                        echo "$j" | tr '\n' ',' >>  results-triald.txt
                        echo "$i" | tr '\n' ',' >>  results-triald.txt
                        echo "$b" | tr '\n' ',' >>  results-triald.txt

                        endtime=$(date -ud "$runtime" +%s)

                        sudo ssh router1 -f sudo tc qdisc replace dev enp6s0f1 parent 1:3 handle 3: bfifo limit 384000

                        sudo ssh router2 -f sudo tc qdisc replace dev enp6s0f0 parent 1:3 handle 3: bfifo limit 384000

                        sudo ssh -p 22 emulator1 -f sudo tc qdisc replace dev enp6s0f1 root netem loss $b% delay 30ms limit 60000

                        sudo ssh -p 22 emulator2 -f sudo tc qdisc replace dev enp6s0f1 root netem loss $b% delay 30ms limit 60000

                        sudo sysctl -w net.ipv4.tcp_congestion_control=$i

                        if [[ "$i" == "bbr" ]]
                        then
                                sudo sysctl -w net.mptcp.mptcp_scheduler=default_pacing
                        elif [[ "$i" == "pcc" ]]
                        then
                                sudo sysctl -w net.mptcp.mptcp_scheduler=default_pacing
                        elif [[ "$i" == "pcc_loss" ]]
                        then
                                sudo sysctl -w net.mptcp.mptcp_scheduler=default_pacing
                        else
                                sudo sysctl -w net.mptcp.mptcp_scheduler=default
                        fi



                        iperf3 -f m -c 192.168.3.1 -p 5101 -C "$i" -P 1 -i 0.1 -t $t | ts '%.s' | tee $i-$j-$b-trial-iperf.txt > /dev/null & rm -f $i-$j-$b-trial-ss.txt 2>&1 & 
                        iperf3 -f m -c 192.168.3.1 -p 5102 -C "reno" -P 2 -i 0.1 -t $t | ts '%.s' | tee $i-$j-$b-trial-iperf1.txt > /dev/null 2>&1 & 
                        
                        while [[ $(date -u +%s) -le $endtime ]]; do ss --no-header -eipn dst 192.168.3.1 or dst 192.168.4.1 | ts '%.s' | tee -a $i-$j-$b-trial-ss.txt > /dev/null;sleep 0.1; done
                        sleep $t;
                        cat $i-$j-$b-trial-iperf.txt | grep "sender" | awk '{print $8}' | tr '\n' ',' >> results-triald.txt

                        echo "reno" | tr '\n' ',' >>  results-triald.txt

                        cat $i-$j-$b-trial-iperf1.txt | grep "sender" | awk '{print $8}' | tr '\n' ',' >> results-triald.txt
                        sed '/fd=3/d' $i-$j-$b-trial-ss.txt > $i-$j-$b-trial-ss1.txt
                        cat $i-$j-$b-trial-ss1.txt | sed -e ':a; /<->$/ { N; s/<->\n//; ba; }'  | grep "iperf3" > $i-$j-$b-trial-ss-processed.txt
                        echo "latency" | tr '\n' ',' >>  results-triald.txt
                        cat $i-$j-$b-trial-ss-processed.txt | grep -oP '\brtt:.*?(\s|$)' |  awk -F '[:,]' '{print $2}' | tr -d ' '  | cut -d '/' -f 1   > srtt-$i-$j-$b-trial.txt
                        awk '{ total += $1; count++ } END { print total/count }' srtt-$i-$j-$b-trial.txt >> results-triald.txt;
                        

                done
        done
done 





t=10

runtime="10 seconds"

for j in {1..3}  
do  
        for b in {0.001,0.005,0.01,0.05,0.1,0.5,0,4,6,8,10}
        do
               

                for i in {reno,pcc,balia,bbr,lia,olia,wvegas,pcc_loss}
                do
                        echo "$j" | tr '\n' ',' >>  results-triale.txt
                        echo "$i" | tr '\n' ',' >>  results-triale.txt
                        echo "$b" | tr '\n' ',' >>  results-triale.txt

                        endtime=$(date -ud "$runtime" +%s)

                        sudo ssh router1 -f sudo tc qdisc replace dev enp6s0f1 parent 1:3 handle 3: bfifo limit 384000

                        sudo ssh router2 -f sudo tc qdisc replace dev enp6s0f0 parent 1:3 handle 3: bfifo limit 384000

                        sudo ssh -p 22 emulator1 -f sudo tc qdisc replace dev enp6s0f1 root netem loss $b% delay 30ms limit 60000

                        sudo ssh -p 22 emulator2 -f sudo tc qdisc replace dev enp6s0f1 root netem loss $b% delay 30ms limit 60000

                        sudo sysctl -w net.ipv4.tcp_congestion_control=$i

                        if [[ "$i" == "bbr" ]]
                        then
                                sudo sysctl -w net.mptcp.mptcp_scheduler=default_pacing
                        elif [[ "$i" == "pcc" ]]
                        then
                                sudo sysctl -w net.mptcp.mptcp_scheduler=default_pacing
                        elif [[ "$i" == "pcc_loss" ]]
                        then
                                sudo sysctl -w net.mptcp.mptcp_scheduler=default_pacing
                        else
                                sudo sysctl -w net.mptcp.mptcp_scheduler=default
                        fi



                        iperf3 -f m -c 192.168.3.1 -p 5101 -C "$i" -P 2 -i 0.1 -t $t | ts '%.s' | tee $i-$j-$b-trial-iperf.txt > /dev/null & rm -f $i-$j-$b-trial-ss.txt 2>&1 & 
                        
                        while [[ $(date -u +%s) -le $endtime ]]; do ss --no-header -eipn dst 192.168.3.1 or dst 192.168.4.1 | ts '%.s' | tee -a $i-$j-$b-trial-ss.txt > /dev/null;sleep 0.1; done
                        sleep $t;
                        cat $i-$j-$b-trial-iperf.txt | grep "sender" | awk '{print $8}' | tr '\n' ',' >> results-triale.txt
                        sed '/fd=3/d' $i-$j-$b-trial-ss.txt > $i-$j-$b-trial-ss1.txt
                        cat $i-$j-$b-trial-ss1.txt | sed -e ':a; /<->$/ { N; s/<->\n//; ba; }'  | grep "iperf3" > $i-$j-$b-trial-ss-processed.txt
                        echo "latency" | tr '\n' ',' >>  results-triale.txt
                        cat $i-$j-$b-trial-ss-processed.txt | grep -oP '\brtt:.*?(\s|$)' |  awk -F '[:,]' '{print $2}' | tr -d ' '  | cut -d '/' -f 1   > srtt-$i-$j-$b-trial.txt
                        awk '{ total += $1; count++ } END { print total/count }' srtt-$i-$j-$b-trial.txt >> results-triale.txt;
                        

                done
        done
done 
