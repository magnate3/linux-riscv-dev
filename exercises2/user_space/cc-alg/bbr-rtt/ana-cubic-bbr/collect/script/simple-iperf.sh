t=10
#limit
b=5
i=0
j=0
echo "$j" | tr '\n' ',' >>  results-triale.txt
echo "$i" | tr '\n' ',' >>  results-triale.txt
echo "$b" | tr '\n' ',' >>  results-triale.txt

runtime="10 seconds"
endtime=$(date -ud "$runtime" +%s)

touch $i-$j-$b-trial-ss.txt
iperf3 -f m -c 10.22.116.221  -p 5201 -C cubic  -P 1 -i 0.1 -t $t | ts '%.s' | tee $i-$j-$b-trial-iperf.txt > /dev/null & rm -f $i-$j-$b-trial-ss.txt 2>&1 & 
#iperf3 -f m -c 10.22.116.221  -p 5202 -C "reno" -P 1 -i 0.1 -t $t | ts '%.s' | tee $i-$j-$b-trial-iperf1.txt > /dev/null 2>&1 & 

while [[ $(date -u +%s) -le $endtime ]]; do ss --no-header -eipn dst 10.22.116.221  or dst 192.168.4.1 | ts '%.s' | tee -a $i-$j-$b-trial-ss.txt > /dev/null;sleep 0.1; done

cat $i-$j-$b-trial-iperf.txt | grep "sender" | awk '{print $8}' | tr '\n' ',' >> results-trialc.txt

echo "reno" | tr '\n' ',' >>  results-trialc.txt

#cat $i-$j-$b-trial-iperf1.txt | grep "sender" | awk '{print $8}' | tr '\n' ',' >> results-trialc.txt
sed '/fd=3/d' $i-$j-$b-trial-ss.txt > $i-$j-$b-trial-ss1.txt
cat $i-$j-$b-trial-ss1.txt | sed -e ':a; /<->$/ { N; s/<->\n//; ba; }'  | grep "iperf3" > $i-$j-$b-trial-ss-processed.txt
echo "latency" | tr '\n' ',' >>  results-trialc.txt
cat $i-$j-$b-trial-ss-processed.txt | grep -oP '\brtt:.*?(\s|$)' |  awk -F '[:,]' '{print $2}' | tr -d ' '  | cut -d '/' -f 1   > srtt-$i-$j-$b-trial.txt
awk '{ total += $1; count++ } END { print total/count }' srtt-$i-$j-$b-trial.txt >> results-trialc.txt;



