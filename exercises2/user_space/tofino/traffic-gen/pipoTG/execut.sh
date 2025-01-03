
#kill any old process running
#ps -elf | grep switch | grep  -v grep | awk '{print $4}' | xargs kill -9
#load module if not loaded
#bf_kdrv_mod_load $SDE_INSTALL

#Compile PIPO-TG p4 code
#/$SDE/../tools/p4_build.sh files/pipoTG.p4
#./files/build.sh

# Start the switch
/$SDE/run_switchd.sh -p pipoTG --arch tf2 &
sleep 20


#Config PORTS
/$SDE/run_bfshell.sh -f files/portConfig.txt 
#$SDE/run_bfshell.sh -f config-100g.txt


#Install RULES
nohup python3 files/tableEntries.py > log &

#rate-show
/$SDE/run_bfshell.sh -f files/view


#kill -9 bf_switchd


