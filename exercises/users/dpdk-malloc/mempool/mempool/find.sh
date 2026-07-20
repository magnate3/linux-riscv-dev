file_paths=($(find ./lib -name '*.h' |  sed 's#.*/##'))
for file in ${file_paths[@]}
do
    echo ${file}
    find ./arm64-armv8a-include -name ${file} | xargs rm -rf
    #file2=($(find ./arm64-armv8a-include -name ${file} |  sed 's#.*/##'))
    if test -z "${file2}"  
    then  
         echo  "not find:"${file}
    else    
         echo  "find:"${file}
    fi   
done 
