#!/bin/bash

#FOLDER_BASE=/home/cc/GPU-Sched/src/runtime/driver/results-2020.08.12-bk2
FOLDER_BASE=/home/ubuntu/GPU-Sched/src/runtime/driver/results-2020.08.13




FOLDERS=(
    ${FOLDER_BASE}/16jobs
    ${FOLDER_BASE}/32jobs
)


# create a "goterror" folder and move files that have "got error" into it
for FOLDER in ${FOLDERS[@]}; do
    FOLDER_GOTERROR=${FOLDER}_goterror
    mkdir ${FOLDER_GOTERROR}
    pushd ${FOLDER}
    mv $(grep "got error" * | awk -F ':' '{print $1}' | sort | uniq) ${FOLDER_GOTERROR}
    popd
done

# move the supporting scheduler files for whichever experiments had an error
for FOLDER in ${FOLDERS[@]}; do
    FOLDER_GOTERROR=${FOLDER}_goterror

    for FILE in `ls ${FOLDER}_goterror`; do
        FILE_BASE=$(basename -- "$FILE")
        EXTENSION="${FILE_BASE##*.}"
        FILE_BASE="${FILE_BASE%.*}"
        #echo ${FOLDER}/${FILE_BASE}
        #echo $EXTENSION
        mv ${FOLDER}/${FILE_BASE}.sched-stats ${FOLDER_GOTERROR}
        mv ${FOLDER}/${FILE_BASE}.sched-log ${FOLDER_GOTERROR}
    done 

done

echo
echo
echo "Dumping error counts..."
echo

for FOLDER in ${FOLDERS[@]}; do
    FOLDER_GOTERROR=${FOLDER}_goterror
    pushd ${FOLDER_GOTERROR} > /dev/null
    for FILE in `ls *.workloader-log`; do
        printf ${FILE}" "
        grep "got error" ${FILE} | wc -l
    done
    popd > /dev/null
    echo
done



echo
echo "Script complete"
echo "Note: if there is a missing sched-stats file, it could be because that job was killed (especially if the mgb process count was high for the file that seems to be missing). That's fine if it's not there, and it's not a problem with this script."
echo
