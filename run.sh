#!/bin/sh


echo "***********************************"
time=$(date "+%Y-%m-%d-%H-%M")
echo "${time}"
################
EPOCH=20
START_NEUR=16
K_SIZE=3
BATCH_SIZE=8
VALID_CASE_NO=-1
NETWORK_NO=0  #0:4block, 1:3block, 5:5blocks.
LEARNING_RATE=1e-5
SOURCE=1 #1:MRI, 0:CT
################

SCRIPT_DIR=$(cd $(dirname $0); pwd) # get upper dir,/home/chen/u-net
UPPER_DIR=$(cd $(dirname $0); cd ..;pwd) # get current dir where this sh script is saved, /home/chen/u-net/unet
TIME_DIR="/src/model/${time}/"
mkdir -m 777 ${TIME_DIR}
echo "make dir:${TIME_DIR}"
cp -r ${SCRIPT_DIR} "${TIME_DIR}/" # copy /home/chen/u-net/unet(script dir) to time dir.
echo "EPOCH:${EPOCH} \
      START_NEUR:${START_NEUR} \
      K_SIZE:${K_SIZE} \
      BATCH_SIZE:${BATCH_SIZE} \
      VALID_CASE_NO:${VALID_CASE_NO} \
      SOURCE:${SOURCE} \
      NETWORK_NO:${NETWORK_NO} \
      LEARNING_RATE:${LEARNING_RATE}">${TIME_DIR}/args
echo "run ${SCRIPT_DIR}/mytrain2.py"
python3 "${TIME_DIR}/unet/mytrain2.py" -t ${time} \
                                    -e ${EPOCH} \
                                    -n ${START_NEUR} \
                                    -k ${K_SIZE} \
                                    -b ${BATCH_SIZE} \
                                    -v ${VALID_CASE_NO} \
                                    -nn ${NETWORK_NO} \
                                    -s ${SOURCE} \
                                    -p ${TIME_DIR} \
                                    -lr ${LEARNING_RATE}
