#!/bin/bash
python3 baseline1.py 1 > record.txt
ret=$?
#
#  returns > 127 are a SIGNAL
#
if [ $ret -gt 127 ]; then
        sig=$((ret - 128))
        echo "Got SIGNAL $sig"
        if [ $sig -eq $(kill -l SIGKILL) ]; then
                echo "process was killed with SIGKILL"
                dmesg > ./dmesg-kill.log
        fi
fi