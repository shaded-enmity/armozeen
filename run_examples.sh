#!/usr/bin/bash

function log() {
  echo "[$(date --rfc-3339=seconds)]: $*"
}

OK='\e[32m✔\e[0m'
FAIL='\e[31m✖\[0m'

for EG in examples/*; do
   ./armozeen-cli $EG 1>/dev/null
   R=$FAIL
   if [[ $? = 0 ]]; then 
           R=$OK
   fi
   log $(printf "%-.60s [%s]\n" "Case \"${EG}\" ......................................................................................................................." $(echo -e $R))
done
