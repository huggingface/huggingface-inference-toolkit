#!/bin/bash
set -e -o pipefail -u

export ENV_FILEPATH=$(mktemp)

trap "rm -f ${ENV_FILEPATH}" EXIT

touch $ENV_FILEPATH

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

${SCRIPT_DIR}/inf2_env.py $@

source $ENV_FILEPATH

rm -f $ENV_FILEPATH

exec ${SCRIPT_DIR}/entrypoint.sh $@