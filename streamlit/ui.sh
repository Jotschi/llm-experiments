#!/bin/bash

set -o nounset
set -o errexit

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

. $SCRIPT_DIR/venv/bin/activate

streamlit run streamlit-ui.py $1
