#! /bin/bash

if ! command -v python3.10 &> /dev/null
then
    if command -v apt-get &> /dev/null
    then
        sudo apt-get install python3.10
    elif command -v dnf &> /dev/null
    then
        sudo dnf install python3.10 libxcrypt-compat
    else
        echo "Python 3.10 not found. Please install Python 3.10"
        exit
    fi
fi

python3.10 -m venv env_bridge
source env_bridge/bin/activate
pip install -e .
