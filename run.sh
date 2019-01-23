#!/bin/bash

source activate planto3d

python test.py && python regen3d.py

source deactivate planto3d