#!/bin/bash

source activate planto3d && //
export FLASK_APP=app.py && //
flask run && //
source deactivate planto3d 