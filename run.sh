#!/bin/bash

source activate planto3d && //
export FLASK_APP=app.py && //
xdg-open http://localhost:5000 && //
flask run && //
source deactivate planto3d 