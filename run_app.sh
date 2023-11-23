# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 02:15:15 2023

@author: yamil
"""
chmod +x run_app.sh
gunicorn -w 4 -b 0.0.0.0:5000 temp:app