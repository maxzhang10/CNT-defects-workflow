#!/bin/bash

echo "workflow_config.json : $(find . -name 'workflow_config.json' | wc -l)"
echo "traj.dump            : $(find . -name 'traj.dump' | wc -l)"
echo "Transmission.png     : $(find . -name 'Transmission.png' | wc -l)"
echo "log                  : $(find . -name 'log' | wc -l)"
echo "self_energy          : $(find . -name 'self_energy' | wc -l)"
find . -type d -name 'self_energy'
