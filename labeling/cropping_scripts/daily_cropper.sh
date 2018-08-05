source /home/e0032276/tmp/bin/activate
python crop_all_spots.py > todo.sh
chmod 777 todo.sh
source ./todo.sh
rm todo.sh
deactivate
