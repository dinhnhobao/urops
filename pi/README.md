`takephoto.sh` is the code executed by the Raspberry Pi
used in the data collection detailed in `../writeup/main.tex`,
called every five minutes using `crontab`:

`crontab -e append:  */5 * * * * /home/pi/takephoto.sh`
