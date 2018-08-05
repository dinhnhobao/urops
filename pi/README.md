### Configuration code for the Raspberry Pi used for data collection
`crontab -e append:  */5 * * * * /home/pi/takephoto.sh`:
saves images to /home/pi/stills/ every five minutes.
