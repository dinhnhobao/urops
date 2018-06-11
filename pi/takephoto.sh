#!/bin/bash
DATE=$(date +"%Y-%m-%d-%H%M")
raspistill --nopreview --exposure auto --awb auto --metering average --quality 100 --exif none --timeout 10 --output /home/pi/stills/$DATE.jpg
