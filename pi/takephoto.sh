#!/bin/bash
DATE=$(date +"%Y-%m-%d-%H%M")
raspistill --nopreview --exposure auto --awb auto --saturation -25 --metering average --quality 100 --exif none --timeout 10 --output /media/pi/pictures/$DATE.jpg
