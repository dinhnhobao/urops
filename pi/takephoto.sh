#!/bin/bash
DATE=$(date +"%Y-%m-%d_%H%M")
raspistill --nopreview --exposure auto --awb auto --metering average --quality 100 --exif none --timeout 10 --output /home/pi/stills/$DATE.jpg
