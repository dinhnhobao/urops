library(imager)
library(stringr)

# Script that, given a directory of pictures and the date entitling their
# labels .csv file, plots the pictures and their labels side-by-side.
# Before running this script, set the working directory as the location of this
# file.

pictures_directory <- "pictures_to_label/"
labels_directory <- "label_csvs/"
date <- "2018-07-24"

file_names <- list.files(pictures_directory, pattern="*.jpg")
labels <- read.csv(paste(labels_directory, date, ".csv", sep=""))
par(mfrow=c(1, 2))
for (index in 1:length(file_names)) {
  image <- load.image(paste(pictures_directory, file_names[index], sep=""))
  plot(image,
       main=file_names[index])
  plot(c(-1, 3),
       c(-1, 3),
       ann=FALSE,
       bty="n",
       type="n",
       xaxt="n",
       yaxt="n")
  text(x=-1,
       y=1,
       labels$label[labels$date_id == 
                      str_sub(file_names[index],
                              1,
                              length(file_names[index]) - 6)], 
       cex = 1.6, col = "black")
  text(x=0.75,
       y=1,
       paste(index, "/", length(file_names), sep=""),
       cex = 1.6, col = "black")
  readline("")
  if (index == length(file_names)) {
    par(mfrow=c(1, 1))
    plot(load.image("white.png"))
  }
}
