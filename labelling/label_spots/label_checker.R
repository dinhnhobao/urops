library(imager)
library(stringr)

directory <- "label_pictures/"
date <- "2018-07-09"
file_names <- list.files(directory, pattern="*.jpg")
labels <- read.csv(paste("label_csvs/", date, ".csv", sep=""))
par(mfrow=c(1, 2))
for (index in 1:length(file_names)) {
  image <- load.image(paste(directory, file_names[index], sep=""))
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
}
par(mfrow=c(1, 1))

# Spot remover.
directory <- "label_pictures/"
file_names <- paste("label_pictures/",
                    list.files(directory, pattern="*_5.jpg"),
                               sep="")
for (index in 1:length(file_names)) {
  system(paste("rm", file_names[index]))
}

