library(imager)
library(stringr)

directory <- "label_pictures/"
date <- readline("Enter date: ")
time_started <- readline("Enter start time: ")
file_names <- list.files(directory, pattern="*.jpg")
labels <- read.csv(paste("label_csvs/", date, "_", time_started, ".csv", sep=""))
par(mfrow=c(1, 2))
for (index in 1:length(file_names)) {
  image <- load.image(paste(directory, file_names[index], sep=""))
  plot(image,
       main=file_names[index])
  plot(c(0, 1),
       c(0, 1),
       ann=FALSE,
       bty="n",
       type="n",
       xaxt="n",
       yaxt="n")
  text(x=0,
       y=0.5,
       labels$label[labels$date_id == 
                      str_sub(file_names[index],
                              1,
                              length(file_names[index]) - 6)], 
       cex = 1.6, col = "black")
  readline("")
}
par(mfrow=c(1, 1))
