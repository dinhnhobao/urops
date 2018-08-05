library(imager)

# Script that facilitates the labelling of a given set of cropped pictures.
# Labels are stored in "label_csvs/{date}.csv" with the variables "date_id"
# and "label".
# Before running this script, set the working directory as the location of this
# file.

pictures_directory <- "pictures_to_label/"
csvs_directory <- "label_csvs/"
date <- "2018-07-24"

file_names <- list.files(pictures_directory,
                         pattern="*.jpg")
date_id <- NULL
label <- NULL
for (index in 1:length(file_names)) {
  image <- load.image(paste(pictures_directory,
                            file_names[index], sep=""))
  plot(image,
       main=paste(file_names[index],
                  paste(index, "/", length(file_names), sep="")))
  date_id <- c(date_id,
               substr(file_names[index],
                      start=1,
                      stop=nchar(file_names[index]) - 4)
  )
  label <- c(label,
             readline("Enter label: "))
  plot(load.image("white.png"))
}

result_df <- data.frame(date_id, label)
rownames(result_df) <- NULL
write.csv(result_df,
          paste(csvs_directory, date, ".csv", sep=""),
          row.names=FALSE)
