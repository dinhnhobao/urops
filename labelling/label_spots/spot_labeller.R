library(imager)

directory <- "label_pictures/"
date <- readline("Enter date: ")
time_started <- readline("Enter start time: ")
file_names <- list.files(directory, pattern="*.jpg")

all_date_ids <- NULL
all_labels <- NULL
for (index in 1:length(file_names)) {
  image <- load.image(paste(directory, file_names[index], sep=""))
  plot(image,
       main=file_names[index])
  all_date_ids <- c(all_date_ids,
                     substr(file_names[index],
                            start=1,
                            stop=nchar(file_names[index]) - 4)
  )
  all_labels <- c(all_labels, readline("Enter label: "))
  plot(load.image("white.png"))
}

result_df <- NULL
result_df$date_id <- all_date_ids
result_df$label <- all_labels
rownames(result_df) <- NULL
write.csv(result_df,
          paste("label_csvs/", date, "_", time_started, ".csv", sep=""),
          row.names=FALSE)
