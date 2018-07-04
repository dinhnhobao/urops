library(imager)

directory <- "label_pictures/"
date <- "2018-06-14"
time_started <- "1335"
file_names <- list.files(directory, pattern="*.jpg")

all_date_ids <- NULL
all_labels <- NULL
for (index in 1:length(file_names)) {
  image <- load.image(paste(directory, file_names[index], sep=""))
  plot(image,
       main=paste(file_names[index],
                  paste(index, "/", length(file_names), sep="")))
  all_date_ids <- c(all_date_ids,
                     substr(file_names[index],
                            start=1,
                            stop=nchar(file_names[index]) - 4)
  )
  input <- readline("Enter label: ")
  all_labels <- c(all_labels, ifelse(input == "", 0, 1))
  plot(load.image("white.png"))
}

result_df <- NULL
result_df$date_id <- all_date_ids
result_df$label <- all_labels
rownames(result_df) <- NULL
write.csv(result_df,
          paste("label_csvs/", date, "_", time_started, ".csv", sep=""),
          row.names=FALSE)
