library(imager)

directory <- "../dummy_spot_pictures/"
date <- readline("Enter date: ")
file_names <- list.files(directory, pattern="*.jpg")

all_date_ids <- NULL
all_labels <- NULL
for (index in 1:length(file_names)) {
  image <- load.image(paste(directory, file_names[index], sep=""))
  plot(image)
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
write.csv(result_df, paste(date, ".csv", sep=""), row.names=FALSE)

