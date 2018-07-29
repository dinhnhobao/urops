spots_to_remove <- 9:20
for (i in spots_to_remove) {
  directory <- "label_pictures/"
  file_names <- paste(directory,
                      list.files(directory, pattern=paste("*_", i, ".jpg", sep="")),
                                 sep="")
  for (index in 1:length(file_names)) {
    system(paste("rm", file_names[index]))
  }
}