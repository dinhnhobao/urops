# Script that removes all cropped pictures with given IDs from supplied
# directory.
# Before running this script, set the working directory as the location of this
# file.

ids_to_remove <- c(1, 22, 35)
pictures_directory <- "pictures_to_label/"

for (i in ids_to_remove) {
  file_names <- paste(pictures_directory,
                      list.files(pictures_directory,
                                 pattern=paste("*_", i, ".jpg", sep="")),
                                 sep="")
  for (index in 1:length(file_names)) {
    system(paste("rm", file_names[index]))
  }
}
