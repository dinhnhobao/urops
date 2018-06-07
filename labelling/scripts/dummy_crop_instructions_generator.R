set.seed(1)

df <- NULL
df$label = 1:54
df$angle = round(runif(54, 0, 360))
df$x_one = round(runif(54, 0, 1000))
df$x_two = df$x_one + 128
df$y_one = round(runif(54, 0, 1000))
df$y_two = df$y_one + 128

write.csv(df, "crop_instructions.csv")
