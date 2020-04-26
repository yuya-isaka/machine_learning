set.seed(0)
x1 <- c(rnorm(300, mean=0, sd=1), runif(10, min=-5, max=5)) 
x2 <- c(rnorm(300, mean=0, sd=1), runif(10, min=-5, max=5))

plot(x1, x2)

