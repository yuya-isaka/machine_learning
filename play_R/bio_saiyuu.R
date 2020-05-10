log_like <- function(mu) {
  70*log(mu) + 30*log(1-mu)
}

mu <- seq(0,1,0.1)

plot(mu, log_like(mu), type="l",
     xlab="mu", ylab="liklihood", main="",
     col="red", lwd=2, xaxt="n")

abline(v=0.7)
axis(1,mu,mu)