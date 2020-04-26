rand <- round(rnorm(50, mean = 0, sd = 1), digits = 1)

hist(rand,freq = F, main = "", xlab = "", xlim = c(-3,3))
points(rand,rep(0,50),pch = "l")
curve(dnorm(x,0,1),add = T,col = "red")
legend(x = "topright",lty = 1 , 
       col = "red",legend = "True Density",cex = 0.8)

estimated <- density(rand)  # カーネル密度を求めた

hist(rand,freq = F,main="",xlab = "",xlim = c(-3,3))  # 用意したrand変数のヒストグラムを書いた
lines(estimated, col = "blue")  # 推定値の線を引いた
curve(dnorm(x,0,1),add = T,col = "red") # その上から本当の正規分布の関数の線をひいた
legend(x = "topright",lty = 1, col = c("red","blue"),legend = c("True Density","Estimated Density"),cex = 0.8)
head(data.frame(x = estimated$x, y = estimated$y))