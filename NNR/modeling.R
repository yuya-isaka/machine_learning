install.packages("ggplot2")
library(ggplot2)
library(dplyr)
set.seed(1234)
X <- rnorm(450,40,10)
N_pre <- c(200,150,100)
N_st1 <- c(rep(30,5),20,20,10)
N_st2 <- c(30,30,20,20,20,20,10)
N_st3 <- c(50,30,20)
N_st <- c(N_st1,N_st2,N_st3)
pref <- rep(1:3,times=N_pre)
school <- rep(1:length(N_st), times=N_st)

a0 <- 50
b0 <- 20

a_pre <- rnorm(3, mean=a0, sd=100) 
b_pre <- rnorm(3, mean=b0, sd=10)

a <- rnorm(length(N_st), mean=rep(a_pre,c(8,7,3)), sd=50) 
b <- rnorm(length(N_st), mean=rep(b_pre,c(8,7,3)), sd=5)

data_frame(X=X,school = as.factor(school),pref = as.factor(pref),
           a=a[school],b=b[school]) 
  mutate(Y=rnorm(450,a+b*X,30))  
  select(X,pref,school,Y)-> st_df


# グラフ化
#都道府県ごと
st_df  
  ggplot(aes(x=X,y=Y,col=pref))+ 
  geom_point(alpha = 0.5)+
  geom_smooth(method = "lm",se=F)

#都道府県1の学校ごと
st_df  
  filter(pref=="1")
  ggplot(aes(x=X,y=Y,col=school))+ 
  geom_point()+
  geom_smooth(method = "lm",se=F)