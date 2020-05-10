

data {
  int<lower=0> N;
  real Y[N];
  real study[N];
}

parameters  {
  real a;
  real b;
  real <lower = 0>sigma;
}

transformed parameters{
  real mu[N];
  for(n in 1:N){
    mu[n] = a+b*study[n];
  }
}

model {
  for (n in 1:N){
    Y[n]~normal(mu[n],sigma);
  }
}