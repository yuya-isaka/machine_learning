data  {
  int N; //試行回数
  int m; //表が出た回数
  int a; //ベータ分布のパラメータ
  int b;
}

parameters {
  real<lower=0,upper=1> mu;
}

model {
  mu~beta(a,b);
  m~binomial(N,mu);
}