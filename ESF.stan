data {
  int<lower=0> n; // number of observations
  int<lower=0> dev; // number of eigenvectors
  int<lower=0> dx; // number of covariates
  vector[n] y; // outcome variable
  matrix[n, dev] EV; // eigenvectors
  matrix[n, dx] x; // covariates
  real<lower=0> scale_global;  // horseshoe parameters
  real<lower=0> slab_scale;
  real<lower=0> slab_df; 
  vector[3] alpha_prior; // other priors
  row_vector[dx] beta_prior[3];
  vector[3] sigma_prior;
}

transformed data {
  // use the QR decomposition on the matrix of covariates
  matrix[n, dx] Q_ast;
  matrix[dx, dx] R_ast;
  matrix[dx, dx] R_inverse;
  Q_ast = qr_Q(x)[, 1:dx] * sqrt(n - 1);
  R_ast = qr_R(x)[1:dx, ] / sqrt(n - 1);
  R_inverse = inverse(R_ast);
}

parameters {
  real<lower=1> nu;
  real<lower=0> sigma;
  real Intercept;
  vector[dx] beta_tilde;
  real<lower=0> aux1_global;
  real<lower=0> aux2_global;
  vector<lower=0>[dev] aux1_local;
  vector<lower=0>[dev] aux2_local;
  real<lower=0> caux;
  vector[dev] z;
}

transformed parameters {
  // RHS prior on the EV matrix 
  vector[dx] beta;
  real <lower=0> tau;
  vector<lower=0>[dev] lambda;
  vector<lower=0>[dev] lambda_tilde;
  vector[dev] beta_ev;
  real <lower=0> c;
  vector[n] fitted;
  tau = aux1_global * sqrt(aux2_global) * scale_global * sigma;
  lambda = aux1_local .* sqrt(aux2_local);
  c = slab_scale * sqrt(caux);
  lambda_tilde = sqrt(c^2 * square(lambda) ./ (c^2 + tau^2*square(lambda)));
  beta_ev = z .* lambda_tilde * tau;
  fitted = Intercept + Q_ast * beta_tilde + EV * beta_ev;
  beta = R_inverse * beta_tilde;
}

model {
  z ~ normal(0, 1);
  aux1_local ~ normal(0, 1);
  aux2_local ~ inv_gamma(0.5, 0.5);
  aux1_global ~ normal(0, 1);
  aux2_global ~ inv_gamma(0.5, 0.5);
  caux ~ inv_gamma(0.5*slab_df, 0.5*slab_df);
  nu ~ gamma(2, 0.1);
  Intercept ~ student_t(alpha_prior[1], alpha_prior[2], alpha_prior[3]);
  beta ~ student_t(beta_prior[1], beta_prior[2], beta_prior[3]);
  sigma ~ student_t(sigma_prior[1], sigma_prior[2], sigma_prior[3]); 
  y ~ student_t(nu, fitted, sigma); 
}

generated quantities {
  vector[n] residual;
  vector[n] esf;
  vector[n] yrep;
  for (i in 1:n) {
    residual[i] = y[i] - fitted[i];
    esf[i] = EV[i] * beta_ev;
    yrep[i] = student_t_rng(nu, fitted[i], sigma);
  }
}

