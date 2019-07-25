// Neural network wth GP prior for latent X with covariance matrix depending on Y:

data 
{
    int<lower=0> Nobs;
    int<lower=0> in_dim;
    int<lower=0> out_dim;
    int<lower=0> hidden_dim_1;
    //real y[Nobs,out_dim];
    //matrix[Nobs,out_dim] y;
    vector[out_dim] y[Nobs];
}

parameters 
{
    //row_vector[Nobs] X;
    //row_vector[in_dim] X[Nobs];
    matrix[Nobs,in_dim] X;
    
    matrix[in_dim, hidden_dim_1] weights_1;
    row_vector[hidden_dim_1] bias_1;
    matrix[hidden_dim_1, out_dim] weights_2;
    row_vector[out_dim] bias_2;
    real<lower=0> error_sigma2;
    real<lower=0> theta_sigma2;
    real<lower=0> rho;
    real<lower=0> alpha;
    real<lower=0> sigma2;
}

transformed parameters
{
    vector[Nobs] X_transformed;
    X_transformed = X[:,1];
}

model 
{
    //vector[in_dim] mu[Nobs];
    vector[Nobs] mu;
    matrix[Nobs, Nobs] L_K;
    matrix[Nobs, Nobs] K;
    
    // Hyperpriors:
    error_sigma2 ~ cauchy(0,5);
    theta_sigma2 ~ gamma(1,1);
    rho ~ inv_gamma(5, 5);
    alpha ~ std_normal();
    sigma2 ~ cauchy(0,5);
    
    // Prior for weights and bias of neural network:
    for (h in 1:hidden_dim_1)
    {
        weights_1[:,h] ~ normal(0,theta_sigma2);
        weights_2[h,:] ~ normal(0,theta_sigma2);
    }
    bias_1 ~ normal(0,theta_sigma2);
    bias_2 ~ normal(0,theta_sigma2);
    
    // Prior for latent X:
    mu = rep_vector(0,Nobs);
    K = cov_exp_quad(y, alpha, rho);
    for (n in 1:Nobs)
    {
        K[n,n] = K[n,n] + sigma2;
        //mu[n] = rep_vector(0,in_dim);
    }
    L_K = cholesky_decompose(K);
    X_transformed ~ multi_normal_cholesky(mu, L_K);

    // Likelihood:
    for (n in 1:Nobs)
    {
       y[n] ~ normal(tanh(X[n,:]*weights_1 + bias_1)*weights_2 + bias_2, error_sigma2);
    }
}

