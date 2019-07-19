// NN wth some reference points tied down:

data 
{
    int<lower=0> Nobs;
    int<lower=0> in_dim;
    int<lower=0> out_dim;
    int<lower=0> hidden_dim_1;
    int<lower=0> n_ref;
    matrix[Nobs-n_ref,out_dim] y;
    matrix[n_ref,out_dim] y_ref;
    matrix[n_ref,in_dim] X_ref;
}

parameters 
{
    matrix<lower=-1,upper=1>[Nobs-n_ref,in_dim] X;
    matrix[in_dim, hidden_dim_1] weights_1;
    row_vector[hidden_dim_1] bias_1;
    matrix[hidden_dim_1, out_dim] weights_2;
    row_vector[out_dim] bias_2;
    real<lower=0> error_sigma2;
    real<lower=0> prior_sigma2;
}

model 
{
    // Priors:
    error_sigma2 ~ cauchy(0,5);
    prior_sigma2 ~ gamma(1,1);
    for (h in 1:hidden_dim_1)
    {
        weights_1[:,h] ~ normal(0,prior_sigma2);
        weights_2[h,:] ~ normal(0,prior_sigma2);
    }
    
    for (n in 1:(Nobs-n_ref))
    {
        X[n,:] ~ uniform(-1,1);
    }
    bias_1 ~ normal(0,prior_sigma2);
    bias_2 ~ normal(0,prior_sigma2);

    // Likelihood:
    for (n in 1:(Nobs-n_ref))
    {
       y[n] ~ normal(tanh(X[n,:]*weights_1 + bias_1)*weights_2 + bias_2, error_sigma2);
    }
    for (n in 1:n_ref)
    {
        y_ref[n] ~ normal(tanh(X_ref[n,:]*weights_1 + bias_1)*weights_2 + bias_2, error_sigma2);
    }
}

