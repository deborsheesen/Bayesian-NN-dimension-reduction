// NN wth some reference points tied down:

data 
{
    int<lower=0> Nobs;
    int<lower=0> in_dim;
    int<lower=0> out_dim;
    int<lower=0> hidden_dim_1;
    matrix[Nobs,out_dim] y;
}

parameters 
{
    matrix[Nobs,in_dim] X;
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
    for (n in 1:Nobs)
    {
        X[n,:] ~ normal(0,prior_sigma2);
    }
    bias_1 ~ normal(0,prior_sigma2);
    bias_2 ~ normal(0,prior_sigma2);

    // Likelihood:
    for (n in 1:Nobs)
    {
       y[n] ~ normal(tanh(X[n,:]*weights_1 + bias_1)*weights_2 + bias_2, error_sigma2);
    }
}

