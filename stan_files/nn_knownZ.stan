// NN wth some reference points tied down:

data 
{
    int<lower=0> Nobs;
    int<lower=0> in_dim;
    int<lower=0> out_dim;
    int<lower=0> hidden_dim;
    matrix[in_dim,Nobs] Z;
    matrix[out_dim,Nobs] X;
}
parameters 
{
    matrix[hidden_dim,in_dim] weights_1;
    vector[hidden_dim] bias_1;
    matrix[out_dim,hidden_dim] weights_2;
    vector[out_dim] bias_2;
    real<lower=0> error_sigma2;
    real<lower=0> prior_sigma2;
}

model 
{  
    // Priors:
    error_sigma2 ~ cauchy(0,5);
    prior_sigma2 ~ cauchy(0,5);
    for (i in 1:in_dim)
    {
        weights_1[:,i] ~ normal(0,prior_sigma2);;
    }
    for (h in 1:hidden_dim)
    {
        weights_2[:,h] ~ normal(0,prior_sigma2);
    }
    bias_1 ~ normal(0,prior_sigma2);
    bias_2 ~ normal(0,prior_sigma2);

    // Likelihood:
    for (n in 1:Nobs)
    {
       X[:,n] ~ normal(weights_2*tanh(weights_1*Z[:,n] + bias_1) + bias_2, error_sigma2);
    }
}

