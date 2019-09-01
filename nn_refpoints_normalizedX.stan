// NN wth some reference points tied down:

data 
{
    int<lower=0> Nobs;
    int<lower=0> in_dim;
    int<lower=0> out_dim;
    int<lower=0> hidden_dim;
    int<lower=0> n_ref;
    matrix[Nobs-n_ref,out_dim] y;
    matrix[n_ref,out_dim] y_ref;
    matrix[n_ref,in_dim] X_ref;
}
parameters 
{
    unit_vector[Nobs-n_ref] X_normalized[in_dim];
    matrix[in_dim, hidden_dim] weights_1;
    row_vector[hidden_dim] bias_1;
    matrix[hidden_dim, out_dim] weights_2;
    row_vector[out_dim] bias_2;
    real<lower=0> error_sigma2;
    real<lower=0> prior_sigma2;
}

transformed parameters
{
    matrix[Nobs-n_ref,in_dim] X;
    for (i in 1:in_dim) 
    {
        X[:,i] = X_normalized[i];
    }
}

model 
{  
    // Latent distribution:
    for (i in 1:in_dim)
    {
        X[:,i] ~ normal(0,1);
    }
    
    // Priors:
    error_sigma2 ~ cauchy(0,5);
    prior_sigma2 ~ cauchy(0,5);
    for (h in 1:hidden_dim)
    {
        weights_1[:,h] ~ normal(0,prior_sigma2);
        weights_2[h,:] ~ normal(0,prior_sigma2);
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

