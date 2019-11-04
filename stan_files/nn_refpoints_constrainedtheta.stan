// NN wth some reference points tied down:

data 
{
    int<lower=0> Nobs;
    int<lower=0> in_dim;
    int<lower=0> out_dim;
    int<lower=0> hidden_dim;
    int<lower=0> n_ref;
    matrix[out_dim,Nobs-n_ref] X;
    matrix[out_dim,n_ref] X_ref;
    matrix[in_dim,n_ref] Z_ref;
}
parameters 
{
    vector[in_dim] Z[Nobs-n_ref];
    unit_vector[hidden_dim] weights_1[in_dim];
    vector[hidden_dim] bias_1;
    matrix[hidden_dim, out_dim] weights_2;
    vector[out_dim] bias_2;
    real<lower=0> error_sigma2;
    real<lower=0> prior_sigma2;
}

transformed parameters
{   
    matrix[in_dim,Nobs-n_ref] Z_tr; 
    matrix[hidden_dim,in_dim] weights_1_tr; 
    for (i in 1:(Nobs-n_ref)) 
    {
        Z_tr[:,i] = Z[i];
    }
    for (i in 1:in_dim)
    {
        weights_1_tr[:,i] = weights_1[i];
    }
}

model 
{  
    // Latent distribution:
    for (i in 1:(Nobs-n_ref))
    {
        Z[i] ~ normal(0,1);
    }    
    
    // Priors:
    error_sigma2 ~ cauchy(0,5);
    prior_sigma2 ~ cauchy(0,5);
    for (i in 1:in_dim)
    {
        weights_1[i] ~ normal(0,prior_sigma2);;
    }
    for (h in 1:hidden_dim)
    {
        weights_2[h,:] ~ normal(0,prior_sigma2);
    }
    bias_1 ~ normal(0,prior_sigma2);
    bias_2 ~ normal(0,prior_sigma2);

    // Likelihood:
    for (n in 1:(Nobs-n_ref))
    {
       X[:,n] ~ normal(weights_2'*tanh(weights_1_tr*Z_tr[:,n] + bias_1) + bias_2, error_sigma2);
    }
    for (n in 1:n_ref)
    {
        X_ref[:,n] ~ normal(weights_2'*tanh(weights_1_tr*Z_ref[:,n] + bias_1) + bias_2, error_sigma2);
    }
}

