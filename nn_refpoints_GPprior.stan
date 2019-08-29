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
    matrix[Nobs,Nobs] pw_dist;
}

parameters 
{
    matrix[Nobs-n_ref,in_dim] X;
    matrix[in_dim, hidden_dim_1] weights_1;
    row_vector[hidden_dim_1] bias_1;
    matrix[hidden_dim_1, out_dim] weights_2;
    row_vector[out_dim] bias_2;
    real<lower=0> error_sigma2;
    real<lower=0> prior_sigma2;
}

transformed parameters
{
    vector[Nobs-n_ref] X_transformed;
    X_transformed = X[:,1];
}

model 
{
    matrix[Nobs,Nobs] Sigma;
    matrix[n_ref,n_ref] Sigma11;
    matrix[n_ref,Nobs-n_ref] Sigma12;
    matrix[Nobs-n_ref,n_ref] Sigma21;
    matrix[Nobs-n_ref,Nobs-n_ref] Sigma22;
    
    vector[Nobs-n_ref] X_mu;
    matrix[Nobs-n_ref,Nobs-n_ref] X_cov;
    
    for (i in 1:Nobs) 
    {
        for (j in 1:Nobs) 
        {
            Sigma[i,j] = exp(-pw_dist[i,j]^2);
        }
    }
    
    Sigma11 = Sigma[1:n_ref,1:n_ref];
    Sigma21 = Sigma[1:n_ref,(n_ref+1):Nobs];
    Sigma12 = Sigma[(n_ref+1):Nobs,1:n_ref];
    Sigma22 = Sigma[(n_ref+1):Nobs,(n_ref+1):Nobs];
    
    // Priors:
    error_sigma2 ~ cauchy(0,5);
    prior_sigma2 ~ gamma(5,5);
    for (h in 1:hidden_dim_1)
    {
        weights_1[:,h] ~ normal(0,prior_sigma2);
        weights_2[h,:] ~ normal(0,prior_sigma2);
    }
    
    X_mu = (Sigma21*inverse_spd(Sigma11)*X_ref)[:,1];
    X_cov = Sigma22-Sigma21*inverse_spd(Sigma11)*Sigma12;
    
    X_transformed ~ multi_normal(X_mu, X_cov);
    
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

