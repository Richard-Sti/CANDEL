data {
// Cepheids:
    int<lower=1> num_cepheids;                      // number of Cepheids
    vector[num_cepheids] mag_cepheid;               // observed apparent magnitudes
    vector[num_cepheids] e_mag_cepheid;             // sqrt of the covariance of cepheid magnitudes
    vector[num_cepheids] logP;                      // observed log10(period)
    vector[num_cepheids] OH;                        // Metallicity

// Hosts:
    int<lower=1>      num_hosts;                    // Number of Hosts
    int<lower=1>      num_anchors;                  // Number of anchors
    vector[num_hosts] mag_SN_unique_Cepheid_host;   // Magnitudes of SNe
    vector[num_hosts] e_mag_SN_unique_Cepheid_host; // Magnitudes of SNe
    vector[num_hosts] czcmb_cepheid_host;           // cz for hosts
    vector[num_hosts] e_czcmb_cepheid_host;         // cz errors
    matrix[num_cepheids,num_hosts+num_anchors] L_Cepheid_host_dist; // Matrix of cepheid hosts

// Anchors:
    real mu_N4258_anchor;
    real e_mu_N4258_anchor;
    real mu_LMC_anchor;
    real e_mu_LMC_anchor;
    
// Misc:
	real sigma_grnd;
	real c;                      // speed of light
}

transformed data {
	int num_hosts_anchors = num_hosts + num_anchors;
}

parameters {
    real<lower=0.1, upper=1.0> h;           // Hubble constant
    vector<lower=-15, upper=6>[num_hosts_anchors] log_r; // latent log distance

    real<lower=-7, upper=-5>   M_w;         // PL zero point
    real<lower=-6, upper=0>    b_w;         // PL logP slope
    real<lower=-2, upper=2>    Z_w;         // PL OH slope
    real<lower=-22, upper=-18> M_SN0;       // SN magnitude (or location parameter for distribution)
	real<lower=1, upper = 500> sigma_v;     // Extra variance in PVs
}

transformed parameters {
    vector[num_hosts_anchors] r = exp(log_r);
    vector[num_hosts_anchors] cz = 100 * h * r;
    vector[num_hosts_anchors] DL = r .* (1 + cz / c);

	vector[num_cepheids]      cepheid_DL = L_Cepheid_host_dist * DL; 
	vector[num_hosts_anchors] mu_host = 5 * log10(DL) + 25;
    vector[num_cepheids]      mu = 5 * log10(cepheid_DL) + 25;

    vector[num_cepheids]      m_pred = mu + M_w + b_w * logP + Z_w * OH;
    vector[num_hosts]         m_SN_pred = M_SN0 + mu_host[1:num_hosts]; 
}

model {  
    // Term for the distance prior
    target += 3 * sum(log_r);

    M_w ~ normal(-5.8946, 0.0239); // Combination of HST and GAIA

    // Redshift likelihood
    for (i in 1:num_hosts)    czcmb_cepheid_host[i] ~ normal(cz[i], sqrt(e_czcmb_cepheid_host[i]^2 + sigma_v^2));

    // dZP should only be applied to the LMC component of the magnitudes
    for (i in 1:num_cepheids) mag_cepheid[i] ~ normal(m_pred[i], e_mag_cepheid[i]);
    for (i in 1:num_hosts)    mag_SN_unique_Cepheid_host[i] ~ normal(m_SN_pred[i], e_mag_SN_unique_Cepheid_host[i]);

    mu_N4258_anchor ~ normal(mu_host[36],e_mu_N4258_anchor);
    mu_LMC_anchor ~ normal(mu_host[37],e_mu_LMC_anchor);

// Selection: 
    target += 1.38 * num_hosts * M_SN0;
}
