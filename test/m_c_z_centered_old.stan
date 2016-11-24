functions {
  vector nfw(vector r, real m200, real c200, real zl, real zs, real dl, real ds, real dls, vector mu, vector tau) {
    real c;
    real Mpc;
    real pc;
    real cMpc;
    real Msun;
    real G;
    real GMsunMpc;
    real h;
    real H0;
    real Om;
    real Ol;
    real Ez;
    real rho_crit;
    real sigma_crit;
    real rho200;
    real r200;
    real rscale;
    real rhos;

    vector[dims(r)[1]] shear;

    c = 2.99792e5; // km / s
    Mpc = 3.0856e19; // km
    pc = Mpc / 1e6 * 1e3; // m
    cMpc = c/Mpc; // Mpc / s

    Msun = 1.989e30; // kg
    

    G = 6.672e-11; // m^3 / kg / s^2
    GMsunMpc = G / (pc*pc*pc) * 1e-18 * Msun; // Mpc^3 / MSun / s^2

    h = 0.7;
    H0 = h * 100.0;
    Om = 0.28;
    Ol = 0.72;


    if(m200==0 || c200==0){
        reject("m=",log10(m200), "c=",c200, " mu=", mu);
    }

    Ez = sqrt(Om*(1.0 + zl)^3 + Ol);

    rho_crit = 3.0*((H0*Ez/Mpc)^2)/(8.0*pi()*GMsunMpc);

    sigma_crit = cMpc^2*ds/(4.0*GMsunMpc*pi()*dl*dls);

    rho200 = 200.0*rho_crit;
    r200 = (3.0*m200/(4.0*pi()*rho200))^(1.0/3.0);

    if (is_nan(r200) || is_inf(r200)) {
        reject("r200=", r200);
    }


    rscale = r200 / c200;
    rhos = (rho200*c200^3)/(3.0*(log1p(c200) - c200/(1.0+c200)));

    if (is_nan(rhos) || is_inf(rhos) || is_nan(rscale) || is_inf(rscale)) {
        reject("rhos=", rhos," rs=",rscale);
    }

//if(rscale < 0.05 || rscale >10){
//    print("rs=",rscale, " m=",log10(m200), " c=",log10(c200)," mu=", mu/log(10), " t=",tau);
//}

    for (i in 1:dims(r)[1]) {
      real term;
      real sigma;
      real mean_sigma;
      real ridrs;
      real x;

      ridrs = r[i]/rscale;

        if (ridrs<0.98) {
            term = 2.0*atanh(sqrt((1.0-ridrs)/(1.0+ridrs)))/sqrt(1.0-ridrs*ridrs);
        }else if (ridrs>1.02){
            term = 2.0*atan(sqrt((ridrs-1.0)/(1.0+ridrs)))/sqrt(ridrs*ridrs-1.0);
        }else{
            x = ridrs-1.0;
            term = 1.0 + x*(-2.0/3.0 + x*(7.0/15.0 + x*(-12.0/35.0 + x*(166.0/630.0))));
        }
      sigma = 2*rscale*rhos*(1.0-term)/(ridrs*ridrs - 1.0);
      mean_sigma = 4.0*rscale*rhos*(term + log(ridrs/2.0))/(ridrs*ridrs);
      shear[i] = fabs((mean_sigma-sigma)/(sigma_crit-sigma)); //reduced shear

        if (is_nan(shear[i]) || is_inf(shear[i])) {
        reject("x=", ridrs," M=", log10(m200), " c=", c200, " z=",zl);
        }

    }

    return shear;
  }
}

data {
  int Nc;
  int Nr;
  vector[Nr] rs[Nc];
  vector[Nr] kappas[Nc];
  vector[Nr] sigma_kappas[Nc];
  real zls[Nc];
  real dz[Nc];
  real zss[Nc];
  real dls[Nc];
  real dss[Nc];
  real dlss[Nc];
  vector[3] mu0; //prior on mean posterior
  vector[3] sigma0;
}

parameters {
  vector[3] mu;
  cholesky_factor_corr[3] L_Omega;	//prior cholesky factor of correlation matrix
  vector<lower=0>[3] tau; //prior scale
  vector[3] log_params[Nc];
}

transformed parameters {
    vector[3] cl_params[Nc];
    vector[Nr] model_kappas[Nc];
    cholesky_factor_cov[3] L; //cholesky factor of covariance matrix

    L = diag_pre_multiply(tau, L_Omega);

    for (i in 1:Nc) {
        cl_params[i] = exp(log_params[i]);
        model_kappas[i] = nfw(rs[i], cl_params[i][1], cl_params[i][2], zls[i], zss[i], dls[i], dss[i], dlss[i], mu, tau);
    }
}

model {
mu ~ normal(mu0, sigma0); // or use diagonal(sigma) but this uses posterior twice
tau ~ normal(0,1);

L_Omega ~ lkj_corr_cholesky(10);

log_params ~ multi_normal_cholesky(mu,L);

for (i in 1:Nc) {
kappas[i] ~ normal(model_kappas[i], sigma_kappas[i]);
zls[i] ~ normal(cl_params[i][3] - 1, dz[i]);
}
}

generated quantities{
cov_matrix[3] sigma; //global covariance matrix
sigma = L * L';
}


