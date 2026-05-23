#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cassert>

namespace PhysConst {
    constexpr double m_e   = 9.1093837015e-31;
    constexpr double m_p   = 1.67262192369e-27;
    constexpr double a0    = 5.29177210903e-11;
    constexpr double pi    = 3.14159265358979323846;
}

static double lgamma_impl(double x) { return std::lgamma(x); }

static double assoc_legendre(int l, int m, double x) {
    bool neg_m = false;
    if (m < 0) { m = -m; neg_m = true; }

    double pmm = 1.0;
    if (m > 0) {
        double fact = 1.0;
        double sx   = std::sqrt((1.0 - x) * (1.0 + x));
        for (int i = 1; i <= m; ++i) {
            pmm  *= -(2 * i - 1) * sx;
        }
    }
    if (l == m) {
        if (neg_m) {
            double ratio = std::exp(lgamma_impl(l - m + 1) - lgamma_impl(l + m + 1));
            if (m % 2 != 0) ratio = -ratio;
            return ratio * pmm;
        }
        return pmm;
    }
    double pmm1 = x * (2 * m + 1) * pmm;
    if (l == m + 1) {
        if (neg_m) {
            double ratio = std::exp(lgamma_impl(l - m + 1) - lgamma_impl(l + m + 1));
            if (m % 2 != 0) ratio = -ratio;
            return ratio * pmm1;
        }
        return pmm1;
    }
    double pll = 0.0;
    for (int ll = m + 2; ll <= l; ++ll) {
        pll  = ((2 * ll - 1) * x * pmm1 - (ll + m - 1) * pmm) / (ll - m);
        pmm  = pmm1;
        pmm1 = pll;
    }
    if (neg_m) {
        double ratio = std::exp(lgamma_impl(l - m + 1) - lgamma_impl(l + m + 1));
        if (m % 2 != 0) ratio = -ratio;
        return ratio * pll;
    }
    return pll;
}

static std::complex<double> sph_harm(int l, int m, double theta, double phi) {
    int abs_m = std::abs(m);
    double log_norm = 0.5 * (std::log(2 * l + 1) - std::log(4 * PhysConst::pi)
                     + lgamma_impl(l - abs_m + 1) - lgamma_impl(l + abs_m + 1));
    double norm = std::exp(log_norm);
    double Plm  = assoc_legendre(l, abs_m, std::cos(theta));
    std::complex<double> phase = std::exp(std::complex<double>(0.0, m * phi));
    std::complex<double> Y = norm * Plm * phase;
    if (m < 0) Y *= (((abs_m % 2) == 0) ? 1.0 : -1.0);
    return Y;
}

static double gen_laguerre(int n, double alpha, double x) {
    if (n == 0) return 1.0;
    double L0 = 1.0;
    double L1 = 1.0 + alpha - x;
    for (int k = 1; k < n; ++k) {
        double Lnew = ((2 * k + 1 + alpha - x) * L1 - (k + alpha) * L0) / (k + 1);
        L0 = L1;
        L1 = Lnew;
    }
    return (n == 1) ? L1 : L1;
}

double reduced_electron_nucleus_mass(int Z, double M = -1.0) {
    if (M < 0.0) {
        if (Z == 1) M = PhysConst::m_p;
        else throw std::invalid_argument("Nuclear mass M must be provided for Z > 1");
    }
    return (PhysConst::m_e * M) / (PhysConst::m_e + M);
}

double reduced_bohr_radius(double mu) {
    return PhysConst::a0 * (PhysConst::m_e / mu);
}

std::vector<double> radial_wavefunction_Rnl(
        int n, int l,
        const std::vector<double>& r,
        int Z = 1, bool use_reduced_mass = true, double M = -1.0)
{
    if (!(n >= 1 && l >= 0 && l <= n - 1))
        throw std::invalid_argument("Quantum numbers must satisfy n >= 1, 0 <= l <= n-1");

    double mu   = use_reduced_mass ? reduced_electron_nucleus_mass(Z, M) : PhysConst::m_e;
    double a_mu = reduced_bohr_radius(mu);

    double log_pref = 1.5 * std::log(2.0 * Z / (n * a_mu))
                    + 0.5 * (lgamma_impl(n - l) - (std::log(2.0 * n) + lgamma_impl(n + l + 1)));
    double pref = std::exp(log_pref);

    std::vector<double> R(r.size());
    for (std::size_t i = 0; i < r.size(); ++i) {
        double rho = 2.0 * Z * r[i] / (n * a_mu);
        double L   = gen_laguerre(n - l - 1, 2 * l + 1, rho);
        R[i] = pref * std::exp(-rho / 2.0) * std::pow(rho, l) * L;
    }
    return R;
}

std::vector<std::complex<double>> spherical_harmonic_Ylm(
        int l, int m,
        const std::vector<double>& theta,
        const std::vector<double>& phi)
{
    if (!(l >= 0 && m >= -l && m <= l))
        throw std::invalid_argument("Quantum numbers must satisfy l >= 0, -l <= m <= l");
    assert(theta.size() == phi.size());

    std::vector<std::complex<double>> Y(theta.size());
    for (std::size_t i = 0; i < theta.size(); ++i)
        Y[i] = sph_harm(l, m, theta[i], phi[i]);
    return Y;
}

struct PsiSliceResult {
    std::vector<double>               Xg;
    std::vector<double>               Zg;
    std::vector<std::complex<double>> psi;
    double                            a_mu;
    int                               grid_pts;
};

PsiSliceResult compute_psi_xz_slice(
        int n, int l, int m,
        int Z = 1, bool use_rm = true, double M = -1.0,
        double extent = 20.0, int grid_pts = 600,
        int phi_mode = 0, double phi_val = 0.0)
{
    if (!(n >= 1 && l >= 0 && l <= n - 1 && m >= -l && m <= l))
        throw std::invalid_argument("Quantum numbers must satisfy n>=1, 0<=l<=n-1, -l<=m<=l");

    double mu   = use_rm ? reduced_electron_nucleus_mass(Z, M) : PhysConst::m_e;
    double a_mu = reduced_bohr_radius(mu);
    double r_max = extent * a_mu;

    std::vector<double> axis(grid_pts);
    for (int i = 0; i < grid_pts; ++i)
        axis[i] = -r_max + 2.0 * r_max * i / (grid_pts - 1);

    int N2 = grid_pts * grid_pts;
    std::vector<double> Xg(N2), Zg(N2), r_flat(N2), theta_flat(N2), phi_flat(N2);

    for (int iz = 0; iz < grid_pts; ++iz) {
        for (int ix = 0; ix < grid_pts; ++ix) {
            int idx     = iz * grid_pts + ix;
            Xg[idx]     = axis[ix];
            Zg[idx]     = axis[iz];
            double x    = axis[ix], z = axis[iz];
            double rr   = std::hypot(x, z);
            r_flat[idx] = rr;
            double cos_th = (rr > 0.0) ? z / rr : 1.0;
            cos_th = std::max(-1.0, std::min(1.0, cos_th));
            theta_flat[idx] = std::acos(cos_th);
            phi_flat[idx]   = (phi_mode == 0) ? (x >= 0.0 ? 0.0 : PhysConst::pi) : phi_val;
        }
    }

    auto R = radial_wavefunction_Rnl(n, l, r_flat, Z, use_rm, M);
    auto Y = spherical_harmonic_Ylm(l, m, theta_flat, phi_flat);

    std::vector<std::complex<double>> psi(N2);
    for (int i = 0; i < N2; ++i) psi[i] = R[i] * Y[i];

    return {Xg, Zg, psi, a_mu, grid_pts};
}

std::vector<double> compute_probability_density(const std::vector<std::complex<double>>& psi) {
    std::vector<double> P(psi.size());
    for (std::size_t i = 0; i < psi.size(); ++i) P[i] = std::norm(psi[i]);
    return P;
}

std::vector<double> compute_radial_probability_distribution(
        const std::vector<double>& R, const std::vector<double>& r)
{
    assert(R.size() == r.size());
    std::vector<double> Pr(R.size());
    for (std::size_t i = 0; i < R.size(); ++i) Pr[i] = r[i] * r[i] * R[i] * R[i];
    return Pr;
}

void write_psi_slice_csv(const std::string& fname, const PsiSliceResult& res) {
    std::ofstream f(fname);
    if (!f) throw std::runtime_error("Cannot open " + fname);

    int gp  = res.grid_pts;
    auto P  = compute_probability_density(res.psi);

    f << "x,z,re_psi,im_psi,prob_density\n";
    f << std::scientific;
    f.precision(10);
    for (int i = 0; i < gp * gp; ++i) {
        f << res.Xg[i] << ','
          << res.Zg[i] << ','
          << res.psi[i].real() << ','
          << res.psi[i].imag() << ','
          << P[i] << '\n';
    }
}

void write_radial_csv(const std::string& fname,
                      const std::vector<double>& r,
                      const std::vector<double>& R,
                      const std::vector<double>& Pr,
                      double a_mu)
{
    std::ofstream f(fname);
    if (!f) throw std::runtime_error("Cannot open " + fname);
    f << "r_over_a_mu,R_nl,radial_prob\n";
    f << std::scientific; f.precision(10);
    for (std::size_t i = 0; i < r.size(); ++i)
        f << r[i] / a_mu << ',' << R[i] << ',' << Pr[i] << '\n';
}

int main() {
    struct OrbitalSpec { int n, l, m; };
    std::vector<OrbitalSpec> orbitals = {
        {1, 0,  0},
        {2, 0,  0},
        {2, 1,  0},
        {2, 1,  1},
        {3, 0,  0},
        {3, 1,  0},
        {3, 2,  0},
        {3, 2,  1},
        {4, 0,  0},
        {4, 3,  0},
    };

    const int Z        = 1;
    const int grid_pts = 400;

    for (auto& orb : orbitals) {
        std::string tag = "n" + std::to_string(orb.n)
                        + "l" + std::to_string(orb.l)
                        + "m" + std::to_string(orb.m);
        std::cout << "Computing " << tag << " ..." << std::flush;

        double extent = 4.0 * (orb.n * orb.n) + 8.0;
        auto res = compute_psi_xz_slice(orb.n, orb.l, orb.m,
                                        Z, true, -1.0,
                                        extent, grid_pts, 0);
        write_psi_slice_csv("psi_" + tag + ".csv", res);

        int Nr    = 2000;
        double r_max = extent * res.a_mu;
        std::vector<double> r_arr(Nr), R_arr, Pr_arr;
        for (int i = 0; i < Nr; ++i)
            r_arr[i] = r_max * (i + 1) / Nr;
        R_arr  = radial_wavefunction_Rnl(orb.n, orb.l, r_arr, Z, true);
        Pr_arr = compute_radial_probability_distribution(R_arr, r_arr);
        write_radial_csv("radial_" + tag + ".csv", r_arr, R_arr, Pr_arr, res.a_mu);

        std::cout << " done (a_mu=" << res.a_mu << " m)\n";
    }

    std::cout << "\nAll CSV files written. Run hydrogen_plot.py to visualize.\n";
    return 0;
}
