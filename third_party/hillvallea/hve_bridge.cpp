// C interface to HillVallEA (S.C. Maree et al., github.com/SCMaree/HillVallEA, GPL-3.0) so that it
// can be driven from Python through ctypes. The objective is supplied as a callback (minimization).
// Licensed under GPL-3.0, like the code it links to.
#include <memory>
#include "HillVallEA/hillvallea.hpp"
#include "HillVallEA/fitness.h"

namespace hillvallea
{
typedef double (*objective_t)(const double *, int);

class callback_problem_t : public fitness_t
{
public:
  objective_t objective;
  vec_t lower, upper;

  callback_problem_t(int dim, const double *lo, const double *hi, int budget, objective_t f) : objective(f)
  {
    number_of_parameters = dim;
    maximum_number_of_evaluations = budget;
    lower.resize(dim);
    upper.resize(dim);
    for (int i = 0; i < dim; ++i) { lower[i] = lo[i]; upper[i] = hi[i]; }
  }
  void get_param_bounds(vec_t &lo, vec_t &hi) const { lo = lower; hi = upper; }
  void define_problem_evaluation(solution_t &sol)
  {
    sol.f = objective(sol.param.data(), (int)number_of_parameters);
    sol.penalty = 0.0;
  }
  std::string name() const { return "callback"; }
};
}

// Runs HillVallEA with the authors' default settings (the quick constructor: AMaLGaM-Univariate local
// optimizer, no time limit). `seeds` (n_seeds x dim, row-major) are added to the first sample of the run
// (requires seed.patch). Writes up to `max_out` elites to out_x / out_f and returns their number.
extern "C" int hve_run(int dim, const double *lo, const double *hi, int budget, int random_seed,
                       hillvallea::objective_t objective, const double *seeds, int n_seeds,
                       double *out_x, double *out_f, int max_out)
{
  using namespace hillvallea;
  std::shared_ptr<callback_problem_t> problem = std::make_shared<callback_problem_t>(dim, lo, hi, budget, objective);
  fitness_pt fitness = problem;
  hillvallea_t opt(fitness, dim, problem->lower, problem->upper, budget, random_seed);
  for (int s = 0; s < n_seeds; ++s)
  {
    vec_t p(dim);
    for (int i = 0; i < dim; ++i) p[i] = seeds[s * dim + i];
    opt.seed_params.push_back(p);
  }
  opt.run();
  int n = (int)std::min((size_t)max_out, opt.elitist_archive.size());
  for (int k = 0; k < n; ++k)
  {
    for (int i = 0; i < dim; ++i) out_x[k * dim + i] = opt.elitist_archive[k]->param[i];
    out_f[k] = opt.elitist_archive[k]->f;
  }
  return n;
}
