// Benchmark of GN_ISRES variants (issue #590). For each problem and seed:
// number of evaluations until f <= f* + tol at a feasible point, or failure
// within maxeval. Output: CSV lines variant,problem,seed,result,evals,f.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>
#include <nlopt.hpp>

static const double PI = 3.14159265358979323846;

typedef std::function<double(const std::vector<double> &)> Fn;

struct Problem {
  std::string name;
  std::vector<double> lb, ub, x0;
  Fn f;
  std::vector<Fn> ineq;  // g(x) <= 0
  std::vector<Fn> eq;    // h(x) = 0
  double fopt, tol, eqtol;
  std::vector<double> xopt;  // known minimizer, for the self-check
};

static double call(unsigned n, const double *x, double *grad, void *data)
{
  (void)grad;
  const Fn &fn = *static_cast<const Fn *>(data);
  return fn(std::vector<double>(x, x + n));
}

static std::vector<Problem> problems()
{
  std::vector<Problem> ps;
  {  // NLopt tutorial problem, as in test/t_tutorial.cxx with ISRES
    Problem p;
    p.name = "tutorial";
    p.lb = {1e-6, 1e-6}; p.ub = {10, 10}; p.x0 = {1.234, 5.678};
    p.f = [](const std::vector<double> &x) { return std::sqrt(x[1]); };
    p.ineq.push_back([](const std::vector<double> &x) { double t = 2 * x[0]; return t * t * t - x[1]; });
    p.ineq.push_back([](const std::vector<double> &x) { double t = -x[0] + 1; return t * t * t - x[1]; });
    p.fopt = 0.544331053951817355154952; p.tol = 1e-3; p.eqtol = 0;
    ps.push_back(p);
  }
  {  // g06 (Floudas and Pardalos)
    Problem p;
    p.name = "g06";
    p.lb = {13, 0}; p.ub = {100, 100}; p.x0 = {56.5, 50};
    p.f = [](const std::vector<double> &x) { return std::pow(x[0] - 10, 3) + std::pow(x[1] - 20, 3); };
    p.ineq.push_back([](const std::vector<double> &x) { return -std::pow(x[0] - 5, 2) - std::pow(x[1] - 5, 2) + 100; });
    p.ineq.push_back([](const std::vector<double> &x) { return std::pow(x[0] - 6, 2) + std::pow(x[1] - 5, 2) - 82.81; });
    p.fopt = -6961.81387558015; p.tol = 0.1; p.eqtol = 0;
    ps.push_back(p);
  }
  {  // g08
    Problem p;
    p.name = "g08";
    p.lb = {1e-3, 1e-3}; p.ub = {10, 10}; p.x0 = {5, 5};
    p.f = [](const std::vector<double> &x) {
      return -std::pow(std::sin(2 * PI * x[0]), 3) * std::sin(2 * PI * x[1]) / (std::pow(x[0], 3) * (x[0] + x[1]));
    };
    p.ineq.push_back([](const std::vector<double> &x) { return x[0] * x[0] - x[1] + 1; });
    p.ineq.push_back([](const std::vector<double> &x) { return 1 - x[0] + std::pow(x[1] - 4, 2); });
    p.fopt = -0.0958250414180359; p.tol = 1e-6; p.eqtol = 0;
    ps.push_back(p);
  }
  {  // g11 (equality constraint)
    Problem p;
    p.name = "g11";
    p.lb = {-1, -1}; p.ub = {1, 1}; p.x0 = {0.5, 0.5};
    p.f = [](const std::vector<double> &x) { return x[0] * x[0] + std::pow(x[1] - 1, 2); };
    p.eq.push_back([](const std::vector<double> &x) { return x[1] - x[0] * x[0]; });
    p.fopt = 0.75; p.tol = 1e-3; p.eqtol = 1e-4;
    ps.push_back(p);
  }
  {  // Rosenbrock, 2 dims, no constraints
    Problem p;
    p.name = "rosenbrock";
    p.lb = {-2, -2}; p.ub = {2, 2}; p.x0 = {-1.2, 1};
    p.f = [](const std::vector<double> &x) { return 100 * std::pow(x[1] - x[0] * x[0], 2) + std::pow(1 - x[0], 2); };
    p.fopt = 0; p.tol = 1e-4; p.eqtol = 0;
    ps.push_back(p);
  }
  {  // Shekel m = 10, 4 dims, no constraints
    Problem p;
    p.name = "shekel10";
    p.lb = {0, 0, 0, 0}; p.ub = {10, 10, 10, 10}; p.x0 = {5, 5, 5, 5};
    p.f = [](const std::vector<double> &x) {
      static const double A[10][4] = {{4,4,4,4},{1,1,1,1},{8,8,8,8},{6,6,6,6},{3,7,3,7},
                                      {2,9,2,9},{5,5,3,3},{8,1,8,1},{6,2,6,2},{7,3.6,7,3.6}};
      static const double c[10] = {.1,.2,.2,.4,.4,.6,.3,.7,.5,.5};
      double s = 0;
      for (int i = 0; i < 10; ++i) {
        double d = 0;
        for (int j = 0; j < 4; ++j) d += (x[j] - A[i][j]) * (x[j] - A[i][j]);
        s -= 1 / (d + c[i]);
      }
      return s;
    };
    p.fopt = -10.536409816692; p.tol = 1e-3; p.eqtol = 0;
    ps.push_back(p);
  }
  {  // g01
    Problem p;
    p.name = "g01";
    p.lb = std::vector<double>(13, 0.0); p.ub = std::vector<double>(13, 1.0);
    p.ub[9] = p.ub[10] = p.ub[11] = 100;
    p.x0 = std::vector<double>(13, 0.5);
    p.f = [](const std::vector<double> &x) {
      double s = 0;
      for (int i = 0; i < 4; ++i) s += 5 * x[i] - 5 * x[i] * x[i];
      for (int i = 4; i < 13; ++i) s -= x[i];
      return s;
    };
    p.ineq.push_back([](const std::vector<double> &x) { return 2*x[0] + 2*x[1] + x[9] + x[10] - 10; });
    p.ineq.push_back([](const std::vector<double> &x) { return 2*x[0] + 2*x[2] + x[9] + x[11] - 10; });
    p.ineq.push_back([](const std::vector<double> &x) { return 2*x[1] + 2*x[2] + x[10] + x[11] - 10; });
    p.ineq.push_back([](const std::vector<double> &x) { return -8*x[0] + x[9]; });
    p.ineq.push_back([](const std::vector<double> &x) { return -8*x[1] + x[10]; });
    p.ineq.push_back([](const std::vector<double> &x) { return -8*x[2] + x[11]; });
    p.ineq.push_back([](const std::vector<double> &x) { return -2*x[3] - x[4] + x[9]; });
    p.ineq.push_back([](const std::vector<double> &x) { return -2*x[5] - x[6] + x[10]; });
    p.ineq.push_back([](const std::vector<double> &x) { return -2*x[7] - x[8] + x[11]; });
    p.xopt = {1,1,1,1,1,1,1,1,1,3,3,3,1};
    p.fopt = -15; p.tol = 1.5e-3; p.eqtol = 0;
    ps.push_back(p);
  }
  {  // g04
    Problem p;
    p.name = "g04";
    p.lb = {78, 33, 27, 27, 27}; p.ub = {102, 45, 45, 45, 45}; p.x0 = {90, 39, 36, 36, 36};
    p.f = [](const std::vector<double> &x) {
      return 5.3578547*x[2]*x[2] + 0.8356891*x[0]*x[4] + 37.293239*x[0] - 40792.141;
    };
    auto u = [](const std::vector<double> &x) { return 85.334407 + 0.0056858*x[1]*x[4] + 0.0006262*x[0]*x[3] - 0.0022053*x[2]*x[4]; };
    auto v = [](const std::vector<double> &x) { return 80.51249 + 0.0071317*x[1]*x[4] + 0.0029955*x[0]*x[1] + 0.0021813*x[2]*x[2]; };
    auto w = [](const std::vector<double> &x) { return 9.300961 + 0.0047026*x[2]*x[4] + 0.0012547*x[0]*x[2] + 0.0019085*x[2]*x[3]; };
    p.ineq.push_back([u](const std::vector<double> &x) { return u(x) - 92; });
    p.ineq.push_back([u](const std::vector<double> &x) { return -u(x); });
    p.ineq.push_back([v](const std::vector<double> &x) { return v(x) - 110; });
    p.ineq.push_back([v](const std::vector<double> &x) { return -v(x) + 90; });
    p.ineq.push_back([w](const std::vector<double> &x) { return w(x) - 25; });
    p.ineq.push_back([w](const std::vector<double> &x) { return -w(x) + 20; });
    p.xopt = {78, 33, 29.9952560256815985, 45, 36.7758129057882073};
    p.fopt = -30665.538671783317; p.tol = 3.0; p.eqtol = 0;
    ps.push_back(p);
  }
  {  // g07
    Problem p;
    p.name = "g07";
    p.lb = std::vector<double>(10, -10.0); p.ub = std::vector<double>(10, 10.0); p.x0 = std::vector<double>(10, 0.0);
    p.f = [](const std::vector<double> &x) {
      return x[0]*x[0] + x[1]*x[1] + x[0]*x[1] - 14*x[0] - 16*x[1] + std::pow(x[2]-10, 2)
        + 4*std::pow(x[3]-5, 2) + std::pow(x[4]-3, 2) + 2*std::pow(x[5]-1, 2) + 5*x[6]*x[6]
        + 7*std::pow(x[7]-11, 2) + 2*std::pow(x[8]-10, 2) + std::pow(x[9]-7, 2) + 45;
    };
    p.ineq.push_back([](const std::vector<double> &x) { return -105 + 4*x[0] + 5*x[1] - 3*x[6] + 9*x[7]; });
    p.ineq.push_back([](const std::vector<double> &x) { return 10*x[0] - 8*x[1] - 17*x[6] + 2*x[7]; });
    p.ineq.push_back([](const std::vector<double> &x) { return -8*x[0] + 2*x[1] + 5*x[8] - 2*x[9] - 12; });
    p.ineq.push_back([](const std::vector<double> &x) { return 3*std::pow(x[0]-2, 2) + 4*std::pow(x[1]-3, 2) + 2*x[2]*x[2] - 7*x[3] - 120; });
    p.ineq.push_back([](const std::vector<double> &x) { return 5*x[0]*x[0] + 8*x[1] + std::pow(x[2]-6, 2) - 2*x[3] - 40; });
    p.ineq.push_back([](const std::vector<double> &x) { return x[0]*x[0] + 2*std::pow(x[1]-2, 2) - 2*x[0]*x[1] + 14*x[4] - 6*x[5]; });
    p.ineq.push_back([](const std::vector<double> &x) { return 0.5*std::pow(x[0]-8, 2) + 2*std::pow(x[1]-4, 2) + 3*x[4]*x[4] - x[5] - 30; });
    p.ineq.push_back([](const std::vector<double> &x) { return -3*x[0] + 6*x[1] + 12*std::pow(x[8]-8, 2) - 7*x[9]; });
    p.xopt = {2.17199634142692, 2.3636830416034, 8.77392573913157, 5.09598443745173, 0.990654756560493,
              1.43057392853463, 1.32164415364306, 9.82872576524495, 8.2800915887356, 8.3759266477347};
    p.fopt = 24.30620906818; p.tol = 2.4e-3; p.eqtol = 0;
    ps.push_back(p);
  }
  {  // g09
    Problem p;
    p.name = "g09";
    p.lb = std::vector<double>(7, -10.0); p.ub = std::vector<double>(7, 10.0); p.x0 = std::vector<double>(7, 0.0);
    p.f = [](const std::vector<double> &x) {
      return std::pow(x[0]-10, 2) + 5*std::pow(x[1]-12, 2) + std::pow(x[2], 4) + 3*std::pow(x[3]-11, 2)
        + 10*std::pow(x[4], 6) + 7*x[5]*x[5] + std::pow(x[6], 4) - 4*x[5]*x[6] - 10*x[5] - 8*x[6];
    };
    p.ineq.push_back([](const std::vector<double> &x) { return -127 + 2*x[0]*x[0] + 3*std::pow(x[1], 4) + x[2] + 4*x[3]*x[3] + 5*x[4]; });
    p.ineq.push_back([](const std::vector<double> &x) { return -282 + 7*x[0] + 3*x[1] + 10*x[2]*x[2] + x[3] - x[4]; });
    p.ineq.push_back([](const std::vector<double> &x) { return -196 + 23*x[0] + x[1]*x[1] + 6*x[5]*x[5] - 8*x[6]; });
    p.ineq.push_back([](const std::vector<double> &x) { return 4*x[0]*x[0] + x[1]*x[1] - 3*x[0]*x[1] + 2*x[2]*x[2] + 5*x[5] - 11*x[6]; });
    p.xopt = {2.33049935147405174, 1.95137236847114592, -0.477541399510615805, 4.36572624923625874,
              -0.624486959100388983, 1.03813099410962173, 1.5942266780671519};
    p.fopt = 680.630057374402; p.tol = 0.068; p.eqtol = 0;
    ps.push_back(p);
  }
  {  // g12
    Problem p;
    p.name = "g12";
    p.lb = {0, 0, 0}; p.ub = {10, 10, 10}; p.x0 = {2, 8, 3};
    p.f = [](const std::vector<double> &x) {
      return -(100 - std::pow(x[0]-5, 2) - std::pow(x[1]-5, 2) - std::pow(x[2]-5, 2)) / 100;
    };
    p.ineq.push_back([](const std::vector<double> &x) {
      double g = HUGE_VAL;
      for (int a = 1; a <= 9; ++a) for (int b = 1; b <= 9; ++b) for (int c = 1; c <= 9; ++c)
        g = std::min(g, std::pow(x[0]-a, 2) + std::pow(x[1]-b, 2) + std::pow(x[2]-c, 2) - 0.0625);
      return g;
    });
    p.xopt = {5, 5, 5};
    p.fopt = -1; p.tol = 1e-4; p.eqtol = 0;
    ps.push_back(p);
  }
  {  // g24
    Problem p;
    p.name = "g24";
    p.lb = {0, 0}; p.ub = {3, 4}; p.x0 = {1.5, 2};
    p.f = [](const std::vector<double> &x) { return -x[0] - x[1]; };
    p.ineq.push_back([](const std::vector<double> &x) {
      return -2*std::pow(x[0], 4) + 8*std::pow(x[0], 3) - 8*x[0]*x[0] + x[1] - 2; });
    p.ineq.push_back([](const std::vector<double> &x) {
      return -4*std::pow(x[0], 4) + 32*std::pow(x[0], 3) - 88*x[0]*x[0] + 96*x[0] + x[1] - 36; });
    p.xopt = {2.32952019747762, 3.17849307411774};
    p.fopt = -5.50801327159536; p.tol = 5.5e-4; p.eqtol = 0;
    ps.push_back(p);
  }
  {  // g11 with a looser equality tolerance
    Problem p;
    p.name = "g11_1e-3";
    p.lb = {-1, -1}; p.ub = {1, 1}; p.x0 = {0.5, 0.5};
    p.f = [](const std::vector<double> &x) { return x[0] * x[0] + std::pow(x[1] - 1, 2); };
    p.eq.push_back([](const std::vector<double> &x) { return x[1] - x[0] * x[0]; });
    p.fopt = 0.75; p.tol = 1e-3; p.eqtol = 1e-3;
    ps.push_back(p);
  }
  return ps;
}

int main(int argc, char *argv[])
{
  const char *variant = argc > 1 ? argv[1] : "?";
  int nseeds = argc > 2 ? std::atoi(argv[2]) : 100;
  int maxeval = argc > 3 ? std::atoi(argv[3]) : 100000;
  std::vector<Problem> ps = problems();
  for (Problem &p : ps) {
    if (p.xopt.empty()) continue;
    double fx = p.f(p.xopt), gmax = -HUGE_VAL;
    for (Fn &g : p.ineq) gmax = std::max(gmax, g(p.xopt));
    if (std::fabs(fx - p.fopt) > 1e-6 * std::max(1.0, std::fabs(p.fopt)) || gmax > 1e-6)
      std::fprintf(stderr, "self-check failed for %s: f(x*) = %.12g, max g = %.3g\n", p.name.c_str(), fx, gmax);
  }
  if (argc > 4) {  // run only the named problem
    std::vector<Problem> keep;
    for (Problem &p : ps) if (p.name == argv[4]) keep.push_back(p);
    ps = keep;
  }
  for (Problem &p : ps) {
    for (int seed = 1; seed <= nseeds; ++seed) {
      nlopt::srand(seed);
      nlopt::opt opt(nlopt::GN_ISRES, (unsigned) p.lb.size());
      opt.set_lower_bounds(p.lb);
      opt.set_upper_bounds(p.ub);
      opt.set_min_objective(call, &p.f);
      for (Fn &g : p.ineq) opt.add_inequality_constraint(call, &g, 1e-8);
      for (Fn &h : p.eq) opt.add_equality_constraint(call, &h, p.eqtol);
      opt.set_stopval(p.fopt + p.tol);
      opt.set_maxeval(maxeval);
      std::vector<double> x = p.x0;
      double minf = HUGE_VAL;
      nlopt::result r;
      try {
        r = opt.optimize(x, minf);
      } catch (std::exception &e) {
        r = nlopt::FAILURE;
      }
      std::printf("%s,%s,%d,%d,%d,%.12g\n", variant, p.name.c_str(), seed, (int) r,
                  opt.get_numevals(), minf);
    }
  }
  return 0;
}
