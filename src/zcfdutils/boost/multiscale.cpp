#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include "converters.h"
#include "nanoflann.hpp"

namespace p = boost::python;
namespace np = boost::python::numpy;

using namespace boost::python;
using namespace pygen;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix_t;
using my_kd_tree_t = nanoflann::KDTreeEigenMatrixAdaptor<Matrix_t>;

struct multiscale
{
    multiscale(Matrix_t input_X, int num_base, double base_radii)
    {
        X = input_X;
        nb = num_base;
        ncp = X.rows();
        r0 = base_radii;
        ndim = X.cols();

        reordered = false;

        // run  checks
        if (nb > ncp)
        {
            std::cout << "Error, n_base > n_nodes, setting full RBF (n_base = n_nodes)" << std::endl;
            nb = ncp;
        }
    };

    void sample_control_points(bool parental_preference = true)
    {
        std::vector<int> inactive_list(ncp);
        std::iota(inactive_list.begin(), inactive_list.end(), 0);

        std::vector<double> sep_dist(ncp, 1e10);
        radii.resize(ncp);
        radii.setConstant(r0);

        std::vector<int> parent(ncp, 0), children;
        std::vector<double> sep_dist_temp;

        std::vector<int> active_list_temp;

        // create tree

        my_kd_tree_t X_tree(ndim, std::cref(X), 20);

        std::vector<double> query_pt(ndim);

        std::vector<size_t> ret_indexes(ncp);
        std::vector<std::pair<Eigen::Index, double>> matches;
        nanoflann::SearchParameters params;
        size_t nMatches;
        nanoflann::KNNResultSet<double> resultsSet(ncp);

        // resultsSet.init(&ret_indexes[0], &out_dists_sqr[0]);

        active_list.resize(ncp);

        int n_active = 0;
        double r;

        int active_node = inactive_list[0];
        int inactive_node, iMax;

        sep_dist[active_node] = -1e10;
        active_list(n_active) = active_node;
        active_list_temp.push_back(active_node);

        base_set.push_back(active_node);
        remove(inactive_list.begin(), inactive_list.end(), active_node);

        n_active = n_active + 1;

        X_tree.index->buildIndex();

        while (n_active < ncp)
        {
            // query all points in tree against last added control point

            // resultsSet.init(&ret_indexes[0], &out_dists_sqr[0]);

            for (int i = 0; i < ndim; ++i)
            {
                query_pt[i] = double(X(active_node, i));
            }

            const double search_radius = sep_dist[active_node];

            const size_t nMatches = X_tree.index->radiusSearch(&query_pt[0], search_radius, matches, params);

            sep_dist[active_node] = -1e10;

            // ordering incorrect for some reason here, need to cross compare to scikit KD tree
            // ordering incorrect whether parental preference is used or not
            // #pragma omp parallel for
            for (int i = 0; i < nMatches; ++i)
            {
                int inactive_node = matches[i].first;
                if (matches[i].second < sep_dist[inactive_node]) // needed this way to ensure negative eliminated values stay the away
                {
                    sep_dist[inactive_node] = matches[i].second;
                    parent[inactive_node] = active_node;
                };
            };

            if (parental_preference)
            {
                // still doesn't work
                copy(sep_dist.begin(), sep_dist.end(), back_inserter(sep_dist_temp));

                if (n_active < nb)
                {
                    for (int i = 0; i < n_active; ++i)
                    {
                        children.push_back(std::count(parent.begin(), parent.end(), active_list[i]));
                    };
                    iMax = argmax(children);
                    for (int i = 0; i < parent.size(); ++i)
                    {
                        if (parent[i] != active_list[iMax])
                        {
                            sep_dist_temp[i] = -1;
                        };
                    };

                    active_node = argmax(sep_dist_temp);
                }
                else
                {
                    active_node = argmax(sep_dist);
                };
            }
            else
            {
                active_node = argmax(sep_dist);
            }

            active_list_temp.push_back(active_node);
            active_list(n_active) = active_node;
            remove(inactive_list.begin(), inactive_list.end(), active_node);

            if (n_active < nb)
            {
                radii(active_node) = r0;
                base_set.push_back(active_node);
            }
            else
            {
                radii(active_node) = sqrt(sep_dist[active_node]);
                remaining_set.push_back(active_node);
            };

            // free up unused memory
            ret_indexes.clear();
            matches.clear();
            query_pt.clear();

            children.clear();
            sep_dist_temp.clear();

            progress_bar(double(n_active) / double(ncp));

            n_active++;
        }
    }

    void progress_bar(double progress)
    {
        int barWidth = 70;
        std::cout << "[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i)
        {
            if (i < pos)
                std::cout << "=";
            else if (i == pos)
                std::cout << ">";
            else
                std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();
    }

    void build_phi_b()
    {
        std::cout << "building phi_b" << std::endl;
        phi_b.resize(nb, nb);
        double r, e, coef;

        int query_node, q;

        Matrix_t X_base = X.block(0, 0, nb, ndim);

        // create tree

        my_kd_tree_t X_tree(ndim, std::cref(X_base), 20);

        std::vector<double> query_pt(ndim);

        size_t nMatches;
        nanoflann::SearchParameters params;
        std::vector<std::pair<Eigen::Index, double>> matches;

        X_tree.index->buildIndex();

        for (int i = 0; i < nb; ++i)
        {
            for (int j = 0; j < ndim; ++j)
            {
                query_pt[j] = double(X_base(i, j));
            }

            const double search_radius = pow(radii[base_set[i]], 2);

            const size_t nMatches = X_tree.index->radiusSearch(&query_pt[0], search_radius, matches, params);

            for (int j = 0; j < nMatches; ++j)
            {
                q = matches[j].first;
                r = matches[j].second;
                e = sqrt(r) / r0;
                assert(e <= 1.0);
                phi_b(i, q) = c2(e);
            };
            progress_bar(double(i) / double(nb));
        };

        // fill via symmetry
        for (int i = 0; i < nb; ++i)
        {
            for (int j = 0; j < nb; ++j)
            {
                coef = std::max(phi_b(i, j), phi_b(j, i));
                phi_b(i, j) = coef;
                phi_b(j, i) = coef;
            };
            progress_bar(double(i) / double(nb));
        };

        phi_b_llt.compute(phi_b);
        std::cout << "built phi_b" << std::endl;
    }

    void build_phi_r()
    {
        std::cout << "building phi_r" << std::endl;
        phi_r.resize(ncp - nb, nb);
        double r, e;
        int query_node, q;

        Matrix_t X_remaining = X.block(nb, 0, ncp - nb, ndim);

        // build tree
        my_kd_tree_t X_tree(ndim, std::cref(X_remaining), 20);

        std::vector<double> query_pt(ndim);

        size_t nMatches;
        nanoflann::SearchParameters params;
        std::vector<std::pair<Eigen::Index, double>> matches;

        X_tree.index->buildIndex();

        for (int i = 0; i < nb; ++i)
        {
            for (int j = 0; j < ndim; ++j)
            {
                // set query point to be i'th base node
                query_pt[j] = double(X(i, j));
            }

            // query remaining set vs base node
            const double search_radius = pow(radii[base_set[i]], 2);

            const size_t nMatches = X_tree.index->radiusSearch(&query_pt[0], search_radius, matches, params);

            for (int j = 0; j < nMatches; ++j)
            {
                q = matches[j].first;
                r = matches[j].second;
                e = sqrt(r) / radii[base_set[i]];
                assert(e <= 1.0);
                phi_r(q, i) = c2(e);
            }
            progress_bar(double(i) / double(nb));
        }
        std::cout << "built phi_r" << std::endl;
    }

    void build_LCSC()
    {
        std::cout << "building LCSC" << std::endl;
        std::vector<Eigen::Triplet<double>> triplet_list;
        int p, q, remainingNode;
        double r, e;

        Matrix_t X_remaining = X.block(nb, 0, ncp - nb, ndim);

        my_kd_tree_t X_tree(ndim, std::cref(X_remaining), 20);
        X_tree.index->buildIndex();

        std::vector<double> query_pt(ndim);
        size_t nMatches;
        nanoflann::SearchParameters params;
        std::vector<std::pair<Eigen::Index, double>> matches;

        X_tree.index->buildIndex();

        for (int i = 0; i < remaining_set.size(); ++i)
        {
            for (int j = 0; j < ndim; ++j)
            {
                query_pt[j] = X_remaining(i, j);
            }

            remainingNode = remaining_set[i];

            const double search_radius = pow(radii[remainingNode], 2);

            const size_t nMatches = X_tree.index->radiusSearch(&query_pt[0], search_radius, matches, params);

            for (int j = 0; j < nMatches; ++j)
            {
                q = matches[j].first;
                r = matches[j].second;
                e = sqrt(r) / radii[remainingNode];
                assert(e <= 1.0);

                triplet_list.push_back(Eigen::Triplet<double>(i, q, c2(e)));
            }
            progress_bar(double(i) / double(remaining_set.size()));
        }
        LCSC.resize(ncp - nb, ncp - nb);
        LCSC.setFromTriplets(triplet_list.begin(), triplet_list.end());
        LCSC.makeCompressed();
        std::cout << "built LCSC" << std::endl;
    }

    void reorder()
    {
        std::cout << "reordering" << std::endl;
        Matrix_t X_new(ncp, ndim);
        Matrix_t dX_new(ncp, ncol);
        Eigen::VectorXd radii_new(ncp);
        int active_node;

        for (int i = 0; i < active_list.size(); ++i)
        {
            active_node = active_list(i);
            X_new.row(i) = X.row(active_node);
            dX_new.row(i) = dX.row(active_node);
            radii_new(i) = radii(active_node);
        }

        X.setZero();
        dX.setZero();
        radii.setZero();

        X = X_new;
        dX = dX_new;
        radii = radii_new;

        base_set.clear();
        remaining_set.clear();

        base_set.resize(nb);
        remaining_set.resize(ncp - nb);

        std::iota(base_set.begin(), base_set.end(), 0);
        std::iota(remaining_set.begin(), remaining_set.end(), nb);

        reordered = true;
        std::cout << "reordered" << std::endl;
    }

    void reorder_dX()
    {
        Matrix_t dX_new(ncp, ncol);
        int active_node;
        for (int i = 0; i < active_list.size(); ++i)
        {
            active_node = active_list(i);
            dX_new.row(i) = dX.row(active_node);
        }
        dX.setZero();
        dX = dX_new;
    }

    void multiscale_solve(Matrix_t dX_input)
    {
        dX = dX_input;
        ncol = dX.cols();

        if (not reordered)
        {
            reorder();
        }
        else
        {
            reorder_dX();
        }

        if (not built)
        {
            build_phi_b();
            build_phi_r();
            build_LCSC();
            built = true;
        }

        Matrix_t a_base = solve_b();
        a = solve_remaining(a_base);
    }

    Matrix_t solve_b()
    {
        std::cout << "solving base set" << std::endl;
        Matrix_t base_dX(nb, ncol);

        base_dX = dX.block(0, 0, nb, ncol);

        // Matrix_t a_base = phi_b.partialPivLu().solve(base_dX);
        Matrix_t a_base = phi_b_llt.solve(base_dX);
        std::cout
            << "Solved base set" << std::endl;

        return a_base;
    }

    Matrix_t solve_remaining(Matrix_t a_base)
    {
        std::cout << "solving remaining" << std::endl;
        int ptr;
        double c;

        Matrix_t dX_res = dX;
        dX_res.block(nb, 0, ncp - nb, ncol) = dX_res.block(nb, 0, ncp - nb, ncol) - phi_r * a_base;

        Matrix_t coef(ncp, ncol);
        coef.block(0, 0, nb, ncol) = a_base;

        for (int i = 0; i < remaining_set.size(); ++i)
        {
            coef.row(i + nb) << dX_res.row(i + nb);
            for (int j = LCSC.outerIndexPtr()[i]; j < LCSC.outerIndexPtr()[i + 1]; ++j)
            {
                ptr = LCSC.innerIndexPtr()[j];
                c = LCSC.valuePtr()[j];
                dX_res.row(ptr + nb) = dX_res.row(ptr + nb) - c * coef.row(i + nb);
            }
        }
        std::cout << "solved remaining" << std::endl;
        return coef;
    }

    void preprocess_V(Matrix_t input_V)
    {
        V = input_V;
        nv = V.rows();
        double radSquared, e;
        int targetNode, totalColEntries, numColEntries;

        my_kd_tree_t X_tree(ndim, std::cref(V), 10);

        std::vector<double> query_pt(ndim);

        std::vector<std::vector<int>> psi_v(nv, std::vector<int>(1, 0));
        std::vector<std::vector<int>> psi_v_val_temp(nv, std::vector<int>(1, 0));

        size_t nMatches;
        nanoflann::SearchParameters params;
        std::vector<std::pair<Eigen::Index, double>> matches;

        X_tree.index->buildIndex();

        int count = 0;
        int j = 0;

        psi_v_rowptr.push_back(0);

        for (int remainingNode : remaining_set)
        {
            for (int i = 0; i < ndim; ++i)
            {
                query_pt[i] = X(remainingNode, i);
            }
            const double search_radius = pow(radii[remainingNode], 2);

            const size_t nMatches = X_tree.index->radiusSearch(&query_pt[0], search_radius, matches, params);

            for (int i = 0; i < nMatches; ++i)
            {
                targetNode = matches[i].first;
                radSquared = matches[i].second;

                e = sqrt(radSquared) / radii[remainingNode];

                assert(e <= 1.0);

                psi_v[targetNode]
                    .push_back(remainingNode);
                psi_v_val_temp[targetNode].push_back(c2(e));
            }
        }

        totalColEntries = 0;

        for (int i = 0; i < nv; ++i)
        {
            numColEntries = psi_v[i].size();
            totalColEntries += numColEntries;
            psi_v_rowptr.push_back(totalColEntries);
            psi_v_val.push_back(0);
            for (int j = 0; j < numColEntries; ++j)
            {
                psi_v_col_index.push_back(psi_v[i][j]);
                psi_v_val.push_back(psi_v_val_temp[i][j]);
            }
        }
    }

    void multiscale_transfer()
    {
        std::cout << "begining transfer" << std::endl;
        double r, e, c, radSquared;
        int q, targetNode;

        dV.resize(nv, ncol);
        dV.setZero();

        Matrix_t X_base = X.block(0, 0, nb, ndim);

        my_kd_tree_t X_tree(ndim, std::cref(X_base), 20);

        std::vector<double> query_pt(ndim);
        size_t nMatches;
        nanoflann::SearchParameters params;
        std::vector<std::pair<Eigen::Index, double>> matches;

        X_tree.index->buildIndex();

        for (int i = 0; i < nv; ++i)
        {
            for (int k = psi_v_rowptr[i]; k < psi_v_rowptr[i + 1]; ++k)
            {
                if (psi_v_col_index[k] == 0)
                {
                    for (int j = 0; j < ndim; ++j)
                    {
                        query_pt[j] = V(i, j);
                    }

                    const double search_radius = pow(r0, 2);
                    const size_t nMatches = X_tree.index->radiusSearch(&query_pt[0], search_radius, matches, params);
                    for (int j = 0; j < nMatches; ++j)
                    {
                        targetNode = matches[j].first;
                        radSquared = matches[j].second;

                        e = sqrt(radSquared) / r0;
                        assert(e <= 1.0);

                        c = c2(e);
                        dV.row(i) += c * a.row(targetNode);
                    }
                }
                else
                {
                    q = psi_v_col_index[k];
                    dV.row(i) += psi_v_val[k] * a.row(q);
                }
            }
            progress_bar(double(i) / double(nv));
        }
        std::cout << "Finished transfer" << std::endl;
    }

    double c2(double r)
    {
        return (pow(1 - r, 4) * (4 * r + 1));
    }

    template <typename T>
    int argmax(std::vector<T> vec)
    {
        return std::distance(vec.begin(), max_element(vec.begin(), vec.end()));
    }

    // getters
    const Matrix_t &get_X() const noexcept
    {
        return X;
    }

    const Matrix_t &get_a() const noexcept
    {
        return a;
    }

    const Matrix_t &get_dV() const noexcept
    {
        return dV;
    }

    const Eigen::VectorXi &get_active_list() const noexcept
    {
        return active_list;
    }

    const Eigen::VectorXd &get_radii() const noexcept
    {
        return radii;
    }

    const Eigen::VectorXi &get_tree_ind() const noexcept
    {
        return tree_ind;
    }

    const Eigen::VectorXd &get_tree_dist() const noexcept
    {
        return tree_dist;
    }

    void set_active_list(Eigen::VectorXd py_active_list)
    {
        // need this extra bit of code to get around python resizing integers
        int num_entries = py_active_list.size();
        active_list.resize(num_entries);
        for (int i = 0; i < num_entries; ++i)
        {
            active_list(i) = int(py_active_list(i));
        }
    }

    void set_radii(Eigen::VectorXd py_radii)
    {
        radii = py_radii;
    }

    Eigen::VectorXd tree_dist;
    Eigen::VectorXi tree_ind;
    Eigen::LLT<Matrix_t> phi_b_llt;

    int nb, ncp, nv, ncol, ndim;
    double r0;
    Matrix_t a, dV, X, V, dX, phi_b, phi_r;
    Eigen::VectorXi active_list;
    std::vector<int> base_set, remaining_set, psi_v_rowptr, psi_v_col_index;
    std::vector<double> psi_v_val;
    Eigen::SparseMatrix<double> LCSC;
    Eigen::VectorXd radii;
    bool reordered, built;
};

BOOST_PYTHON_MODULE(multiscale)
{
    Py_Initialize();
    np::initialize();

    pygen::convert<double>(pygen::Converters::All, false);
    pygen::convert<int>(pygen::Converters::All, false);

    class_<multiscale>("multiscale", init<Matrix_t, int, double>())
        .def("sample_control_points", &multiscale::sample_control_points)
        .def("multiscale_solve", &multiscale::multiscale_solve)
        .def("preprocess_V", &multiscale::preprocess_V)
        .def("multiscale_transfer", &multiscale::multiscale_transfer)

        .def("get_X", &multiscale::get_X, py::return_value_policy<py::copy_const_reference>())
        .def("get_a", &multiscale::get_a, py::return_value_policy<py::copy_const_reference>())
        .def("get_dV", &multiscale::get_dV, py::return_value_policy<py::copy_const_reference>())
        .def("get_radii", &multiscale::get_radii, py::return_value_policy<py::copy_const_reference>())
        .def("get_active_list", &multiscale::get_active_list, py::return_value_policy<py::copy_const_reference>())
        .def("get_tree_dist", &multiscale::get_tree_dist, py::return_value_policy<py::copy_const_reference>())
        .def("get_tree_ind", &multiscale::get_tree_ind, py::return_value_policy<py::copy_const_reference>())
        .def("set_active_list", &multiscale::set_active_list)
        .def("set_radii", &multiscale::set_radii);
}
