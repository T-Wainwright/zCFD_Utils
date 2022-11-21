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

struct multiscale
{
    multiscale(Matrix_t input_X, int num_base, double base_radii)
    {
        X = input_X;
        nb = num_base;
        ncp = X.rows();
        r0 = base_radii;

        // run  checks
        if (nb > ncp)
        {
            std::cout << "Error, n_base > n_nodes, setting full RBF (n_base = n_nodes)" << std::endl;
            nb = ncp;
        }
    };

    void sample_control_points()
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

        using my_kd_tree_t = nanoflann::KDTreeEigenMatrixAdaptor<Matrix_t>;
        my_kd_tree_t X_tree(3, std::cref(X), 20);

        std::vector<double> query_pt(3);

        std::vector<size_t> ret_indexes(ncp);
        std::vector<double> out_dists_sqr(ncp);
        nanoflann::KNNResultSet<double> resultsSet(ncp);

        resultsSet.init(&ret_indexes[0], &out_dists_sqr[0]);

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

        while (n_active < ncp)
        {
            // query all points in tree against last added control point

            resultsSet.init(&ret_indexes[0], &out_dists_sqr[0]);

            for (int i = 0; i < 3; ++i)
            {
                query_pt[i] = double(X(active_node, i));
            }

            X_tree.index->findNeighbors(resultsSet, &query_pt[0]);

            // ordering incorrect for some reason here, need to cross compare to scikit KD tree
            // ordering incorrect whether parental preference is used or not
            for (int i = 0; i < ncp; ++i)
            {
                inactive_node = ret_indexes[i];
                if (out_dists_sqr[i] < (sep_dist[inactive_node] * abs(sep_dist[inactive_node]))) // needed this way to ensure negative eliminated values stay the away
                {
                    sep_dist[inactive_node] = sqrt(out_dists_sqr[i]);
                    parent[inactive_node] = active_node;
                };
            };

            if (0)
            {
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

            // parent[active_node] = active_node;

            if (n_active < nb)
            {
                radii(active_node) = r0;
                base_set.push_back(active_node);
            }
            else
            {
                radii(active_node) = sep_dist[active_node];
                remaining_set.push_back(active_node);
            };

            sep_dist[active_node] = -1e10;

            ret_indexes.clear();
            out_dists_sqr.clear();

            children.clear();
            sep_dist_temp.clear();

            progress_bar(n_active);

            n_active++;
        }
    }

    void test_tree(int query_node)
    {
        using my_kd_tree_t = nanoflann::KDTreeEigenMatrixAdaptor<Matrix_t>;
        my_kd_tree_t X_tree(3, std::cref(X), 10);

        std::vector<double> query_pt(3);

        std::vector<size_t> ret_indexes(ncp);
        std::vector<double> out_dists_sqr(ncp);
        nanoflann::KNNResultSet<double> resultsSet(ncp);

        resultsSet.init(&ret_indexes[0], &out_dists_sqr[0]);

        tree_dist.resize(ncp);
        tree_ind.resize(ncp);

        for (int i = 0; i < 3; ++i)
        {
            query_pt[i] = double(X(query_node, i));
        }

        X_tree.index->findNeighbors(resultsSet, &query_pt[0]);

        for (int i = 0; i < ncp; ++i)
        {
            tree_dist(i) = sqrt(out_dists_sqr[i]);
            tree_ind(i) = ret_indexes[i];
        }
    }

    void progress_bar(int n_active)
    {
        int barWidth = 70;
        double progress = double(n_active) / double(ncp);
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

    Matrix_t build_phi_b()
    {
        Matrix_t phi_b(nb, nb);
        double r, e, coef;
        for (int i = 0; i < nb; ++i)
        {
            for (int j = 0; j < nb; ++j)
            {
                r = (X.row(base_set[i]) - X.row(base_set[j])).squaredNorm();
                e = r / pow(radii[base_set[j]], 2);
                if (e <= 1.0)
                {
                    phi_b(i, j) = c2(sqrt(e));
                }
            };
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
        };
        return phi_b;
    }

    Matrix_t build_phi_r()
    {
        Matrix_t phi_r(ncp - nb, nb);
        double r, e;
        int base_node, remaining_node;
        for (int i = 0; i < remaining_set.size(); ++i)
        {
            for (int j = 0; j < base_set.size(); ++j)
            {
                remaining_node = remaining_set[i];
                base_node = base_set[j];
                r = (X.row(base_node) - X.row(remaining_node)).squaredNorm();
                e = r / pow(radii[base_node], 2);
                if (e <= 1.0)
                {
                    phi_r(i, j) = c2(sqrt(e));
                }
            }
        }
        return phi_r;
    }

    Eigen::SparseMatrix<double> build_LCSC()
    {
        std::vector<Eigen::Triplet<double>> triplet_list;
        int p, q;
        double r, e;

        for (int i = 0; i < remaining_set.size(); ++i)
        {
            p = remaining_set[i];
            for (int j = 0; j < i + 1; ++j)
            {
                q = remaining_set[j];
                r = (X.row(p) - X.row(q)).squaredNorm();
                e = r / pow(radii[q], 2);
                if (e <= 1.0)
                {
                    triplet_list.push_back(Eigen::Triplet<double>(i, j, c2(sqrt(e))));
                }
            }
        }
        Eigen::SparseMatrix<double> LCSC(ncp - nb, ncp - nb);
        LCSC.setFromTriplets(triplet_list.begin(), triplet_list.end());
        LCSC.makeCompressed();
        return LCSC;
    }

    void reorder()
    {
        Matrix_t X_new(ncp, 3);
        Matrix_t dX_new(ncp, 3);
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
    }

    void multiscale_solve(Matrix_t dX_input)
    {
        dX = dX_input;
        reorder();
        Matrix_t phi_b = build_phi_b();
        Matrix_t phi_r = build_phi_r();
        Eigen::SparseMatrix<double> LCRC = build_LCSC();

        Matrix_t a_base = solve_b(phi_b);
        a = solve_remaining(phi_r, LCRC, a_base);
    }

    Matrix_t solve_b(Matrix_t phi_b)
    {
        Matrix_t base_dX(nb, 3);

        base_dX = dX.block(0, 0, nb, 3);
        Eigen::LLT<Matrix_t> llt;
        llt.compute(phi_b);
        Matrix_t a_base = llt.solve(base_dX);
        std::cout << "Solved base set" << std::endl;

        return a_base;
    }

    Matrix_t solve_remaining(Matrix_t phi_r, Eigen::SparseMatrix<double> LCRC, Matrix_t a_base)
    {
        int ptr;
        double c;

        Matrix_t dX_res(ncp, 3);
        for (int i = 0; i < ncp; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                dX_res(i, j) = dX(i, j);
            }
        }
        dX_res.block(nb, 0, ncp - nb, 3) = dX_res.block(nb, 0, ncp - nb, 3) - phi_r * a_base;

        Matrix_t coef(ncp, 3);
        coef.block(0, 0, nb, 3) = a_base;

        for (int i = 0; i < remaining_set.size(); ++i)
        {
            coef.row(i + nb) << dX_res.row(i + nb);
            for (int j = LCRC.outerIndexPtr()[i]; j < LCRC.outerIndexPtr()[i + 1]; ++j)
            {
                ptr = LCRC.innerIndexPtr()[j];
                c = LCRC.valuePtr()[j];
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
        double r, e;

        int count = 0;
        int j = 0;

        psi_v_rowptr.push_back(0);

        for (int i = 0; i < nv; ++i)
        {
            psi_v_col_index.push_back(0);
            ++j;
            for (int k = nb; k < ncp; ++k)
            {
                r = (V.row(i) - X.row(k)).squaredNorm();

                e = r / pow(radii[k], 2);
                if (e <= 1.0)
                {
                    psi_v_col_index.push_back(k);
                    ++j;
                }
            }
            psi_v_rowptr.push_back(j);
        }
    }

    void multiscale_transfer()
    {
        double r, e, c;
        int q;

        dV.resize(nv, 3);
        dV.setZero();

        for (int i = 0; i < nv; ++i)
        {
            for (int k = psi_v_rowptr[i]; k < psi_v_rowptr[i + 1]; ++k)
            {
                if (psi_v_col_index[k] == 0)
                {
                    for (int j = 0; j < nb; ++j)
                    {
                        r = (V.row(i) - X.row(j)).squaredNorm();
                        e = r / pow(radii[j], 2);
                        if (e <= 1.0)
                        {
                            c = c2(sqrt(e));
                            dV.row(i) += c * a.row(j);
                        }
                    }
                }
                else
                {
                    q = psi_v_col_index[k];
                    r = (V.row(i) - X.row(q)).squaredNorm();
                    e = r / pow(radii[q], 2);
                    if (e <= 1.0)
                    {
                        c = c2(sqrt(e));
                        dV.row(i) += c * a.row(q);
                    }
                }
            }
        }
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

    Eigen::VectorXd tree_dist;
    Eigen::VectorXi tree_ind;

    int nb, ncp, nv;
    double r0;
    Matrix_t a, dV, X, V, dX;
    Eigen::VectorXi active_list;
    std::vector<int> base_set, remaining_set, psi_v_rowptr, psi_v_col_index;
    Eigen::VectorXd radii;
};

BOOST_PYTHON_MODULE(multiscale)
{
    Py_Initialize();
    np::initialize();

    pygen::convert<double>(pygen::Converters::All);
    pygen::convert<int>(pygen::Converters::All);

    class_<multiscale>("multiscale", init<Matrix_t, int, double>())
        .def("sample_control_points", &multiscale::sample_control_points)
        .def("multiscale_solve", &multiscale::multiscale_solve)
        .def("preprocess_V", &multiscale::preprocess_V)
        .def("multiscale_transfer", &multiscale::multiscale_transfer)
        .def("test_tree", &multiscale::test_tree)
        .def("get_X", &multiscale::get_X, py::return_value_policy<py::copy_const_reference>())
        .def("get_a", &multiscale::get_a, py::return_value_policy<py::copy_const_reference>())
        .def("get_dV", &multiscale::get_dV, py::return_value_policy<py::copy_const_reference>())
        .def("get_radii", &multiscale::get_radii, py::return_value_policy<py::copy_const_reference>())
        .def("get_active_list", &multiscale::get_active_list, py::return_value_policy<py::copy_const_reference>())
        .def("get_tree_dist", &multiscale::get_tree_dist, py::return_value_policy<py::copy_const_reference>())
        .def("get_tree_ind", &multiscale::get_tree_ind, py::return_value_policy<py::copy_const_reference>());
}
