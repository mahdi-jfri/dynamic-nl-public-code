#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "nonlinear_ppr.h"

namespace py = pybind11;
using namespace nlppr;

PYBIND11_MODULE(nl_ppr, m) {
    py::enum_<StepFn>(m, "StepFn")
        .value("IDENTITY", SF_IDENTITY)
        .value("SIGMOID", SF_SIGMOID)
        .value("TANH", SF_TANH)
        .value("SOFTPLUS", SF_SOFTPLUS)
        .value("RELU", SF_RELU)
        .value("CLAMP_SYM", SF_CLAMP_SYM)
        .value("STANH", SF_STANH)
        .value("LEAKY05", SF_LEAKY05)
        .value("SHTANH", SF_SHTANH)
        .value("HTANH", SF_HTANH);

    py::enum_<ThresholdMode>(m, "ThresholdMode")
        .value("DEGREE", TH_DEGREE)
        .value("ROWSUM", TH_ROWSUM);

    py::class_<NonlinearPPR>(m, "NonlinearPPR")
        .def(py::init<int64_t, int, double, double, double, int, double, int>(),
             py::arg("n"), py::arg("F"), py::arg("alpha"),
             py::arg("beta"), py::arg("K"),
             py::arg("step_fn"), py::arg("step_param") = 1.0,
             py::arg("threshold_mode") = (int)TH_DEGREE)
        .def_readonly("n", &NonlinearPPR::n)
        .def_readonly("F", &NonlinearPPR::F)
        .def_readonly("alpha", &NonlinearPPR::alpha)
        .def_readonly("beta", &NonlinearPPR::beta)
        .def_readonly("K", &NonlinearPPR::K)
        .def("set_features", [](NonlinearPPR& self,
                py::array_t<double, py::array::c_style | py::array::forcecast> X) {
            if (X.ndim() != 2)
                throw std::runtime_error("X must be 2D");
            if (X.shape(0) != self.F || X.shape(1) != self.n)
                throw std::runtime_error("X shape must be [F, N]");
            self.set_features(X.data());
        })
        .def("initial_operation", [](NonlinearPPR& self,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> edges,
                double eps) {
            if (edges.ndim() != 2 || edges.shape(1) != 2)
                throw std::runtime_error("edges must have shape [E, 2]");
            return self.initial_operation(edges.data(), edges.shape(0), eps);
        }, py::arg("edges"), py::arg("eps"))
        .def("snapshot_operation", [](NonlinearPPR& self,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> events,
                double eps) {
            if (events.ndim() != 2 || events.shape(1) != 3)
                throw std::runtime_error("events must have shape [M, 3]");
            return self.snapshot_operation(events.data(), events.shape(0), eps);
        }, py::arg("events"), py::arg("eps"))
        .def("snapshot_operation_batched", [](NonlinearPPR& self,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> events,
                double eps) {
            if (events.ndim() != 2 || events.shape(1) != 3)
                throw std::runtime_error("events must have shape [M, 3]");
            return self.snapshot_operation_batched(
                events.data(), events.shape(0), eps);
        }, py::arg("events"), py::arg("eps"))
        .def("apply_edge_events", [](NonlinearPPR& self,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> events) {
            if (events.ndim() != 2 || events.shape(1) != 3)
                throw std::runtime_error("events must have shape [M, 3]");
            self.apply_edge_events(events.data(), events.shape(0));
        })
        .def("reset_state", &NonlinearPPR::reset_state)
        .def("cleanup", &NonlinearPPR::cleanup, py::arg("eps"))
        .def("get_z", [](const NonlinearPPR& self) {
            py::array_t<double> out({(py::ssize_t)self.F, (py::ssize_t)self.n});
            self.get_z(out.mutable_data());
            return out;
        })
        .def("get_y", [](const NonlinearPPR& self) {
            py::array_t<double> out({(py::ssize_t)self.F, (py::ssize_t)self.n});
            self.get_y(out.mutable_data());
            return out;
        })
        .def("get_r", [](const NonlinearPPR& self) {
            py::array_t<double> out({(py::ssize_t)self.F, (py::ssize_t)self.n});
            self.get_r(out.mutable_data());
            return out;
        })
        .def("degrees", [](const NonlinearPPR& self) {
            auto d = self.degrees();
            py::array_t<int32_t> out({(py::ssize_t)d.size()});
            std::memcpy(out.mutable_data(), d.data(),
                        sizeof(int32_t) * d.size());
            return out;
        })
        .def("total_residual_l1", &NonlinearPPR::total_residual_l1);
}
