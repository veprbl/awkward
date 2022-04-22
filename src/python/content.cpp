// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/python/content.cpp", line)

#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/chrono.h>

#include "awkward/builder/ArrayBuilderOptions.h"

#include "awkward/python/util.h"
#include "awkward/datetime_util.h"

#include "awkward/python/content.h"

////////// ArrayBuilder

bool
builder_fromiter_iscomplex(const py::handle& obj) {
#if PY_MAJOR_VERSION < 3
  return py::isinstance(obj, py::module::import("__builtin__").attr("complex"));
#else
  return py::isinstance(obj, py::module::import("builtins").attr("complex"));
#endif
}

void
builder_datetime(ak::ArrayBuilder& self, const py::handle& obj) {
  if (py::isinstance<py::str>(obj)) {
    auto date_time = py::module::import("numpy").attr("datetime64")(obj);
    auto ptr = date_time.attr("astype")(py::module::import("numpy").attr("int64"));
    auto units = py::str(py::module::import("numpy").attr("dtype")(date_time)).cast<std::string>();
    self.datetime(ptr.cast<int64_t>(), units);
  }
  else if (py::isinstance(obj, py::module::import("numpy").attr("datetime64"))) {
    auto ptr = obj.attr("astype")(py::module::import("numpy").attr("int64"));
    self.datetime(ptr.cast<int64_t>(), py::str(obj.attr("dtype")));
  }
  else {
    throw std::invalid_argument(
      std::string("cannot convert ")
      + obj.attr("__repr__")().cast<std::string>() + std::string(" (type ")
      + obj.attr("__class__").attr("__name__").cast<std::string>()
      + std::string(") to an array element") + FILENAME(__LINE__));
  }
}

void
builder_timedelta(ak::ArrayBuilder& self, const py::handle& obj) {
  if (py::isinstance<py::str>(obj)) {
    auto date_time = py::module::import("numpy").attr("timedelta64")(obj);
    auto ptr = date_time.attr("astype")(py::module::import("numpy").attr("int64"));
    auto units = py::str(py::module::import("numpy").attr("dtype")(date_time)).cast<std::string>();
    self.timedelta(ptr.cast<int64_t>(), units);
  }
  else if (py::isinstance(obj, py::module::import("numpy").attr("timedelta64"))) {
    auto ptr = obj.attr("astype")(py::module::import("numpy").attr("int64"));
    self.timedelta(ptr.cast<int64_t>(), py::str(obj.attr("dtype")));
  }
  else {
    throw std::invalid_argument(
      std::string("cannot convert ")
      + obj.attr("__repr__")().cast<std::string>() + std::string(" (type ")
      + obj.attr("__class__").attr("__name__").cast<std::string>()
      + std::string(") to an array element") + FILENAME(__LINE__));
  }
}

void
builder_fromiter(ak::ArrayBuilder& self, const py::handle& obj) {
  if (obj.is(py::none())) {
    self.null();
  }
  else if (py::isinstance<py::bool_>(obj)) {
    self.boolean(obj.cast<bool>());
  }
  else if (py::isinstance<py::int_>(obj)) {
    self.integer(obj.cast<int64_t>());
  }
  else if (py::isinstance<py::float_>(obj)) {
    self.real(obj.cast<double>());
  }
  else if (builder_fromiter_iscomplex(obj)) {
    self.complex(obj.cast<std::complex<double>>());
  }
  else if (py::isinstance<py::bytes>(obj)) {
    self.bytestring(obj.cast<std::string>());
  }
  else if (py::isinstance<py::str>(obj)) {
    self.string(obj.cast<std::string>());
  }
  else if (py::isinstance<py::tuple>(obj)) {
    py::tuple tup = obj.cast<py::tuple>();
    self.begintuple(tup.size());
    for (size_t i = 0;  i < tup.size();  i++) {
      self.index((int64_t)i);
      builder_fromiter(self, tup[i]);
    }
    self.endtuple();
  }
  else if (py::isinstance<py::dict>(obj)) {
    py::dict dict = obj.cast<py::dict>();
    self.beginrecord();
    for (auto pair : dict) {
      if (!py::isinstance<py::str>(pair.first)) {
        throw std::invalid_argument(
          std::string("keys of dicts in 'fromiter' must all be strings")
          + FILENAME(__LINE__));
      }
      std::string key = pair.first.cast<std::string>();
      self.field_check(key.c_str());
      builder_fromiter(self, pair.second);
    }
    self.endrecord();
  }
  else if (py::isinstance(obj, py::module::import("awkward").attr("Array"))) {
    builder_fromiter(self, obj.attr("tolist")());
  }
  else if (py::isinstance(obj, py::module::import("awkward").attr("Record"))) {
    builder_fromiter(self, obj.attr("tolist")());
  }
  else if (py::isinstance<py::iterable>(obj)) {
    py::iterable seq = obj.cast<py::iterable>();
    self.beginlist();
    for (auto x : seq) {
      builder_fromiter(self, x);
    }
    self.endlist();
  }
  else if (py::isinstance<py::array>(obj)) {
    builder_fromiter(self, obj.attr("tolist")());
  }
  else if (py::isinstance(obj, py::module::import("numpy").attr("datetime64"))) {
    builder_datetime(self, obj);
  }
  else if (py::isinstance(obj, py::module::import("numpy").attr("timedelta64"))) {
    builder_timedelta(self, obj);
  }
  else if (py::isinstance(obj, py::module::import("numpy").attr("bool_"))) {
    self.boolean(obj.cast<bool>());
  }
  else if (py::isinstance(obj, py::module::import("numpy").attr("integer"))) {
    self.integer(obj.cast<int64_t>());
  }
  else if (py::isinstance(obj, py::module::import("numpy").attr("floating"))) {
    self.real(obj.cast<double>());
  }
  else {

    throw std::invalid_argument(
      std::string("cannot convert ")
      + obj.attr("__repr__")().cast<std::string>() + std::string(" (type ")
      + obj.attr("__class__").attr("__name__").cast<std::string>()
      + std::string(") to an array element") + FILENAME(__LINE__));
  }
}

py::class_<ak::ArrayBuilder>
make_ArrayBuilder(const py::handle& m, const std::string& name) {
  return (py::class_<ak::ArrayBuilder>(m, name.c_str())
      .def(py::init([](int64_t initial, double resize) -> ak::ArrayBuilder {
        return ak::ArrayBuilder(ak::ArrayBuilderOptions(initial, resize));
      }), py::arg("initial") = 1024, py::arg("resize") = 1.5)
      .def_property_readonly("_ptr",
                             [](const ak::ArrayBuilder* self) -> size_t {
        return reinterpret_cast<size_t>(self);
      })
      .def("__len__", &ak::ArrayBuilder::length)
      .def("clear", &ak::ArrayBuilder::clear)
      // .def("type", [](const ak::ArrayBuilder& self, const std::map<std::string, std::string>& typestrs) -> std::shared_ptr<ak::Type> {
      //   return unbox_content(::builder_snapshot(self.builder()))->type(typestrs);
      // })
      .def("form", [](const ak::ArrayBuilder& self) -> py::object {
        ::EmptyBuffersContainer container;
        int64_t form_key_id = 0;
        return py::str(self.to_buffers(container, form_key_id));
      })
      .def("to_buffers", [](const ak::ArrayBuilder& self) -> py::object {
        ::NumpyBuffersContainer container;
        int64_t form_key_id = 0;
        std::string form = self.to_buffers(container, form_key_id);
        py::tuple out(3);
        out[0] = py::str(form);
        out[1] = py::int_(self.length());
        out[2] = container.container();
        return out;
      })
      .def("snapshot", [](const ak::ArrayBuilder& self) -> py::object {
        return ::builder_snapshot(self.builder());
      })
      // .def("__getitem__", &getitem<ak::ArrayBuilder>)
      // .def("__iter__", [](const ak::ArrayBuilder& self) -> ak::Iterator {
      //   return ak::Iterator(unbox_content(::builder_snapshot(self.builder())));
      // })
      .def("null", &ak::ArrayBuilder::null)
      .def("boolean", &ak::ArrayBuilder::boolean)
      .def("integer", &ak::ArrayBuilder::integer)
      .def("real", &ak::ArrayBuilder::real)
      .def("complex", &ak::ArrayBuilder::complex)
      .def("datetime", &builder_datetime)
      .def("timedelta", &builder_timedelta)
      .def("bytestring",
           [](ak::ArrayBuilder& self, const py::bytes& x) -> void {
        self.bytestring(x.cast<std::string>());
      })
      .def("string", [](ak::ArrayBuilder& self, const py::str& x) -> void {
        self.string(x.cast<std::string>());
      })
      .def("beginlist", &ak::ArrayBuilder::beginlist)
      .def("endlist", &ak::ArrayBuilder::endlist)
      .def("begintuple", &ak::ArrayBuilder::begintuple)
      .def("index", &ak::ArrayBuilder::index)
      .def("endtuple", &ak::ArrayBuilder::endtuple)
      .def("beginrecord",
           [](ak::ArrayBuilder& self, const py::object& name) -> void {
        if (name.is(py::none())) {
          self.beginrecord();
        }
        else {
          std::string cppname = name.cast<std::string>();
          self.beginrecord_check(cppname.c_str());
        }
      }, py::arg("name") = py::none())
      .def("field", [](ak::ArrayBuilder& self, const std::string& x) -> void {
        self.field_check(x);
      })
      .def("endrecord", &ak::ArrayBuilder::endrecord)
      .def("fromiter", &builder_fromiter)
  );
}
