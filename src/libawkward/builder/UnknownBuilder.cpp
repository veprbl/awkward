// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/UnknownBuilder.cpp", line)

#include <stdexcept>

#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/BoolBuilder.h"
#include "awkward/builder/DatetimeBuilder.h"
#include "awkward/builder/Int64Builder.h"
#include "awkward/builder/Float64Builder.h"
#include "awkward/builder/Complex128Builder.h"
#include "awkward/builder/StringBuilder.h"
#include "awkward/builder/ListBuilder.h"
#include "awkward/builder/TupleBuilder.h"
#include "awkward/builder/RecordBuilder.h"

#include "awkward/builder/UnknownBuilder.h"

namespace awkward {
  const BuilderPtr
  UnknownBuilder::fromempty(const int64_t initial) {
    return std::make_shared<UnknownBuilder>(initial, 0);
  }

  UnknownBuilder::UnknownBuilder(const int64_t initial,
                                 int64_t nullcount)
      : initial_(initial)
      , nullcount_(nullcount) { }

  const std::string
  UnknownBuilder::classname() const {
    return "UnknownBuilder";
  };

  const std::string
  UnknownBuilder::to_buffers(BuffersContainer& container, int64_t& form_key_id) const {
    if (nullcount_ == 0) {
      std::stringstream form_key;
      form_key << "node" << (form_key_id++);

      return "{\"class\": \"EmptyArray\", \"form_key\": \""
             + form_key.str() + "\"}";
    }
    else {
      std::stringstream outer_form_key;
      std::stringstream inner_form_key;
      outer_form_key << "node" << (form_key_id++);
      inner_form_key << "node" << (form_key_id++);

      container.full_buffer(outer_form_key.str() + "-index", nullcount_, -1, "i8");

      return std::string("{\"class\": \"IndexedOptionArray\", \"index\": \"i64\", \"content\": ")
             + "{\"class\": \"EmptyArray\", \"form_key\": \""
             + inner_form_key.str() + "\"}, \"form_key\": \""
             + outer_form_key.str() + "\"}";
    }
  }

  int64_t
  UnknownBuilder::length() const {
    return nullcount_;
  }

  void
  UnknownBuilder::clear() {
    nullcount_ = 0;
  }

  bool
  UnknownBuilder::active() const {
    return false;
  }

  const BuilderPtr
  UnknownBuilder::null() {
    nullcount_++;
    return shared_from_this();
  }

  const BuilderPtr
  UnknownBuilder::boolean(bool x) {
    BuilderPtr out = BoolBuilder::fromempty(initial_);
    if (nullcount_ != 0) {
      out = OptionBuilder::fromnulls(initial_, nullcount_, out);
    }
    out.get()->boolean(x);
    return std::move(out);
  }

  const BuilderPtr
  UnknownBuilder::integer(int64_t x) {
    BuilderPtr out = Int64Builder::fromempty(initial_);
    if (nullcount_ != 0) {
      out = OptionBuilder::fromnulls(initial_, nullcount_, out);
    }
    out.get()->integer(x);
    return std::move(out);
  }

  const BuilderPtr
  UnknownBuilder::real(double x) {
    BuilderPtr out = Float64Builder::fromempty(initial_);
    if (nullcount_ != 0) {
      out = OptionBuilder::fromnulls(initial_, nullcount_, out);
    }
    out.get()->real(x);
    return std::move(out);
  }

  const BuilderPtr
  UnknownBuilder::complex(std::complex<double> x) {
    BuilderPtr out = Complex128Builder::fromempty(initial_);
    if (nullcount_ != 0) {
      out = OptionBuilder::fromnulls(initial_, nullcount_, out);
    }
    out.get()->complex(x);
    return std::move(out);
  }

  const BuilderPtr
  UnknownBuilder::datetime(int64_t x, const std::string& unit) {
    BuilderPtr out = DatetimeBuilder::fromempty(initial_, unit);
    if (nullcount_ != 0) {
      out = OptionBuilder::fromnulls(initial_, nullcount_, out);
    }
    out.get()->datetime(x, unit);
    return std::move(out);
  }

  const BuilderPtr
  UnknownBuilder::timedelta(int64_t x, const std::string& unit) {
    BuilderPtr out = DatetimeBuilder::fromempty(initial_, unit);
    if (nullcount_ != 0) {
      out = OptionBuilder::fromnulls(initial_, nullcount_, out);
    }
    out.get()->timedelta(x, unit);
    return std::move(out);
  }

  const BuilderPtr
  UnknownBuilder::string(const char* x, int64_t length, const char* encoding) {
    BuilderPtr out = StringBuilder::fromempty(initial_, encoding);
    if (nullcount_ != 0) {
      out = OptionBuilder::fromnulls(initial_, nullcount_, out);
    }
    out.get()->string(x, length, encoding);
    return std::move(out);
  }

  const BuilderPtr
  UnknownBuilder::beginlist() {
    BuilderPtr out = ListBuilder::fromempty(initial_);
    if (nullcount_ != 0) {
      out = OptionBuilder::fromnulls(initial_, nullcount_, out);
    }
    out.get()->beginlist();
    return std::move(out);
  }

  const BuilderPtr
  UnknownBuilder::endlist() {
    throw std::invalid_argument(
      std::string("called 'end_list' without 'begin_list' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  UnknownBuilder::begintuple(int64_t numfields) {
    BuilderPtr out = TupleBuilder::fromempty(initial_);
    if (nullcount_ != 0) {
      out = OptionBuilder::fromnulls(initial_, nullcount_, out);
    }
    out.get()->begintuple(numfields);
    return std::move(out);
  }

  const BuilderPtr
  UnknownBuilder::index(int64_t index) {
    throw std::invalid_argument(
      std::string("called 'index' without 'begin_tuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  UnknownBuilder::endtuple() {
    throw std::invalid_argument(
      std::string("called 'end_tuple' without 'begin_tuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  UnknownBuilder::beginrecord(const char* name, bool check) {
    BuilderPtr out = RecordBuilder::fromempty(initial_);
    if (nullcount_ != 0) {
      out = OptionBuilder::fromnulls(initial_, nullcount_, out);
    }
    out.get()->beginrecord(name, check);
    return std::move(out);
  }

  void
  UnknownBuilder::field(const char* key, bool check) {
    throw std::invalid_argument(
      std::string("called 'field' without 'begin_record' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  UnknownBuilder::endrecord() {
    throw std::invalid_argument(
      std::string("called 'end_record' without 'begin_record' at the same level before it")
      + FILENAME(__LINE__));
  }

}
