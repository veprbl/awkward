// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/Float64Builder.cpp", line)

#include <stdexcept>

#include "awkward/builder/Complex128Builder.h"
#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/UnionBuilder.h"

#include "awkward/builder/Float64Builder.h"

namespace awkward {
  const BuilderPtr
  Float64Builder::fromempty(const int64_t initial) {
    return std::make_shared<Float64Builder>(initial,
                                            GrowableBuffer<double>::empty(initial));
  }

  const BuilderPtr
  Float64Builder::fromint64(const int64_t initial,
                            const GrowableBuffer<int64_t>& old) {
    return std::make_shared<Float64Builder>(
      initial,
      std::move(GrowableBuffer<int64_t>::copy_as<double>(old)));
  }

  Float64Builder::Float64Builder(const int64_t initial,
                                 GrowableBuffer<double> buffer)
      : initial_(initial)
      , buffer_(std::move(buffer)) { }

  GrowableBuffer<double>
  Float64Builder::buffer() {
    // FIXME: swap with an empty buffer
    return std::move(buffer_);
  }

  const std::string
  Float64Builder::classname() const {
    return "Float64Builder";
  }

  const std::string
  Float64Builder::to_buffers(BuffersContainer& container, int64_t& form_key_id) const {
    std::stringstream form_key;
    form_key << "node" << (form_key_id++);

    buffer_.concatenate(
      reinterpret_cast<double*>(
        container.empty_buffer(form_key.str() + "-data",
        buffer_.length() * (int64_t)sizeof(double))));

    return "{\"class\": \"NumpyArray\", \"primitive\": \"float64\", \"form_key\": \""
           + form_key.str() + "\"}";
  }

  int64_t
  Float64Builder::length() const {
    return (int64_t)buffer_.length();
  }

  void
  Float64Builder::clear() {
    buffer_.clear();
  }

  bool
  Float64Builder::active() const {
    return false;
  }

  const BuilderPtr
  Float64Builder::null() {
    BuilderPtr out = OptionBuilder::fromvalids(initial_, shared_from_this());
    out.get()->null();
    return std::move(out);
  }

  const BuilderPtr
  Float64Builder::boolean(bool x) {
    BuilderPtr out = UnionBuilder::fromsingle(initial_, shared_from_this());
    out.get()->boolean(x);
    return std::move(out);
  }

  const BuilderPtr
  Float64Builder::integer(int64_t x) {
    buffer_.append((double)x);
    return nullptr;
  }

  const BuilderPtr
  Float64Builder::real(double x) {
    buffer_.append(x);
    return nullptr;
  }

  const BuilderPtr
  Float64Builder::complex(std::complex<double> x) {
    BuilderPtr out = Complex128Builder::fromfloat64(initial_, std::move(buffer_));
    out.get()->complex(x);
    return std::move(out);
  }

  const BuilderPtr
  Float64Builder::datetime(int64_t x, const std::string& unit) {
    BuilderPtr out = UnionBuilder::fromsingle(initial_, shared_from_this());
    out.get()->datetime(x, unit);
    return std::move(out);
  }

  const BuilderPtr
  Float64Builder::timedelta(int64_t x, const std::string& unit) {
    BuilderPtr out = UnionBuilder::fromsingle(initial_, shared_from_this());
    out.get()->timedelta(x, unit);
    return std::move(out);
  }

  const BuilderPtr
  Float64Builder::string(const char* x, int64_t length, const char* encoding) {
    BuilderPtr out = UnionBuilder::fromsingle(initial_, shared_from_this());
    out.get()->string(x, length, encoding);
    return std::move(out);
  }

  const BuilderPtr
  Float64Builder::beginlist() {
    BuilderPtr out = UnionBuilder::fromsingle(initial_, shared_from_this());
    out.get()->beginlist();
    return std::move(out);
  }

  const BuilderPtr
  Float64Builder::endlist() {
    throw std::invalid_argument(
      std::string("called 'end_list' without 'begin_list' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  Float64Builder::begintuple(int64_t numfields) {
    BuilderPtr out = UnionBuilder::fromsingle(initial_, shared_from_this());
    out.get()->begintuple(numfields);
    return std::move(out);
  }

  const BuilderPtr
  Float64Builder::index(int64_t index) {
    throw std::invalid_argument(
      std::string("called 'index' without 'begin_tuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  Float64Builder::endtuple() {
    throw std::invalid_argument(
      std::string("called 'end_tuple' without 'begin_tuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  Float64Builder::beginrecord(const char* name, bool check) {
    BuilderPtr out = UnionBuilder::fromsingle(initial_, shared_from_this());
    out.get()->beginrecord(name, check);
    return std::move(out);
  }

  void
  Float64Builder::field(const char* key, bool check) {
    throw std::invalid_argument(
      std::string("called 'field' without 'begin_record' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  Float64Builder::endrecord() {
    throw std::invalid_argument(
      std::string("called 'end_record' without 'begin_record' at the same level before it")
      + FILENAME(__LINE__));
  }

}
