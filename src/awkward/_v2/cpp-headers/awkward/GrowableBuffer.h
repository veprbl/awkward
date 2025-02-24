// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_GROWABLEBUFFER_H_
#define AWKWARD_GROWABLEBUFFER_H_

#include <cstring>
#include <vector>
#include <memory>
#include <numeric>
#include <cmath>
#include <iostream>
#include <utility>

namespace awkward {

  template <typename PRIMITIVE>
  class Panel {
  public:
    Panel(size_t reserved)
      : ptr_(std::unique_ptr<PRIMITIVE[]>(new PRIMITIVE[reserved])),
        length_(0),
        reserved_(reserved),
        next_(nullptr) {  }

    Panel(std::unique_ptr<PRIMITIVE[]> ptr, size_t length, size_t reserved)
      : ptr_(std::move(ptr)),
        length_(length),
        reserved_(reserved),
        next_(nullptr) {
    }

    PRIMITIVE &operator[](size_t i) {
      return ptr_.get()[i];
    }

    Panel*
    append_panel(size_t reserved) {
      next_ = std::move(std::unique_ptr<Panel>(
        new Panel(reserved)));
      return next_.get();
    }

    /// @brief Inserts one `datum` into the panel.
    void
    fill_panel(PRIMITIVE datum) {
      ptr_.get()[length_++] = datum;
    }

    std::unique_ptr<Panel>&
    next() {
      return next_;
    }

    size_t
    length() {
      if (next_) {
        return next_->length() + length_;
      }
      else {
        return length_;
      }
    }

    size_t
    reserved() {
      return reserved_;
    }

    void
    concatenate_to(PRIMITIVE* to_ptr, size_t offset) const noexcept {
      memcpy(to_ptr + offset, reinterpret_cast<void*>(ptr_.get()), length_ * sizeof(PRIMITIVE));
      if (next_) {
        next_->concatenate_to(to_ptr, offset + length_);
      }
    }

    template<typename TO_PRIMITIVE>
    void
    copy_as(TO_PRIMITIVE* to_ptr, size_t offset) {
      for (size_t i = 0; i < length_; i++) {
        to_ptr[offset++] = static_cast<TO_PRIMITIVE>(ptr_.get()[i]);
      }
      if (next_) {
        next_->copy_as(to_ptr, offset);
      }
    }

  private:
    /// @brief Unique pointer to the panel data.
    std::unique_ptr<PRIMITIVE[]> ptr_;

    /// @brief The length of the panel data.
    size_t length_;

    /// @brief Reserved size of the panel.
    size_t reserved_;

    /// @brief Pointer to the next Panel.
    std::unique_ptr<Panel> next_;
  };

  /// @class GrowableBuffer
  ///
  /// @brief Discontiguous, one-dimensional buffer which consists of
  /// multiple contiguous, one-dimensional arrays that can grow
  /// indefinitely by calling #append.
  ///
  /// The buffer starts by reserving #initial number of slots. When the
  /// number of slots used reaches the number reserved, a new panel is
  /// allocated that has a size equal to #initial.
  ///
  /// When {@link ArrayBuilder#snapshot ArrayBuilder::snapshot} is called,
  /// these buffers are copied to the new Content array.
  template <typename PRIMITIVE>
  class GrowableBuffer {
  public:
    /// @brief Creates an empty GrowableBuffer.
    ///
    /// @param initial Initial size configuration for building a panel.
    static GrowableBuffer<PRIMITIVE>
    empty(int64_t initial) {
      return empty(initial, 0);
    }

    /// @brief Creates an empty GrowableBuffer with a minimum reservation.
    ///
    /// @param initial Initial size configuration for building a panel.
    /// @param minreserve The initial reservation will be the maximum
    /// of `minreserve` and #initial.
    static GrowableBuffer<PRIMITIVE>
    empty(int64_t initial, int64_t minreserve) {
      int64_t actual = initial;
      if (actual < minreserve) {
        actual = minreserve;
      }
      return GrowableBuffer(initial,
        std::unique_ptr<PRIMITIVE[]>(new PRIMITIVE[actual]),
        0, actual);
    }

    /// @brief Creates a GrowableBuffer in which all elements are initialized to `0`.
    ///
    /// @param initial Initial size configuration for building a panel.
    /// @param length The number of elements to initialize (and the
    /// GrowableBuffer's initial #length).
    ///
    /// This is similar to NumPy's
    /// [zeros](https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html).
    static GrowableBuffer<PRIMITIVE>
    zeros(int64_t initial, int64_t length) {
      int64_t actual = initial;
      if (actual < length) {
        actual = length;
      }
      auto ptr = std::unique_ptr<PRIMITIVE[]>(new PRIMITIVE[actual]);
      PRIMITIVE* rawptr = ptr.get();
      for (int64_t i = 0;  i < length;  i++) {
        rawptr[i] = 0;
      }
      return GrowableBuffer(initial, std::move(ptr), length, actual);
    }

    /// @brief Creates a GrowableBuffer in which all elements are initialized
    /// to a given value.
    ///
    /// @param initial Initial size configuration for building a panel.
    /// @param value The initialization value.
    /// @param length The number of elements to initialize (and the
    /// GrowableBuffer's initial #length).
    ///
    /// This is similar to NumPy's
    /// [full](https://docs.scipy.org/doc/numpy/reference/generated/numpy.full.html).
    static GrowableBuffer<PRIMITIVE>
    full(int64_t initial, PRIMITIVE value, int64_t length) {
      int64_t actual = initial;
      if (actual < length) {
        actual = length;
      }
      auto ptr = std::unique_ptr<PRIMITIVE[]>(new PRIMITIVE[actual]);
      PRIMITIVE* rawptr = ptr.get();
      for (int64_t i = 0;  i < length;  i++) {
        rawptr[i] = value;
      }
      return GrowableBuffer<PRIMITIVE>(initial, std::move(ptr), length, actual);
    }

    /// @brief Creates a GrowableBuffer in which the elements are initialized
    /// to numbers counting from `0` to `length`.
    ///
    /// @param initial Initial size configuration for building a panel.
    /// @param length The number of elements to initialize (and the
    /// GrowableBuffer's initial #length).
    ///
    /// This is similar to NumPy's
    /// [arange](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html).
    static GrowableBuffer<PRIMITIVE>
    arange(int64_t initial, int64_t length) {
      int64_t actual = initial;
      if (actual < length) {
        actual = length;
      }
      auto ptr = std::unique_ptr<PRIMITIVE[]>(new PRIMITIVE[actual]);
      PRIMITIVE* rawptr = ptr.get();
      for (int64_t i = 0;  i < length;  i++) {
        rawptr[i] = (PRIMITIVE)i;
      }
      return GrowableBuffer(initial, std::move(ptr), length, actual);
    }

    /// @brief Takes a (possibly multi-panels) GrowableBuffer<PRIMITIVE>
    /// and makes another (one panel) GrowableBuffer<TO_PRIMITIVE>.
    ///
    /// Used to change the data type of buffer content from `PRIMITIVE`
    /// to `TO_PRIMITIVE` for building arrays.
    template<typename TO_PRIMITIVE>
    static GrowableBuffer<TO_PRIMITIVE>
    copy_as(const GrowableBuffer<PRIMITIVE>& other) {
      auto len = other.length();
      int64_t actual = (len < other.initial_) ? other.initial_ : len;

      auto ptr = std::unique_ptr<TO_PRIMITIVE[]>(new TO_PRIMITIVE[actual]);
      TO_PRIMITIVE* rawptr = ptr.get();

      other.panel_->copy_as(rawptr, 0);

      return GrowableBuffer<TO_PRIMITIVE>(actual, std::move(ptr), len, actual);
    }

    /// @brief Creates a GrowableBuffer from a full set of parameters.
    ///
    /// @param initial Initial size configuration for building a panel.
    /// @param ptr Reference-counted pointer to the array buffer.
    /// @param length Currently used number of elements.
    /// @param reserved Currently allocated number of elements.
    ///
    /// Although the #length increments every time #append is called,
    /// it is always less than or equal to #reserved because of
    /// allocations of new panels.
    GrowableBuffer(int64_t initial,
                   std::unique_ptr<PRIMITIVE[]> ptr,
                   int64_t length,
                   int64_t reserved)
        : initial_(initial),
          panel_(std::unique_ptr<Panel<PRIMITIVE>>(new Panel<PRIMITIVE>(std::move(ptr), (size_t)length, (size_t)reserved))),
          ptr_(panel_.get()) {
    }

    /// @brief Creates a GrowableBuffer by allocating a new buffer, taking an
    /// initial #reserved from #initial.
    GrowableBuffer(int64_t initial)
        : GrowableBuffer(initial,
                         std::unique_ptr<PRIMITIVE[]>(new PRIMITIVE[initial]),
                         0,
                         initial) { }

    /// @brief Move constructor
    ///
    /// panel_ is move-only
    GrowableBuffer(GrowableBuffer&& other) noexcept
      : initial_(other.initial_),
        panel_(std::move(other.panel_)),
        ptr_(other.ptr_) { }

    /// @brief Currently used number of elements.
    ///
    /// Although the #length increments every time #append is called,
    /// it is always less than or equal to #reserved because of
    /// allocations of new panels.
    int64_t
    length() const {
      return (int64_t)panel_->length();
    }

    /// @brief Discards accumulated data, the #reserved returns to
    /// initial, and a new #ptr is allocated.
    void
    clear() {
      panel_ = std::move(std::unique_ptr<Panel<PRIMITIVE>>(new Panel<PRIMITIVE>((size_t)initial_)));
      ptr_ = panel_.get();
    }

    /// @brief Inserts one `datum` into the panel, possibly triggering
    /// allocation of a new panel.
    ///
    /// This increases the #length by 1; if the new #length is larger than
    /// #reserved, a new panel will be allocated.
    void
    append(PRIMITIVE datum) {
      if (ptr_->length() == ptr_->reserved()) {
        add_panel((size_t)ceil(ptr_->reserved() * 1.5));
      }
      fill_panel(datum);
    }

    /// @brief Inserts an entire array into the panel(s), possibly triggering
    /// allocation of a new panel.
    ///
    /// If the size is larger than the empty slots in the current panel, then,
    /// first, the empty slots are filled and then a new panel will be allocated
    /// for the rest of the array elements.
    void
    extend(PRIMITIVE* ptr, size_t size) {
      size_t empty_slots = ptr_->reserved() - ptr_->length();
      if (size > empty_slots) {
        for (size_t i = 0; i < empty_slots; i++) {
          fill_panel(ptr[i]);
        }
        add_panel(size - empty_slots > ptr_->reserved() ?
                  size - empty_slots : ptr_->reserved());
        for (size_t i = empty_slots; i < size; i++) {
          fill_panel(ptr[i]);
        }
      }
      else {
        for (size_t i = 0; i < size; i++) {
          fill_panel(ptr[i]);
        }
      }
    }

    /// @brief Like append, but the type signature returns the reference to `PRIMITIVE`.
    PRIMITIVE& append_and_get_ref(PRIMITIVE datum) {
      append(datum);
      return (*ptr_)[ptr_->length() - 1];
    }

    /// @brief Copies and concatenates all accumulated data from multiple panels to one
    /// contiguously allocated `external_pointer`.
    void
    concatenate(PRIMITIVE* external_pointer) const noexcept {
      if (external_pointer) {
        panel_->concatenate_to(external_pointer, 0);
      }
    }

  private:
    /// @brief Inserts one `datum` into the panel.
    void
    fill_panel(PRIMITIVE datum) {
      ptr_->fill_panel(datum);
    }

    /// @brief Creates a new panel with slots equal to #reserved.
    /// and updates the current panel pointer to it.
    void
    add_panel(size_t reserved) {
      ptr_ = ptr_->append_panel(reserved);
    }

    /// @brief Initial size configuration for building a panel.
    int64_t initial_;

    /// @brief The first panel.
    std::unique_ptr<Panel<PRIMITIVE>> panel_;

    /// @brief A pointer to a current panel.
    Panel<PRIMITIVE>* ptr_;
  };
}

#endif // AWKWARD_GROWABLEBUFFER_H_
