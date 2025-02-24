// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_LISTBUILDER_H_
#define AWKWARD_LISTBUILDER_H_

#include <vector>

#include "awkward/common.h"
#include "awkward/GrowableBuffer.h"
#include "awkward/builder/Builder.h"

namespace awkward {

  /// @class ListBuilder
  ///
  /// @brief Builder node that accumulates lists.
  class LIBAWKWARD_EXPORT_SYMBOL ListBuilder: public Builder {
  public:
    /// @brief Create an empty ListBuilder.
    /// @param initial Configuration initial for building an array;
    /// these are passed to every Builder's constructor.
    static const BuilderPtr
      fromempty(const int64_t initial);

    /// @brief Create a ListBuilder from a full set of parameters.
    ///
    /// @param initial Configuration initial for building an array;
    /// these are passed to every Builder's constructor.
    /// @param offsets Contains the accumulated offsets (like
    /// {@link ListOffsetArrayOf#offsets ListOffsetArray::offsets}).
    /// @param content Builder for the data in the nested lists.
    /// @param begun If `true`, the ListBuilder is in a state after
    /// #beginlist and before #endlist and is #active; if `false`,
    /// it is not.
    ListBuilder(const int64_t initial,
                GrowableBuffer<int64_t> offsets,
                const BuilderPtr& content,
                bool begun);

    /// @brief User-friendly name of this class: `"ListBuilder"`.
    const std::string
      classname() const override;

    const std::string
      to_buffers(BuffersContainer& container, int64_t& form_key_id) const override;

    int64_t
      length() const override;

    void
      clear() override;

    /// @copydoc Builder::active()
    ///
    /// Calling #beginlist makes a ListBuilder active; #endlist makes it
    /// inactive.
    bool
      active() const override;

    const BuilderPtr
      null() override;

    const BuilderPtr
      boolean(bool x) override;

    const BuilderPtr
      integer(int64_t x) override;

    const BuilderPtr
      real(double x) override;

    const BuilderPtr
      complex(std::complex<double> x) override;

    const BuilderPtr
      datetime(int64_t x, const std::string& unit) override;

    const BuilderPtr
      timedelta(int64_t x, const std::string& unit) override;

    const BuilderPtr
      string(const char* x, int64_t length, const char* encoding) override;

    const BuilderPtr
      beginlist() override;

    const BuilderPtr
      endlist() override;

    const BuilderPtr
      begintuple(int64_t numfields) override;

    const BuilderPtr
      index(int64_t index) override;

    const BuilderPtr
      endtuple() override;

    const BuilderPtr
      beginrecord(const char* name, bool check) override;

    void
      field(const char* key, bool check) override;

    const BuilderPtr
      endrecord() override;

    const int64_t
      initial() const { return initial_; }

    const GrowableBuffer<int64_t>& buffer() const { return offsets_; }

    const BuilderPtr builder() const { return content_; }

    bool begun() { return begun_; }

    void
      maybeupdate(const BuilderPtr builder);

  private:
    const int64_t initial_;
    GrowableBuffer<int64_t> offsets_;
    BuilderPtr content_;
    bool begun_;
  };
}

#endif // AWKWARD_LISTBUILDER_H_
