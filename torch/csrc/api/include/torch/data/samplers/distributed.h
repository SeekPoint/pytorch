#pragma once

#include <torch/csrc/Export.h>
#include <torch/data/samplers/base.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace serialize {
class OutputArchive;
class InputArchive;
} // namespace serialize
} // namespace torch

namespace torch {
namespace data {
namespace samplers {
/*
3.4 Sampler in C++
因为某些公司是C++开发，他们也有迫切的使用pytorch的需求，所以pytorch也提供了C++ API，我们接下来就看看如何实现。

3.4.1 定义
其类定义在：torch\csrc\api\include\torch\data\samplers\distributed.h

我们可以看到，DistributedSampler 是基类，主要成员变量是：
    size_t size_ 文件大小
    size_t num_replicas_ 副本数目
    size_t rank_ 本sampler 对应哪个进程或者GPU
    size_t epoch 本次训练的epoch
    bool allow_duplicates_ 是否允许备份
接下来是两个派生类： DistributedRandomSampler 和 DistributedSequentialSampler。

*/
/// A `Sampler` that selects a subset of indices to sample from and defines a
/// sampling behavior. In a distributed setting, this selects a subset of the
/// indices depending on the provided num_replicas and rank parameters. The
/// `Sampler` performs a rounding operation based on the `allow_duplicates`
/// parameter to decide the local sample count.
template <typename BatchRequest = std::vector<size_t>>
class DistributedSampler : public Sampler<BatchRequest> {
 public:
  DistributedSampler(
      size_t size,
      size_t num_replicas = 1,
      size_t rank = 0,
      bool allow_duplicates = true)
      : size_(size),
        num_replicas_(num_replicas),
        rank_(rank),
        epoch_(0),
        allow_duplicates_(allow_duplicates) {}

  /// Set the epoch for the current enumeration. This can be used to alter the
  /// sample selection and shuffling behavior.
  void set_epoch(size_t epoch) {
    epoch_ = epoch;
  }

  size_t epoch() const {
    return epoch_;
  }

 protected:
  size_t local_sample_count() {
    if (allow_duplicates_) {
      return (size_ + num_replicas_ - 1) / num_replicas_;
    } else {
      return size_ / num_replicas_;
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  size_t size_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  size_t num_replicas_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  size_t rank_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  size_t epoch_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool allow_duplicates_;
};

/// Select samples randomly. The sampling order is shuffled at each `reset()`
/// call.
class TORCH_API DistributedRandomSampler : public DistributedSampler<> {
 public:
  DistributedRandomSampler(
      size_t size,
      size_t num_replicas = 1,
      size_t rank = 0,
      bool allow_duplicates = true);

  /// Resets the `DistributedRandomSampler` to a new set of indices.
  void reset(optional<size_t> new_size = nullopt) override;

  /// Returns the next batch of indices.
  optional<std::vector<size_t>> next(size_t batch_size) override;

  /// Serializes the `DistributedRandomSampler` to the `archive`.
  void save(serialize::OutputArchive& archive) const override;

  /// Deserializes the `DistributedRandomSampler` from the `archive`.
  void load(serialize::InputArchive& archive) override;

  /// Returns the current index of the `DistributedRandomSampler`.
  size_t index() const noexcept;

 private:
  void populate_indices();

  size_t begin_index_;
  size_t end_index_;
  size_t sample_index_;
  std::vector<size_t> all_indices_;
};

/// Select samples sequentially.
class TORCH_API DistributedSequentialSampler : public DistributedSampler<> {
 public:
  DistributedSequentialSampler(
      size_t size,
      size_t num_replicas = 1,
      size_t rank = 0,
      bool allow_duplicates = true);

  /// Resets the `DistributedSequentialSampler` to a new set of indices.
  void reset(optional<size_t> new_size = nullopt) override;

  /// Returns the next batch of indices.
  optional<std::vector<size_t>> next(size_t batch_size) override;

  /// Serializes the `DistributedSequentialSampler` to the `archive`.
  void save(serialize::OutputArchive& archive) const override;

  /// Deserializes the `DistributedSequentialSampler` from the `archive`.
  void load(serialize::InputArchive& archive) override;

  /// Returns the current index of the `DistributedSequentialSampler`.
  size_t index() const noexcept;

 private:
  void populate_indices();

  size_t begin_index_;
  size_t end_index_;
  size_t sample_index_;
  std::vector<size_t> all_indices_;
};

} // namespace samplers
} // namespace data
} // namespace torch
