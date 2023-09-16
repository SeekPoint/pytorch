#include <torch/data/samplers/distributed.h>
#include <torch/serialize/archive.h>
#include <torch/types.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

namespace torch {
namespace data {
namespace samplers {
'''
3.4.2 实现
类的具体实现位于：torch\csrc\api\src\data\samplers\distributed.cpp

3.4.2.1 DistributedRandomSampler
我们首先看看DistributedRandomSampler。
其作用就是依据本worker 的 rank_获取打乱的index。我们按照逻辑顺序讲解各个函数。

    初始化时候会调用 reset(size_) 进行 shuffle。
    reset 函数作用是让sampler指向一组新的indices：
        首先调用 populate_indices() 设置对应本rank的起始index，终止index。
        populate_indices 函数之中，会对数据index 进行设置，即配置了 all_indices_，也同时配置了本rank的起始index，终止index。
        然后对 all_indices_ 进行shuffle。
    next 函数就相对简单了，因为主要工作被reset做了，所以此时数据已经被随机打乱了，所以找到起始位置，返回数据中对行数即可。

因为下面用到了 iota 函数，可能有的同学不熟悉，这里说明下iota的作用：

    std::vector<int> test;
    test.resize(10);
    std::iota(test.begin(), test.end(), 5);// 将从 5 开始的 10 次递增值赋值给 test

    //test结果:5 6 7 8 9 10 11 12 13 14

具体代码如下：
'''
DistributedRandomSampler::DistributedRandomSampler(
    size_t size,
    size_t num_replicas,
    size_t rank,
    bool allow_duplicates)
    : DistributedSampler(size, num_replicas, rank, allow_duplicates),
      begin_index_(0),
      end_index_(0),
      sample_index_(0) {
  // shuffle first time.
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  reset(size_);
}

// 注意，每次加载新epoch时候，都要调用reset，因此对于next函数来说，工作已经很小
optional<std::vector<size_t>> DistributedRandomSampler::next(
    size_t batch_size) {
  if (sample_index_ == end_index_) {  // 已经提取完数据
    return nullopt;
  }

  size_t end = sample_index_ + batch_size;  // 本次迭代的终止位置
  if (end > end_index_) {
    end = end_index_;
  }

  auto iter = all_indices_.begin();  // 因为此时数据已经被随机打乱了，找到起始位置即可
  std::vector<size_t> res(iter + sample_index_, iter + end);  // 从所有数据中提取前面若干行
  sample_index_ = end;
  return res;
}

//3.3.3 C++
//我们也可以提前看看在C++ 代码的DistributedRandomSampler，这是C++ API，也起到python同样作用。
//我们可以看到设置种子和shuffle如下：

//// 每次加载新epoch时候，都要调用reset
void DistributedRandomSampler::reset(optional<size_t> new_size) {
  size_ = new_size.value_or(size_);
  populate_indices();

  std::mt19937 rand(epoch_);

  // 对于数据进行shuffle
  std::shuffle(all_indices_.begin(), all_indices_.end(), rand);
  sample_index_ = begin_index_;
}

void DistributedRandomSampler::populate_indices() {
  size_t num_local_samples = local_sample_count();
  // 得到样本数量
  size_t sample_count =
      num_replicas_ == 1 ? size_ : num_local_samples * num_replicas_;
  all_indices_.resize(sample_count);

  // std::iota 的作用是用顺序递增的值赋值指定范围内的元素
  // 这里是给all_indices_设置从0开始到sample_count这些数值
  std::iota(std::begin(all_indices_), std::end(all_indices_), 0);

  // 如果sample count大于size_，则需要给多出来的那些index再赋一些数值
  for (size_t i = size_; i < sample_count; ++i) {
    // we may have added duplicate samples to make all
    // replicas to have the same number of samples.
    all_indices_[i] = i - size_;
  }
  begin_index_ = rank_ * num_local_samples;  // 对应本rank的起始index
  end_index_ = begin_index_ + num_local_samples;  // 对应本rank的终止index
  sample_index_ = begin_index_;
}

void DistributedRandomSampler::save(serialize::OutputArchive& archive) const {
  archive.write(
      "sample_index_",
      torch::tensor(static_cast<int64_t>(sample_index_)),
      /*is_buffer=*/true);
  archive.write(
      "epoch_",
      torch::tensor(static_cast<int64_t>(epoch_)),
      /*is_buffer=*/true);
}

void DistributedRandomSampler::load(serialize::InputArchive& archive) {
  auto tensor = torch::empty(1, torch::kInt64);
  archive.read("epoch_", tensor, /*is_buffer=*/true);
  epoch_ = tensor.item<int64_t>();
  // call reset() after loading epoch_ to populate indices.
  reset(size_);

  tensor = torch::empty(1, torch::kInt64);
  archive.read("sample_index_", tensor, /*is_buffer=*/true);
  sample_index_ = tensor.item<int64_t>();
}

size_t DistributedRandomSampler::index() const noexcept {
  return sample_index_;
}

/*
3.4.2.2 DistributedSequentialSampler
然后看看 DistributedSequentialSampler。
其作用就是依据本worker 的 rank_获取顺序的index。我们按照逻辑顺序讲解各个函数。
    reset 函数就简单多了，使用populate_indices按照顺序设置index即可。
    next 函数就相对复杂，不但要顺序返回index，还需要设置下次的起始位置。
*/
DistributedSequentialSampler::DistributedSequentialSampler(
    size_t size,
    size_t num_replicas,
    size_t rank,
    bool allow_duplicates)
    : DistributedSampler(size, num_replicas, rank, allow_duplicates),
      begin_index_(0),
      end_index_(0),
      sample_index_(0) {
  populate_indices();   // 这里会设定本rank对应的起始位置
}

optional<std::vector<size_t>> DistributedSequentialSampler::next(
    size_t batch_size) {
  if (sample_index_ == end_index_) {  // 已经循环结束
    return nullopt;
  }

  size_t end = sample_index_ + batch_size;  // 本次的终止行
  if (end > end_index_) {
    end = end_index_;
  }

  std::vector<size_t> res(end - sample_index_);  // 返回的vector大小
  // 给res设置从sample_index_开始递增(end - sample_index_)这么大的这些数值，这就是顺序返回了index
  std::iota(std::begin(res), std::end(res), sample_index_);
  if (end >= size_) {
    for (size_t& index : res) {  //遍历 vector，得到本次的index
      index = index % size_;
    }
  }
  sample_index_ = end;  // 设置下次开始行
  return res;
}

void DistributedSequentialSampler::reset(optional<size_t> new_size) {
  size_t size = new_size.value_or(size_);
  if (size != size_) {
    size_ = size;
    populate_indices();
  } else {
    sample_index_ = begin_index_;
  }
}

void DistributedSequentialSampler::populate_indices() {
  begin_index_ = rank_ * local_sample_count();  // 本rank对应的起始位置
  end_index_ = begin_index_ + local_sample_count();
  sample_index_ = begin_index_;
}

void DistributedSequentialSampler::save(
    serialize::OutputArchive& archive) const {
  archive.write(
      "sample_index_",
      torch::tensor(static_cast<int64_t>(sample_index_)),
      /*is_buffer=*/true);
}

void DistributedSequentialSampler::load(serialize::InputArchive& archive) {
  auto tensor = torch::empty(1, torch::kInt64);
  archive.read("sample_index_", tensor, /*is_buffer=*/true);
  sample_index_ = tensor.item<int64_t>();
}

size_t DistributedSequentialSampler::index() const noexcept {
  return sample_index_;
}

} // namespace samplers
} // namespace data
} // namespace torch
