#pragma once

#include <cstdint>
#include <functional>
#include <memory>

#include <c10/util/hash.h>

namespace torch {
namespace autograd {

struct Node;

/// Represents a particular input of a function.
/*
Autograd graph的edge和vertices

如果将PyTorch的autograd系统看作是一个图（graph）的话，那么每个Function实例就是graph中的节点（nodes/vertices），
各个Function实例之间则是通过Edge连接的。
Edge是个结构体，通过 (Function, input_nr) 的配对来代表graph中的edge：
*/
struct Edge {
  Edge() noexcept : function(nullptr), input_nr(0) {}

  Edge(std::shared_ptr<Node> function_, uint32_t input_nr_) noexcept
      : function(std::move(function_)), input_nr(input_nr_) {}

  /// Convenience method to test if an edge is valid.
  bool is_valid() const noexcept {
    return function != nullptr;
  }

  // Required for use in associative containers.
  bool operator==(const Edge& other) const noexcept {
    return this->function == other.function && this->input_nr == other.input_nr;
  }

  bool operator!=(const Edge& other) const noexcept {
    return !(*this == other);
  }

  /// The function this `Edge` points to.
  std::shared_ptr<Node> function; // 指向目标的Node

  /// The identifier of a particular input to the function.
  uint32_t input_nr; //指定本Edge是function的第几个输入
};
} // namespace autograd
} // namespace torch

// The idiomatic way of enabling use of a custom type as the key of hash
// containers in C++11. This method removes the requirement of having to pass
// a custom hasher to std::unordered_{map, set}.
// See http://en.cppreference.com/w/cpp/utility/hash for more information.
namespace std {
template <>
struct hash<torch::autograd::Edge> {
  // These type aliases are required by the standard.
  using argument_type = torch::autograd::Edge;
  using return_type = size_t;
  return_type operator()(const argument_type& edge) const noexcept {
    return c10::get_hash(edge.function, edge.input_nr);
  }
};
} // namespace std
