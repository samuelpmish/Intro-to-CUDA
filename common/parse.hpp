#pragma once

#include <tuple>
#include <array>
#include <vector>
#include <string>
#include <iostream>

namespace impl {

    template < typename T >
    struct is_arraylike{ constexpr operator bool() { return false; } };

    template < typename T >
    struct is_arraylike< std::vector<T> >{ constexpr operator bool() { return true; } };

    template < typename T, std::size_t n >
    struct is_arraylike< std::array<T, n> >{ constexpr operator bool() { return true; } };

////////////////////////////////////////////////////////////////////////////////

    template < typename T >
    struct scalar;

    template < typename T >
    struct scalar< std::vector<T> >{ using type = T; };

    template < typename T, std::size_t n >
    struct scalar< std::array<T, n>>{ using type = T; };

////////////////////////////////////////////////////////////////////////////////

    template < typename T >
    std::size_t size(T) { return 0; }

    template < typename T >
    std::size_t size(std::vector< T > x) { return x->size(); }

    template < typename T, std::size_t n >
    std::size_t size(std::array< T, n >) { return n; }

}

////////////////////////////////////////////////////////////////////////////////

template <typename ... T>
void parse(int argc, char * argv[], std::tuple< std::string, T * > ... flags) {

  int i = 1;

  auto get_next_arg = [&](std::string flagname) {
    if (i < argc) {
      return argv[i++];
    } else {
      std::cout << "error: expected an argument after " << flagname << std::endl;
      exit(1);
    }
  };

  auto convert_and_assign_value = [&](auto * value_ptr, std::string arg) {

    using value_t = std::remove_reference_t<decltype(*value_ptr)>;

    if constexpr (std::is_same_v<value_t, bool>) {
      *value_ptr = true;
    }

    if constexpr (std::is_integral_v<value_t> && !std::is_same_v<value_t, bool>) {
      *value_ptr = std::atoi(get_next_arg(arg));
    }

    if constexpr (std::is_floating_point_v<value_t>) {
      *value_ptr = std::atof(get_next_arg(arg));
    }

    if constexpr (std::is_same_v<value_t, std::string>) {
      *value_ptr = get_next_arg(arg);
    }

  };

  auto assign_value_if_match = [&](auto flag, std::string arg) {
    
    std::string pattern = std::get<0>(flag);
    auto * value_ptr = std::get<1>(flag);
    using value_t = std::remove_reference_t<decltype(*value_ptr)>;

    if (pattern == arg) {

      convert_and_assign_value(value_ptr, arg);

      if constexpr (impl::is_arraylike<value_t>{}) {
        using scalar_t = typename impl::scalar<value_t>::type;
        std::size_t sz = impl::size(*value_ptr);
        auto & arr = *value_ptr;

        if (i+sz > argc) {
          std::cout << "error: expected " << sz << " arguments after " << pattern << std::endl;
          exit(1);
        }

        for (int j = 0; j < sz; j++) {
          convert_and_assign_value(&arr[j], arg);
        }
      }

      return 1;

    } else {

      return 0;

    }

  };

  while (i < argc) {
    std::string arg = argv[i++];

    int matches = (assign_value_if_match(flags, arg) || ...);

    if (matches == 0) {
        std::cout << "error: argument \"" << arg << "\" did not match any known flags." << std::endl;
    }

    if (matches >= 2) {
        std::cout << "error: argument \"" << arg << "\" matched more than once, remove duplicate flag entries." << std::endl;
    }
  }

}