#pragma once

#include <random>

int random_integer(int min, int max) {
  static std::random_device rd;
  static std::mt19937 g(rd());
  std::uniform_int_distribution< int > dist(min, max);
  return dist(g);
}

double random_real(double min, double max) {
  static std::random_device rd;
  static std::mt19937 g(rd());
  std::uniform_real_distribution< double > dist(min, max);
  return dist(g);
}