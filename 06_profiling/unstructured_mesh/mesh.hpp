#pragma once

#include <unordered_map>
#include <array>
#include <vector>
#include <iostream>
#include <cinttypes>
#include <random>
#include <algorithm>

#include "vec.hpp"
#include "morton.hpp"

////////////////////////////////////////////////////////////////////////////////

using vec2i = vec<2, int>;
using vec3i = vec<3, int>;
using vec4i = vec<4, int>;

struct ivec_hash {
  template < uint32_t n >
  std::size_t operator()(const vec<n, int> & a) const {
    std::size_t h = 0;
    for (uint32_t i = 0; i < n; i++) {
      h ^= std::hash<int>{}(a[i])  + 0x9e3779b9 + (h << 6) + (h >> 2); 
    }
    return h;
  }   
};

vec3 mean(vec3 x, vec3 y) {
  return vec3{0.5 * (x[0] + y[0]), 0.5 * (x[1] + y[1]), 0.5 * (x[2] + y[2])};
}

vec3 mean(vec3 u, vec3 v, vec3 w) {
  constexpr double one_third = 1.0 / 3.0;
  return vec3{
    one_third * (u[0] + v[0] + w[0]), 
    one_third * (u[1] + v[1] + w[1]), 
    one_third * (u[2] + v[2] + w[2])
  };
}

struct TriangleMesh {
  std::vector<vec3> vertices;
  std::vector<vec3i> triangles;
};

TriangleMesh subdivide(TriangleMesh & mesh) {
  TriangleMesh output{};
  output.triangles.reserve(4 * mesh.triangles.size());

  output.vertices = mesh.vertices;
  auto mid = [&](int i, int j) {
    static std::unordered_map< vec2i, int, ivec_hash > midpoint_index;
    vec2i key = {std::min(i,j), std::max(i,j)};
    if (midpoint_index.count(key)) {
      return midpoint_index[key];
    } else {
      int vertex_id = midpoint_index[key] = output.vertices.size();
      output.vertices.push_back(normalize(mean(mesh.vertices[i], mesh.vertices[j])));
      return vertex_id;
    }
  };

  for (auto tri : mesh.triangles) {
    int i = tri[0];
    int j = tri[1];
    int k = tri[2];
    int m_ij = mid(i, j);
    int m_jk = mid(j, k);
    int m_ki = mid(k, i);

    output.triangles.push_back(vec3i{i, m_ij, m_ki});
    output.triangles.push_back(vec3i{m_ki, m_ij, m_jk});
    output.triangles.push_back(vec3i{m_jk, m_ij, j});
    output.triangles.push_back(vec3i{m_jk, k, m_ki});
  }

  return output;
}

TriangleMesh icosphere(int subdivision) {
  vec3 icosahedron_vertices[12] = {{0., 0., -1.}, {0., 0., 1.}, {-0.894427, 0., -0.447214}, {0.894427, 0., 0.447214}, {0.723607, -0.525731, -0.447214}, {0.723607, 0.525731, -0.447214}, {-0.723607, -0.525731, 0.447214}, {-0.723607, 0.525731, 0.447214}, {-0.276393, -0.850651, -0.447214}, {-0.276393, 0.850651, -0.447214}, {0.276393, -0.850651, 0.447214}, {0.276393, 0.850651, 0.447214}};
  vec3i icosahedron_triangles[20] = {{5, 3, 4}, {3, 5, 11}, {10, 8, 4}, {8, 10, 6}, {1, 10, 3}, {1, 3, 11}, {9, 5, 0}, {5, 4, 0}, {9, 0, 2}, {0, 8, 2}, {2, 7, 9}, {7, 2, 6}, {11, 7, 1}, {1, 7, 6}, {0, 4, 8}, {2, 8, 6}, {3, 10, 4}, {10, 1, 6}, {9, 11, 5}, {11, 9, 7}};

  TriangleMesh mesh{
    std::vector< vec3 >(icosahedron_vertices, icosahedron_vertices+12),
    std::vector< vec3i >(icosahedron_triangles, icosahedron_triangles + 20)
  };

  for (int k = 0; k < subdivision; k++) {
    mesh = subdivide(mesh);
  }

  return mesh;
}

////////////////////////////////////////////////////////////////////////////////

enum Ordering { RANDOMIZED, MORTON };

TriangleMesh renumber(const TriangleMesh & mesh, Ordering o) {

  std::vector< int > vertex_permutation(mesh.vertices.size());
  std::vector< int > triangle_permutation(mesh.triangles.size());

  if (o == Ordering::RANDOMIZED) {
    static std::random_device rd;
    static std::mt19937 g(rd());

    for (int i = 0; i < mesh.vertices.size(); i++) {
      vertex_permutation[i] = i;
    }

    for (int i = 0; i < mesh.triangles.size(); i++) {
      triangle_permutation[i] = i;
    }

    std::shuffle(vertex_permutation.begin(), vertex_permutation.end(), g);
    std::shuffle(triangle_permutation.begin(), triangle_permutation.end(), g);
  }
  
  if (o == Ordering::MORTON) {
    std::vector< uint64_t > vertex_morton_codes(mesh.vertices.size());
    for (int i = 0; i < mesh.vertices.size(); i++) {
      vec3 v = mesh.vertices[i];
      vertex_permutation[i] = i;
      vertex_morton_codes[i] = morton::encode(v[0], v[1], v[2]);
    }
    std::sort(
      vertex_permutation.begin(), 
      vertex_permutation.end(), 
      [&](int i, int j){
      return vertex_morton_codes[i] < vertex_morton_codes[j];
      }
    );

    std::vector< uint64_t > triangle_morton_codes(mesh.triangles.size());
    for (int i = 0; i < mesh.triangles.size(); i++) {
      vec3i tri = mesh.triangles[i];
      vec3 c = mean(mesh.vertices[tri[0]], mesh.vertices[tri[1]], mesh.vertices[tri[2]]);
      triangle_permutation[i] = i;
      triangle_morton_codes[i] = morton::encode(c[0], c[1], c[2]);
    }

    std::sort(
      triangle_permutation.begin(), 
      triangle_permutation.end(), 
      [&](int i, int j){
        return triangle_morton_codes[i] < triangle_morton_codes[j];
      }
    );
  }

  TriangleMesh output = mesh;
  for (int i = 0; i < mesh.vertices.size(); i++) {
    output.vertices[vertex_permutation[i]] = mesh.vertices[i];
  }

  for (int i = 0; i < mesh.triangles.size(); i++) {
    vec3i tri = mesh.triangles[triangle_permutation[i]];
    tri[0] = vertex_permutation[tri[0]];
    tri[1] = vertex_permutation[tri[1]];
    tri[2] = vertex_permutation[tri[2]];
    output.triangles[i] = tri;
  }
  return output;
  
};