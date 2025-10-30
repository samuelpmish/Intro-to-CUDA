// exercise: start by 

#include <unordered_map>
#include <array>
#include <vector>
#include <iostream>
#include <cinttypes>
#include <random>
#include <algorithm>

#include "mesh.hpp"
#include "binary_io.hpp"

#include "timer.hpp"

////////////////////////////////////////////////////////////////////////////////

timer stopwatch;

////////////////////////////////////////////////////////////////////////////////

void z_rotate_cpu(TriangleMesh & mesh) {
  stopwatch.start();
  for (int i = 0; i < mesh.vertices.size(); i++) {
    vec3 v = mesh.vertices[i];
    mesh.vertices[i] = vec3{-v[1], v[0], v[2]};
  }
  stopwatch.stop();

  std::cout << "(cpu) z_rotation calculation took " << stopwatch.elapsed() * 1000.0f << " ms" << std::endl;
}

std::vector< vec3 > calculate_normals_cpu(const TriangleMesh & mesh) {
  std::vector< vec3 > normals(mesh.triangles.size());

  stopwatch.start();
  for (int i = 0; i < mesh.triangles.size(); i++) {
    vec3i tri = mesh.triangles[i];
    vec3 v0 = mesh.vertices[tri[0]];
    vec3 v1 = mesh.vertices[tri[1]];
    vec3 v2 = mesh.vertices[tri[2]];
    normals[i] = normalize(cross(v1 - v0, v2 - v0));
  }
  stopwatch.stop();

  std::cout << "(cpu) normal calculation took " << stopwatch.elapsed() * 1000.0f << " ms" << std::endl;

  return normals;
}

////////////////////////////////////////////////////////////////////////////////

__global__ void z_rotate_kernel(vec3 * vertices, int n) {

  // TODO: implement the calculation performed in z_rotate_cpu
  
}

__global__ void calculate_normals_kernel(const vec3 * vertices, const vec3i * triangles, vec3 * normals, int n) {

  // TODO: implement the calculation performed in calculate_normals_cpu

}

void z_rotate_gpu(TriangleMesh & mesh) {
  vec3 * d_vertices;
  cudaMalloc(&d_vertices, sizeof(vec3) * mesh.vertices.size());

  cudaMemcpy(d_vertices, &mesh.vertices[0], sizeof(vec3) * mesh.vertices.size(), cudaMemcpyHostToDevice);

  stopwatch.start();
  {
    int threads_per_block = 256;
    int blocks = (mesh.vertices.size() + threads_per_block - 1) / threads_per_block;
    z_rotate_kernel<<< blocks, threads_per_block >>>(d_vertices, mesh.vertices.size());
    cudaDeviceSynchronize();
  }
  stopwatch.stop();
  std::cout << "(gpu) z_rotation calculation took " << stopwatch.elapsed() * 1000.0f << " ms" << std::endl;

  cudaFree(d_vertices);
}

void calculate_normals_gpu(const TriangleMesh & mesh) {

  vec3 * d_vertices;
  vec3 * d_normals;
  vec3i * d_triangles;
  cudaMalloc(&d_vertices, sizeof(vec3) * mesh.vertices.size());
  cudaMalloc(&d_normals, sizeof(vec3) * mesh.triangles.size());
  cudaMalloc(&d_triangles, sizeof(vec3i) * mesh.triangles.size());

  cudaMemcpy(d_vertices, &mesh.vertices[0], sizeof(vec3) * mesh.vertices.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_triangles, &mesh.triangles[0], sizeof(vec3i) * mesh.triangles.size(), cudaMemcpyHostToDevice);

  stopwatch.start();
  {
    int threads_per_block = 256;
    int blocks = (mesh.triangles.size() + threads_per_block - 1) / threads_per_block;
    calculate_normals_kernel<<< blocks, threads_per_block >>>(
      d_vertices, 
      d_triangles, 
      d_normals, 
      mesh.triangles.size()
    );
    cudaDeviceSynchronize();
  }
  stopwatch.stop();
  std::cout << "(gpu) normal calculation took " << stopwatch.elapsed() * 1000.0f << " ms" << std::endl;

  cudaFree(d_vertices);
  cudaFree(d_normals);
  cudaFree(d_triangles);

}

////////////////////////////////////////////////////////////////////////////////

int main() {

  std::cout << "generating mesh ... ";
  TriangleMesh mesh = icosphere(9);
  std::cout << "done" << std::endl;
  std::cout << "Mesh has " << mesh.vertices.size() << " vertices and ";
  std::cout << mesh.triangles.size() << " triangles" << std::endl << std::endl;

  z_rotate_cpu(mesh);
  calculate_normals_cpu(mesh);

  z_rotate_gpu(mesh);
  calculate_normals_gpu(mesh);

  std::cout << std::endl << "randomizing mesh numbering ... ";
  mesh = renumber(mesh, Ordering::RANDOMIZED);
  std::cout << "done" << std::endl << std::endl;

  z_rotate_cpu(mesh);
  calculate_normals_cpu(mesh);

  z_rotate_gpu(mesh);
  calculate_normals_gpu(mesh);

}