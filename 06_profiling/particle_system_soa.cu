#include <vector>

#include "vec.hpp"
#if 0
////////////////////////////////////////////////////////////////////////////////

struct ParticleSystem {
    ParticleSystem(int n) : mass(n), 
                            charge(n),
                            force(n),
                            velocity(n),
                            position(n) {}

    std::vector< float > mass;
    std::vector< float > charge;
    std::vector< vec3f > force;
    std::vector< vec3f > velocity;
    std::vector< vec3f > position;
};

void zero_forces(std::vector< Particle > & particles) {
    for (auto particle : particles) {
        particle.force = {0.0f, 0.0f, 0.0f};
    }
}

void coulomb_force(std::vector< Particle > & particles, float Q, vec3f X) {
    for (auto particle : particles) {
        vec3f r = particle.position - X;
        particle.force += r * ((particle.charge * Q) / dot(r, r));
    }
}

void lorentz_force(std::vector< Particle > & particles, vec3f B) {
    for (auto particle : particles) {
        particle.force += particle.charge * cross(particle.velocity, B);
    }
}

void gravitational_force(std::vector< Particle > & particles, float g) {
    for (auto particle : particles) {
        particle.force += vec3f{0.0f, 0.0f, -g};
    }
}

void time_integrate(std::vector< Particle > & particles, float dt) {
    for (auto particle : particles) {
        vec3f acceleration = particle.force / particle.mass;
        particle.position += particle.velocity * dt;
        particle.velocity += acceleration * dt;
    }
}

void simulate(std::vector< Particle > & particles) {

    float dt = 0.01f;
    float Q = 4.2f;
    vec3f X = {0.0f, 0.0f, 0.0f};
    vec3f B = {1.0f, 2.0f, 3.0f};
    float g = -9.81f;

    int num_steps = 100;
    for (int i = 0; i < num_steps; i++) {
        // calculate net force on each particle
        zero_forces(particles);
        coulomb_force(particles, Q, X);
        lorentz_force(particles, B);
        gravitational_force(particles, g);

        // advance through time
        time_integrate(particles, dt);
    }

}

////////////////////////////////////////////////////////////////////////////////

#endif
int main() {





}