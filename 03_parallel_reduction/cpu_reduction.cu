#include <iostream>

#include "timer.hpp"

double f(double x) {
    return 4.0 / (1.0 + x * x);
}

//  1
// ⌠   4        
// |  ---  dx = π
// ⌡  1+x²      
//  0 
double calculate_pi(int n) {
    double dx = 1.0 / n;

    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double x = (i + 0.5) * dx;
        sum += f(x) * dx; // numerical integration
    }
    
    return sum;
}

int main() {

    static constexpr int n = 100'000'000;
    static constexpr double pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862090;

    double pi_approx;

    float time_ms = 1000.0f * time([&](){
        pi_approx = calculate_pi(n);
    });

    std::cout << "computed pi approximation in " << time_ms << " ms, with error " << fabs(pi - pi_approx) << std::endl;

}