exercise:

1. implement the following function declared in vector.hpp: 

// y := a * x + b * y
void axpby(double a, const vector & x, double b, vector & v);

hint: match the pattern of the other operator overloads by
defining an axpby kernel that operates on the pointers held by
the vector class. 

2. identify expressions in cg.hpp that can be replaced by axpby's