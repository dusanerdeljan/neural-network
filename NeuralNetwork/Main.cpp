#include <iostream>
#include "Matrix.h"

int main()
{
	Matrix a({ 1, 2, 3, 4, 5 });
	std::cout << a << std::endl;
	std::vector<double> v = a.GetColumnVector();
	for (int x : v)
		std::cout << x << " ";
	std::cin.get();
	return 0;
}