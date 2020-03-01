#include "Matrix.h"
#include <random>
#include <algorithm>
#include <numeric>
#include <cstdlib>

#ifdef _DEBUG
#define LOG(x) std::cout << x << std::endl
#endif // _DEBUG


Matrix::Matrix() : m_Rows(0), m_Columns(0), m_Matrix()
{
}

Matrix::Matrix(unsigned int rows, unsigned int columns, double initValue) : m_Rows(rows), m_Columns(columns), m_Matrix(rows*columns)
{
	if (initValue == -1)
		Randomize();
	else
		std::fill(m_Matrix.begin(), m_Matrix.end(), initValue);
}

Matrix::Matrix(const Matrix & matrix) : m_Rows(matrix.m_Rows), m_Columns(matrix.m_Columns), m_Matrix(matrix.m_Matrix)
{
}

Matrix::Matrix(Matrix && matrix) : m_Rows(matrix.m_Rows), m_Columns(matrix.m_Columns), m_Matrix(std::move(matrix.m_Matrix))
{

}

Matrix::Matrix(const std::vector<double>& data) : m_Rows(data.size()), m_Columns(1), m_Matrix(data)
{
}

#ifdef _DEBUG
Matrix::Matrix(const std::vector<std::vector<double>>& matrix) : m_Rows(matrix.size()), m_Columns(matrix[0].size()), m_Matrix(matrix.size()*matrix[0].size())
{
	for (unsigned int i = 0; i < m_Rows; ++i)
	{
		for (unsigned int j = 0; j < m_Columns; ++j)
		{
			(*this)(i, j) = matrix[i][j];
		}
	}
}
#endif // _DEBUG


Matrix & Matrix::operator=(const Matrix & matrix)
{
	m_Rows = matrix.m_Rows; m_Columns = matrix.m_Columns;
	m_Matrix = matrix.m_Matrix;
	return *this;
}

Matrix & Matrix::operator=(Matrix && matrix)
{
	m_Rows = matrix.m_Rows; m_Columns = matrix.m_Columns; m_Matrix = std::move(matrix.m_Matrix);
	return *this;
}

Matrix::~Matrix()
{
}

double Matrix::Sum() const
{
	double sum = 0.0;
	return std::accumulate(m_Matrix.begin(), m_Matrix.end(), sum);
}

void Matrix::Randomize(double min, double max)
{

	std::random_device randomDevice;
	std::mt19937 engine(randomDevice());
	std::uniform_real_distribution<double> valueDistribution(min, max);
	for (unsigned int i = 0; i < m_Rows*m_Columns; ++i)
	{
		m_Matrix[i] = valueDistribution(engine);
	}
}

void Matrix::ZeroOut()
{
	std::fill(m_Matrix.begin(), m_Matrix.end(), 0);
}

std::vector<double> Matrix::GetColumnVector() const
{
#ifdef _DEBUG
	if (m_Columns != 1)
		throw MatrixError("Number of columns has to be 1 in order to make column vector!");
#endif // _DEBUG
	return m_Matrix;
}

void Matrix::SaveMatrix(std::ofstream & outfile) const
{
	outfile.write((char*)(&m_Rows), sizeof(m_Rows));
	outfile.write((char*)(&m_Columns), sizeof(m_Columns));
	outfile.write((char*)&m_Matrix[0], sizeof(double)*m_Rows*m_Columns);
}

Matrix & Matrix::MapFunction(Activation::ActivationFunction * func)
{
	for (unsigned int i = 0; i < m_Rows*m_Columns; ++i)
	{
		m_Matrix[i] = func->Function(m_Matrix[i]);
	}
	return *this;
}

Matrix & Matrix::MapDerivative(Activation::ActivationFunction * func)
{
	for (unsigned int i = 0; i < m_Rows*m_Columns; ++i)
	{
		m_Matrix[i] = func->Derivative(m_Matrix[i]);
	}
	return *this;
}

double & Matrix::operator()(unsigned int row, unsigned int column)
{
#ifdef _DEBUG
	if ((column + row*m_Columns) >= m_Rows*m_Columns)
		throw MatrixError("Index out of range!");
#endif // _DEBUG
	return m_Matrix[column + row*m_Columns];
}

const double & Matrix::operator()(unsigned int row, unsigned int column) const
{
#ifdef _DEBUG
	if ((column + row*m_Columns) >= m_Rows*m_Columns)
		throw MatrixError("Index out of range!");
#endif // _DEBUG
	return m_Matrix[column + row*m_Columns];
}

double & Matrix::operator[](const std::pair<unsigned int, unsigned int> index)
{
#ifdef _DEBUG
	if ((index.second + index.first*m_Columns) >= m_Rows*m_Columns)
		throw MatrixError("Index out of range!");
#endif // _DEBUG
	return m_Matrix[index.second + index.first*m_Columns];
}

const double & Matrix::operator[](const std::pair<unsigned int, unsigned int> index) const
{
#ifdef _DEBUG
	if ((index.second + index.first*m_Columns) >= m_Rows*m_Columns)
		throw MatrixError("Index out of range!");
#endif // _DEBUG
	return m_Matrix[index.second + index.first*m_Columns];
}

Matrix & Matrix::operator+=(const Matrix & other)
{
#ifdef _DEBUG
	if (!HasSameDimension(other))
		throw MatrixError("Matrices do not have the same dimension!");
#endif // _DEBUG
	std::transform(m_Matrix.begin(), m_Matrix.end(), other.m_Matrix.begin(), m_Matrix.begin(), std::plus<double>());
	return *this;
}

Matrix & Matrix::operator+=(double scalar)
{
	std::for_each(m_Matrix.begin(), m_Matrix.end(), [scalar](double& x) { x += scalar; });
	return *this;
}

Matrix & Matrix::operator-=(const Matrix & other)
{
#ifdef _DEBUG
	if (!HasSameDimension(other))
		throw MatrixError("Matrices do not have the same dimension!");
#endif // _DEBUG
	std::transform(m_Matrix.begin(), m_Matrix.end(), other.m_Matrix.begin(), m_Matrix.begin(), std::minus<double>());
	return *this;
}

Matrix & Matrix::operator-=(double scalar)
{
	std::for_each(m_Matrix.begin(), m_Matrix.end(), [scalar](double& x) { x -= scalar; });
	return *this;
}

Matrix & Matrix::operator*=(double scalar)
{
	std::for_each(m_Matrix.begin(), m_Matrix.end(), [scalar](double& x) { x *= scalar; });
	return *this;
}

Matrix & Matrix::operator*=(const Matrix & other)
{
#ifdef _DEBUG
	if (m_Columns != other.m_Rows)
		throw MatrixError("Number of columns of the left matrix has to match number of rows of the right matrix!");
#endif // _DEBUG
	*this = *this * other;
	return *this;
}

Matrix & Matrix::operator/=(double scalar)
{
#ifdef _DEBUG
	if (scalar == 0)
		throw MatrixError("Cannot divide by zero!");
#endif // _DEBUG
	std::for_each(m_Matrix.begin(), m_Matrix.end(), [scalar](double& x) { x /= scalar; });
	return *this;
}

Matrix & Matrix::DotProduct(const Matrix & other)
{
#ifdef _DEBUG
	if (!HasSameDimension(other))
		throw MatrixError("Matrices do not have the same dimension!");
#endif // _DEBUG
	for (unsigned int i = 0; i < m_Rows*m_Columns; ++i)
	{
		m_Matrix[i] *= other.m_Matrix[i];
	}
	return *this;
}

Matrix & Matrix::Transpose()
{
	if (m_Rows != 1 && m_Columns != 1)
	{
		std::vector<double> transposedMatrix(m_Rows*m_Columns);
		for (unsigned int i = 0; i < m_Rows; ++i)
		{
			for (unsigned int j = 0; j < m_Columns; ++j)
			{
				transposedMatrix[i + j*m_Rows] = m_Matrix[j + i*m_Columns];
			}
		}
		m_Matrix = std::move(transposedMatrix);
	}
	std::swap(m_Rows, m_Columns);
	return *this;
}

Matrix Matrix::LoadMatrix(std::ifstream & infile)
{
	unsigned int rows, columns;
	infile.read((char*)&rows, sizeof(rows));
	infile.read((char*)&columns, sizeof(columns));
	Matrix matrix(rows, columns);
	double* m = new double[rows*columns];
	infile.read((char*)m, sizeof(double)*rows*columns);
	std::vector<double> mat(m, m + rows*columns);
	matrix.m_Matrix = std::move(mat);
	return matrix;
}

Matrix Matrix::OuterProduct(const Matrix & a, const Matrix & b)
{
	Matrix result(b.m_Columns, a.m_Columns, 0);
	for (unsigned int i = 0; i < a.m_Columns; ++i)
	{
		for (unsigned int j = 0; j < b.m_Columns; ++j)
		{
			result(i, j) = a(0, i) * b(0, j);
		}
	}
	return result;
}

Matrix Matrix::OneHot(unsigned int one, unsigned int size)
{
	Matrix matrix(1, size, 0);
	matrix(0, one) = 1;
	return matrix;
}

Matrix Matrix::DotProduct(const Matrix & left, const Matrix & right)
{
#ifdef _DEBUG
	if (!left.HasSameDimension(right))
		throw MatrixError("Matrices do not have the same dimension!");
#endif // _DEBUG
	Matrix result(left.m_Rows, left.m_Columns);
	result.m_Matrix = left.m_Matrix;
	std::transform(result.m_Matrix.begin(), result.m_Matrix.end(), right.m_Matrix.begin(), result.m_Matrix.begin(), std::multiplies<double>());
	return result;
}

Matrix Matrix::Transpose(const Matrix & matrix)
{
	Matrix result(matrix);
	if (matrix.m_Rows != 1 && matrix.m_Columns != 1)
	{
		for (unsigned int i = 0; i < matrix.m_Rows; ++i)
		{
			for (unsigned int j = 0; j < matrix.m_Columns; ++j)
			{
				result.m_Matrix[i + j*matrix.m_Rows] = matrix.m_Matrix[j + i*matrix.m_Columns];
			}
		}
	}
	std::swap(result.m_Rows, result.m_Columns);
	return result;
}

Matrix Matrix::BuildColumnMatrix(unsigned int rows, double value)
{
	Matrix matrix(rows, 1);
	std::fill(matrix.m_Matrix.begin(), matrix.m_Matrix.end(), value);
	return matrix;
}

bool Matrix::HasSameDimension(const Matrix & other) const
{
	return m_Rows == other.m_Rows && m_Columns == other.m_Columns;
}

std::ostream & operator<<(std::ostream & out, const Matrix & m)
{
	for (unsigned int i = 0; i < m.m_Rows; ++i)
	{
		for (unsigned int j = 0; j < m.m_Columns; ++j)
		{
			out << m.m_Matrix[j + i*m.m_Columns] << " ";
		}
		out << std::endl;
	}
	return out;
}

Matrix operator+(const Matrix & left, const Matrix & right)
{
#ifdef _DEBUG
	if (!left.HasSameDimension(right))
		throw MatrixError("Matrices do not have the same dimension!");
#endif // _DEBUG
	Matrix result(left.m_Rows, left.m_Columns, 0);
	for (unsigned int i = 0; i < result.m_Rows*result.m_Columns; i++)
	{
		result.m_Matrix[i] = left.m_Matrix[i] + right.m_Matrix[i];
	}
	return result;
}

Matrix operator*(const Matrix & matrix, double scalar)
{
	Matrix result(matrix.m_Rows, matrix.m_Columns, 0);
	for (unsigned int i = 0; i < result.m_Rows*result.m_Columns; i++)
	{
		result.m_Matrix[i] = matrix.m_Matrix[i] * scalar;
	}
	return result;
}

Matrix operator*(double scalar, const Matrix & matrix)
{
	return matrix*scalar;
}

Matrix operator-(const Matrix & left, const Matrix & right)
{
#ifdef _DEBUG
	if (!left.HasSameDimension(right))
		throw MatrixError("Matrices do not have the same dimension!");
#endif // _DEBUG

	Matrix result(left.m_Rows, left.m_Columns, 0);
	for (unsigned int i = 0; i < result.m_Rows*result.m_Columns; i++)
	{
		result.m_Matrix[i] = left.m_Matrix[i] - right.m_Matrix[i];
	}
	return result;
}

Matrix operator-(double scalar, const Matrix & matrix)
{
	Matrix result(matrix.m_Rows, matrix.m_Columns);
	result.Map([scalar](double x) { return scalar - x; });
	return result;
}

Matrix operator*(const Matrix & left, const Matrix & right)
{
#ifdef _DEBUG
	if (left.m_Columns != right.m_Rows)
		throw MatrixError("Number of columns of the left matrix has to match number of rows of the right matrix!");
#endif // _DEBUG

	Matrix result(left.m_Rows, right.m_Columns, 0);
	for (unsigned int i = 0; i < left.m_Rows; ++i)
	{
		for (unsigned int j = 0; j < right.m_Columns; ++j)
		{
			for (unsigned int k = 0; k < left.m_Columns; ++k)
			{
				result(i, j) += left(i, k) * right(k, j);
			}
		}
	}
	return result;
}

Matrix operator/(const Matrix & matrix, double scalar)
{
#ifdef _DEBUG
	if (scalar == 0)
		throw MatrixError("Cannot divide by zero!");
#endif // _DEBUG

	Matrix result(matrix.m_Rows, matrix.m_Columns, 0);
	for (unsigned int i = 0; i < result.m_Rows*result.m_Columns; i++)
	{
		result.m_Matrix[i] = matrix.m_Matrix[i] / scalar;
	}
	return result;
}
