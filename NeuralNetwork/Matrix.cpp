#include "Matrix.h"
#include <random>
#include <cstdlib>

#ifdef _DEBUG
	#define LOG(x) std::cout << x << std::endl
#endif // _DEBUG


Matrix::Matrix(unsigned int rows, unsigned int columns, double initValue) : m_Rows(rows), m_Columns(columns), m_Matrix(new double[rows*columns])
{
	/*for (unsigned int i = 0; i < m_Rows*m_Columns; i++)
	{
		m_Matrix[i] = initValue;
	}*/
	Randomize();
}

Matrix::Matrix(const Matrix & matrix) : m_Rows(matrix.m_Rows), m_Columns(matrix.m_Columns), m_Matrix(new double[matrix.m_Rows*matrix.m_Columns])
{
	memcpy(m_Matrix, matrix.m_Matrix, sizeof(double)*m_Rows*m_Columns);
}

Matrix::Matrix(Matrix && matrix) : m_Rows(matrix.m_Rows), m_Columns(matrix.m_Columns), m_Matrix(matrix.m_Matrix)
{
	matrix.m_Matrix = nullptr;

}

Matrix::Matrix(const std::vector<double>& data) : m_Rows(data.size()), m_Columns(1), m_Matrix(new double[data.size()])
{
	for (unsigned int i = 0; i < m_Rows; ++i)
	{
		m_Matrix[i] = data[i];
	}
}

Matrix & Matrix::operator=(const Matrix & matrix)
{
	m_Rows = matrix.m_Rows; m_Columns = matrix.m_Columns;
	double* newMatrix = new double[m_Rows*m_Columns];
	memcpy(newMatrix, matrix.m_Matrix, sizeof(double)*m_Rows*m_Columns);
	delete m_Matrix;
	m_Matrix = newMatrix;
	return *this;
}

Matrix & Matrix::operator=(Matrix && matrix)
{
	m_Rows = matrix.m_Rows; m_Columns = matrix.m_Columns; m_Matrix = matrix.m_Matrix;
	matrix.m_Matrix = nullptr;
	return *this;
}

Matrix::~Matrix()
{
	delete m_Matrix;
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

std::vector<double> Matrix::GetColumnVector() const
{
	if (m_Columns != 1)
		throw MatrixError("Number of columns has to be 1 in order to make column vector!");
	std::vector<double> columnVector(m_Rows);
	for (unsigned int i = 0; i < m_Rows; ++i)
		columnVector[i] = m_Matrix[i];
	return columnVector;
}

Matrix & Matrix::MapFunction(ActivationFunctions::ActivationFunction * func)
{
	for (unsigned int i = 0; i < m_Rows*m_Columns; ++i)
	{
		m_Matrix[i] = func->Function(m_Matrix[i]);
	}
	return *this;
}

Matrix & Matrix::MapDerivative(ActivationFunctions::ActivationFunction * func)
{
	for (unsigned int i = 0; i < m_Rows*m_Columns; ++i)
	{
		m_Matrix[i] = func->Function(m_Matrix[i]);
	}
	return *this;
}

Matrix & Matrix::operator+=(const Matrix & other)
{
	if (!HasSameDimension(other))
		throw MatrixError("Matrices do not have the same dimension!");
	for (unsigned int i = 0; i < m_Rows*m_Columns; ++i)
	{
		m_Matrix[i] += other.m_Matrix[i];
	}
	return *this;
}

Matrix & Matrix::operator-=(const Matrix & other)
{
	if (!HasSameDimension(other))
		throw MatrixError("Matrices do not have the same dimension!");
	for (unsigned int i = 0; i < m_Rows*m_Columns; ++i)
	{
		m_Matrix[i] -= other.m_Matrix[i];
	}
	return *this;
}

Matrix & Matrix::operator*=(double scalar)
{
	for (unsigned int i = 0; i < m_Rows*m_Columns; ++i)
	{
		m_Matrix[i] *= scalar;
	}
	return *this;
}

Matrix & Matrix::operator*=(const Matrix & other)
{
	if (m_Columns != other.m_Rows)
		throw MatrixError("Number of columns of the left matrix has to match number of rows of the right matrix!");
	*this = *this * other;
	return *this;
}

Matrix & Matrix::DotProduct(const Matrix & other)
{
	if (!HasSameDimension(other))
		throw MatrixError("Matrices do not have the same dimension!");
	for (unsigned int i = 0; i < m_Rows*m_Columns; ++i)
	{
		m_Matrix[i] *= other.m_Matrix[i];
	}
	return *this;
}

Matrix & Matrix::Transpose()
{
	double* transposedMatrix = new double[m_Rows*m_Columns];
	for (unsigned int i = 0; i < m_Rows; ++i)
	{
		for (unsigned int j = 0; j < m_Columns; ++j)
		{
			transposedMatrix[i + j*m_Rows] = m_Matrix[j + i*m_Columns];
		}
	}
	delete m_Matrix;
	m_Matrix = transposedMatrix;
	std::swap(m_Rows, m_Columns);
	return *this;
}

Matrix Matrix::DotProduct(const Matrix & left, const Matrix & right)
{
	if (!left.HasSameDimension(right))
		throw MatrixError("Matrices do not have the same dimension!");
	Matrix result(left.m_Rows, left.m_Columns);
	for (unsigned int i = 0; i < result.m_Rows*result.m_Columns; i++)
	{
		result.m_Matrix[i] = left.m_Matrix[i] * right.m_Matrix[i];
	}
	return result;
}

Matrix Matrix::Transpose(const Matrix & matrix)
{
	Matrix result(matrix.m_Rows, matrix.m_Columns);
	for (unsigned int i = 0; i < matrix.m_Rows; ++i)
	{
		for (unsigned int j = 0; j < matrix.m_Columns; ++j)
		{
			result.m_Matrix[i + j*matrix.m_Rows] = matrix.m_Matrix[j + i*matrix.m_Columns];
		}
	}
	std::swap(result.m_Rows, result.m_Columns);
	return result;
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
	if (!left.HasSameDimension(right))
		throw MatrixError("Matrices do not have the same dimension!");
	Matrix result(left.m_Rows, left.m_Columns);
	for (unsigned int i = 0; i < result.m_Rows*result.m_Columns; i++)
	{
		result.m_Matrix[i] = left.m_Matrix[i] + right.m_Matrix[i];
	}
	return result;
}

Matrix operator*(const Matrix & matrix, double scalar)
{
	Matrix result(matrix.m_Rows, matrix.m_Columns);
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
	if (!left.HasSameDimension(right))
		throw MatrixError("Matrices do not have the same dimension!");
	Matrix result(left.m_Rows, left.m_Columns);
	for (unsigned int i = 0; i < result.m_Rows*result.m_Columns; i++)
	{
		result.m_Matrix[i] = left.m_Matrix[i] - right.m_Matrix[i];
	}
	return result;
}

Matrix operator*(const Matrix & left, const Matrix & right)
{
	if (left.m_Columns != right.m_Rows)
		throw MatrixError("Number of columns of the left matrix has to match number of rows of the right matrix!");
	Matrix result(left.m_Rows, right.m_Columns);
	for (unsigned int i = 0; i < result.m_Rows; ++i)
	{
		for (unsigned int j = 0; j < result.m_Columns; ++j)
		{
			for (unsigned int k = 0; k < left.m_Columns; ++k)
			{
				result.m_Matrix[j + i*result.m_Columns] += left.m_Matrix[k + i*left.m_Columns] * right.m_Matrix[j + k*right.m_Columns];
			}
		}
	}
	return result;
}
