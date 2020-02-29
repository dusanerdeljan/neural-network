#pragma once
#include "ActivationFunctions.h"
#include <vector>
#include <iostream>

struct MatrixError : std::runtime_error
{
	MatrixError(const char* error) : std::runtime_error(error) {}
};

class Matrix
{
private:
	unsigned int m_Rows;
	unsigned int m_Columns;
	double* m_Matrix;
public:
	Matrix();
	Matrix(unsigned int rows, unsigned int columns, double initValue = -1);
	Matrix(const Matrix& matrix);
	Matrix(Matrix&& matrix);
	Matrix(const std::vector<double>& data);
#ifdef _DEBUG
	Matrix(const std::vector<std::vector<double>>& matrix);
#endif // _DEBUG
	Matrix& operator=(const Matrix& matrix);
	Matrix& operator=(Matrix&& matrix);
	~Matrix();

	inline unsigned int GetWidth() const { return m_Columns; }
	inline unsigned int GetHeight() const { return m_Rows; }

	void Randomize(double min = -1, double max = 1);
	void ZeroOut();
	std::vector<double> GetColumnVector() const;

	Matrix& MapFunction(Activation::ActivationFunction* func);
	Matrix& MapDerivative(Activation::ActivationFunction* func);

	double& operator()(unsigned int row, unsigned int column);
	const double& operator()(unsigned int row, unsigned int column) const;
	double& operator[](const std::pair<unsigned int, unsigned int> index);
	const double& operator[](const std::pair<unsigned int, unsigned int> index) const;

	Matrix& operator +=(const Matrix& other);
	Matrix& operator +=(double scalar);
	Matrix& operator -=(const Matrix& other);
	Matrix& operator -=(double scalar);
	Matrix& operator *=(double scalar);
	Matrix& operator *= (const Matrix& other);
	Matrix& operator /= (double scalar);
	Matrix& DotProduct(const Matrix& other);
	Matrix& Transpose();
	template<typename _Func> Matrix& Map(_Func func);

	friend std::ostream& operator << (std::ostream& out, const Matrix& m);

	friend Matrix operator+(const Matrix& left, const Matrix& right);
	friend Matrix operator*(const Matrix& matrix, double scalar);
	friend Matrix operator*(double scalar, const Matrix& matrix);
	friend Matrix operator-(const Matrix& left, const Matrix& right);
	friend Matrix operator-(double scalar, const Matrix& matrix);
	friend Matrix operator*(const Matrix& left, const Matrix& right);
	friend Matrix operator/(const Matrix& matrix, double scalar);

	static Matrix OuterProduct(const Matrix& a, const Matrix& b);
	static Matrix OneHot(unsigned int one, unsigned int size);
	static Matrix DotProduct(const Matrix& left, const Matrix& right);
	static Matrix Transpose(const Matrix& matrix);
	static Matrix BuildColumnMatrix(unsigned int rows, double value);
	template<typename _Func> static Matrix Map(const Matrix& matrix, _Func func);
private:
	bool HasSameDimension(const Matrix& other) const;
};

template<typename _Func>
inline Matrix & Matrix::Map(_Func func)
{
	for (unsigned int i = 0; i < m_Rows*m_Columns; ++i)
		m_Matrix[i] = func(m_Matrix[i]);
	return *this;
}

template<typename _Func>
inline Matrix Matrix::Map(const Matrix & matrix, _Func func)
{
	Matrix result = matrix;
	result.Map(func);
	return result;
}
