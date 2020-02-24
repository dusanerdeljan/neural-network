#pragma once
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
	Matrix(unsigned int rows, unsigned int columns, double initValue = 0);
	Matrix(const Matrix& matrix);
	Matrix(Matrix&& matrix);
	Matrix(const std::vector<double>& data);
	Matrix& operator=(const Matrix& matrix);
	Matrix& operator=(Matrix&& matrix);
	~Matrix();

	inline unsigned int GetWidth() const { return m_Columns; }
	inline unsigned int GetHeight() const { return m_Rows; }

	void Randomize(double min = -1, double max = 1);
	std::vector<double> GetColumnVector() const;
	template<typename _Func> Matrix& Map(_Func func);

	Matrix& operator +=(const Matrix& other);
	Matrix& operator -=(const Matrix& other);
	Matrix& operator *=(double scalar);
	Matrix& operator *= (const Matrix& other);
	Matrix& DotProduct(const Matrix& other);
	Matrix& Transpose();

	friend std::ostream& operator << (std::ostream& out, const Matrix& m);

	friend Matrix operator+(const Matrix& left, const Matrix& right);
	friend Matrix operator*(const Matrix& matrix, double scalar);
	friend Matrix operator*(double scalar, const Matrix& matrix);
	friend Matrix operator-(const Matrix& left, const Matrix& right);
	friend Matrix operator*(const Matrix& left, const Matrix& right);

	template<typename _Func> static Matrix Map(const Matrix& matrix, _Func func);
	static Matrix DotProduct(const Matrix& left, const Matrix& right);
	static Matrix Transpose(const Matrix& matrix);
private:
	bool HasSameDimension(const Matrix& other) const;
};

template<typename _Func>
inline Matrix & Matrix::Map(_Func func)
{
	for (unsigned int i = 0; i < m_Rows*m_Columns; i++)
	{
		m_Matrix[i] = func(m_Matrix[i]);
	}
	return *this;
}

template<typename _Func>
inline Matrix Matrix::Map(const Matrix & matrix, _Func func)
{
	Matrix result(matrix.m_Rows, matrix.m_Columns);
	for (unsigned int i = 0; i < matrix.m_Rows*matrix.m_Columns; i++)
	{
		result.m_Matrix[i] = func(matrix.m_Matrix[i]);
	}
	return result;
}

