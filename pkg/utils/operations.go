package utils

import (
	"gonum.org/v1/gonum/mat"
)

func MultiplyMatrices(a, b *mat.Dense) *mat.Dense {
	result := mat.NewDense(a.RawMatrix().Rows, b.RawMatrix().Cols, nil)
	result.Mul(a, b)
	return result
}

func TransposeMatrix(m *mat.Dense) *mat.Dense {
	result := mat.NewDense(m.RawMatrix().Cols, m.RawMatrix().Rows, nil)
	result.CloneFrom(m.T())
	return result
}

func EyeMatrix(size int) *mat.Dense {
	I := mat.NewDense(size, size, nil)
	for i := 0; i < size; i++ {
		I.Set(i, i, 1.0)
	}
	return I
}
