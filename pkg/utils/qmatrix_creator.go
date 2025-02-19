package utils

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type QMatrixCreator struct{}

func NewQMatrixCreator() *QMatrixCreator {
	return &QMatrixCreator{}
}

func (q *QMatrixCreator) GetMatrix(stateChange []float64) *mat.Dense {
	n := len(stateChange)
	Q := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		Q.Set(i, i, math.Pow(stateChange[i], 2))
	}
	return Q
}
