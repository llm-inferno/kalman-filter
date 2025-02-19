package utils

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type RMatrixCreator struct {
	errorLevel  float64
	tPercentile float64
	gamma       float64
}

func NewRMatrixCreator(errorLevel, tPercentile, gamma float64) *RMatrixCreator {
	return &RMatrixCreator{
		errorLevel:  errorLevel,
		tPercentile: tPercentile,
		gamma:       gamma,
	}
}

func (r *RMatrixCreator) GetMatrix(meanMeasure []float64) *mat.Dense {
	n := len(meanMeasure)
	R := mat.NewDense(n, n, nil)
	factor := math.Pow(r.errorLevel/r.tPercentile, 2) / r.gamma
	for i := 0; i < n; i++ {
		R.Set(i, i, factor*math.Pow(meanMeasure[i], 2))
	}
	return R
}
