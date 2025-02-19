package utils

import (
	"math"

	"github.ibm.com/modeling-analysis/kalman-filter/pkg/config"

	"gonum.org/v1/gonum/mat"
)

func NumericalJacobian(f func(*mat.VecDense) *mat.VecDense, x *mat.VecDense, outputSize int) *mat.Dense {
	n := x.Len()
	J := mat.NewDense(outputSize, n, nil) // Jacobian matrix

	// Define epsilon based on delta * |x|
	delta := config.Delta // Load from config (e.g., 0.01)

	for j := 0; j < n; j++ {
		epsilon := delta * math.Abs(x.AtVec(j))
		if epsilon == 0 {
			epsilon = delta // Use a small default epsilon if the component is zero
		}

		// Forward perturbation
		xPerturbedFwd := mat.NewVecDense(n, nil)
		xPerturbedFwd.CloneFromVec(x)
		xPerturbedFwd.SetVec(j, x.AtVec(j)+epsilon)

		// Backward perturbation
		xPerturbedBwd := mat.NewVecDense(n, nil)
		xPerturbedBwd.CloneFromVec(x)
		xPerturbedBwd.SetVec(j, x.AtVec(j)-epsilon)

		// Compute function values
		fxFwd := f(xPerturbedFwd)
		fxBwd := f(xPerturbedBwd)

		// Compute the partial derivative using symmetric difference
		for i := 0; i < outputSize; i++ {
			J.Set(i, j, (fxFwd.AtVec(i)-fxBwd.AtVec(i))/(2*epsilon))
		}
	}
	return J
}
