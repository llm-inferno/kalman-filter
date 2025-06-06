package tests

import (
	"math"
	"testing"

	"kalman-filter/pkg/utils"

	"gonum.org/v1/gonum/mat"
)

/*
Test for NumericalJacobian function with an analytical function. Here the Jacobian matrix of the function f:R^3 -> R^4 with components:
y1= x1
y2 = 5x3
y3 = 4x2^2 - 2x3
y4 = x1^2 * x3 is
J = [1 0 0; 0 0 5; 0 8*x2 -2;2*x1*x3 0 x1^2]
This example shows that the Jacobian matrix need not be a square matrix.
*/

var tolerance = 1e-5
var inputLen = 3
var outputLen = 4 // output length of f(x).

func TestNumericalJacobian(t *testing.T) {
	foo := func(x *mat.VecDense) *mat.VecDense {
		x1, x2, x3 := x.AtVec(0), x.AtVec(1), x.AtVec(2)
		fx := mat.NewVecDense(outputLen, nil)
		fx.SetVec(0, x1)
		fx.SetVec(1, 5*x3)
		fx2 := 4*math.Pow(x2, 2) - 2*x3
		fx3 := math.Pow(x1, 2) * x3
		fx.SetVec(2, fx2)
		fx.SetVec(3, fx3)

		return fx
	}

	x := mat.NewVecDense(inputLen, []float64{1, 2, 3})
	expectedOutput := mat.NewDense(outputLen, inputLen, []float64{
		1, 0, 0,
		0, 0, 5,
		0, 8 * x.AtVec(1), -2,
		2 * x.AtVec(0) * x.AtVec(2), 0, math.Pow(x.AtVec(0), 2),
	})
	gotOutput := utils.NumericalJacobian(foo, x, outputLen)

	rows := gotOutput.RawMatrix().Rows
	cols := gotOutput.RawMatrix().Cols

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if math.Abs(expectedOutput.At(i, j)-gotOutput.At(i, j)) > tolerance {
				t.Errorf("Mismatch at (%d, %d): got %f, expected %f", i, j, gotOutput.At(i, j), expectedOutput.At(i, j))
			}
		}
	}
}
