package tests

import (
	"testing"

	"github.ibm.com/modeling-analysis/kalman-filter/pkg/core"
	"github.ibm.com/modeling-analysis/kalman-filter/pkg/utils"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestNewExtendedKalmanFilter(t *testing.T) {
	stateDim, measDim := 4, 2
	initState := mat.NewVecDense(stateDim, []float64{1, 2, 3, 4})
	initCov := utils.EyeMatrix(stateDim)

	ekf, err := core.NewExtendedKalmanFilter(stateDim, measDim, initState, initCov)
	assert.NoError(t, err)
	assert.NotNil(t, ekf)
	assert.Equal(t, stateDim, ekf.Xdim)
	assert.Equal(t, measDim, ekf.Zdim)
	assert.Equal(t, initState.Len(), ekf.X.Len())
	assert.Equal(t, initCov.RawMatrix().Rows, ekf.P.RawMatrix().Rows)
}

func TestEKF_Predict(t *testing.T) {
	stateDim, measDim := 4, 2
	initState := mat.NewVecDense(stateDim, []float64{1, 1, 1, 1})
	initCov := utils.EyeMatrix(stateDim)
	processNoise := utils.EyeMatrix(stateDim)

	ekf, _ := core.NewExtendedKalmanFilter(stateDim, measDim, initState, initCov)
	err := ekf.Predict(processNoise)
	assert.NoError(t, err)
	assert.NotNil(t, ekf.X)
	assert.NotNil(t, ekf.P)
}

func TestEKF_Update(t *testing.T) {
	stateDim, measDim := 4, 2
	initState := mat.NewVecDense(stateDim, []float64{1, 1, 1, 1})
	initCov := utils.EyeMatrix(stateDim)
	measurement := mat.NewVecDense(measDim, []float64{0.5, -0.5})
	measNoise := utils.EyeMatrix(measDim)

	ekf, _ := core.NewExtendedKalmanFilter(stateDim, measDim, initState, initCov)
	err := ekf.Update(measurement, measNoise)
	assert.NoError(t, err)
	assert.NotNil(t, ekf.X)
	assert.NotNil(t, ekf.P)
}

func TestEKF_Getters(t *testing.T) {
	ekf, _ := core.NewExtendedKalmanFilter(4, 2, nil, nil)

	if ekf.State() != ekf.X {
		t.Errorf("State() did not return the correct reference")
	}
	if ekf.CovMatrix() != ekf.P {
		t.Errorf("CovMatrix() did not return the correct reference")
	}
	if ekf.Qmatrix() != ekf.Q {
		t.Errorf("Qmatrix() did not return the correct reference")
	}
	if ekf.Rmatrix() != ekf.R {
		t.Errorf("Rmatrix() did not return the correct reference")
	}
	if ekf.Innovation() != ekf.Y {
		t.Errorf("Innovation() did not return the correct reference")
	}
	if ekf.InnovationCov() != ekf.S {
		t.Errorf("InnovationCov() did not return the correct reference")
	}
	if ekf.KalmanGain() != ekf.K {
		t.Errorf("KalmanGain() did not return the correct reference")
	}
}

func TestEKF_SetQ(t *testing.T) {
	ekf, _ := core.NewExtendedKalmanFilter(3, 2, nil, nil)
	Q := utils.EyeMatrix(3)

	if err := ekf.SetQ(Q); err != nil {
		t.Errorf("SetQ failed with valid dimensions: %v", err)
	}

	Q_wrong := mat.NewDense(2, 2, nil)
	if err := ekf.SetQ(Q_wrong); err == nil {
		t.Errorf("SetQ should fail with incorrect dimensions but did not")
	}
}

func TestEKF_SetR(t *testing.T) {
	ekf, _ := core.NewExtendedKalmanFilter(3, 2, nil, nil)
	R := utils.EyeMatrix(2)

	if err := ekf.SetR(R); err != nil {
		t.Errorf("SetR failed with valid dimensions: %v", err)
	}

	R_wrong := mat.NewDense(3, 3, nil)
	if err := ekf.SetR(R_wrong); err == nil {
		t.Errorf("SetR should fail with incorrect dimensions but did not")
	}
}

func TestEKF_SetfF(t *testing.T) {
	ekf, _ := core.NewExtendedKalmanFilter(3, 2, nil, nil)

	validF := func(x *mat.VecDense) *mat.VecDense {
		return x
	}
	if err := ekf.SetfF(validF); err != nil {
		t.Errorf("SetfF failed with a valid function: %v", err)
	}

	invalidF := func(x *mat.VecDense) *mat.VecDense {
		return mat.NewVecDense(2, nil)
	}
	if err := ekf.SetfF(invalidF); err == nil {
		t.Errorf("SetfF should fail with an invalid function but did not")
	}
}

func TestEKF_SethH(t *testing.T) {
	ekf, _ := core.NewExtendedKalmanFilter(3, 2, nil, nil)

	validH := func(x *mat.VecDense) *mat.VecDense {
		return mat.NewVecDense(2, nil)
	}
	if err := ekf.SethH(validH); err != nil {
		t.Errorf("SethH failed with a valid function: %v", err)
	}

	invalidH := func(x *mat.VecDense) *mat.VecDense {
		return mat.NewVecDense(3, nil)
	}
	if err := ekf.SethH(invalidH); err == nil {
		t.Errorf("SethH should fail with an invalid function but did not")
	}
}

func TestEKF_SetStateLimiter(t *testing.T) {
	ekf, _ := core.NewExtendedKalmanFilter(3, 2, nil, nil)
	minVals := []float64{-1, -1, -1}
	maxVals := []float64{1, 1, 1}

	if err := ekf.SetStateLimiter(minVals, maxVals); err != nil {
		t.Errorf("SetStateLimiter failed with valid limits: %v", err)
	}

	invalidMin := []float64{-1, -1}
	if err := ekf.SetStateLimiter(invalidMin, maxVals); err == nil {
		t.Errorf("SetStateLimiter should fail with mismatched dimensions but did not")
	}
}
