package tests

import (
	"testing"

	"github.ibm.com/modeling-analysis/kalman-filter/pkg/utils"

	"github.ibm.com/modeling-analysis/kalman-filter/pkg/core"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestNewKalmanFilterND(t *testing.T) {
	stateDim, measDim := 4, 2
	initState := mat.NewVecDense(stateDim, []float64{1, 2, 3, 4})
	initCov := utils.EyeMatrix(stateDim)

	kf, err := core.NewKalmanFilterND(stateDim, measDim, initState, initCov)
	assert.NoError(t, err)
	assert.NotNil(t, kf)
	assert.Equal(t, stateDim, kf.Xdim)
	assert.Equal(t, measDim, kf.Zdim)
	assert.Equal(t, stateDim, kf.X.Len())
	assert.Equal(t, stateDim, kf.P.RawMatrix().Rows)
}

func TestKF_Predict(t *testing.T) {
	stateDim, measDim := 4, 2
	initState := mat.NewVecDense(stateDim, []float64{1, 2, 3, 4})
	initCov := utils.EyeMatrix(stateDim)
	kf, _ := core.NewKalmanFilterND(stateDim, measDim, initState, initCov)

	Q := utils.EyeMatrix(stateDim)
	err := kf.Predict(Q)
	assert.NoError(t, err)
	assert.NotNil(t, kf.P)
}

func TestKF_Update(t *testing.T) {
	stateDim, measDim := 4, 2
	initState := mat.NewVecDense(stateDim, []float64{1, 2, 3, 4})
	initCov := utils.EyeMatrix(stateDim)
	kf, _ := core.NewKalmanFilterND(stateDim, measDim, initState, initCov)

	z := mat.NewVecDense(measDim, []float64{5, 6})
	R := utils.EyeMatrix(measDim)
	err := kf.Update(z, R)
	assert.NoError(t, err)
	assert.NotNil(t, kf.X)
}

func TestKF_InvalidStateDimension(t *testing.T) {
	_, err := core.NewKalmanFilterND(0, 2, nil, nil)
	assert.Error(t, err)
}

func TestKF_InvalidMeasurementDimension(t *testing.T) {
	_, err := core.NewKalmanFilterND(4, 0, nil, nil)
	assert.Error(t, err)
}

func TestKF_InvalidInitStateDimension(t *testing.T) {
	initState := mat.NewVecDense(3, nil) // Incorrect state dimension
	_, err := core.NewKalmanFilterND(4, 2, initState, nil)
	assert.Error(t, err)
}

func TestKF_InvalidInitCovDimension(t *testing.T) {
	initCov := mat.NewDense(3, 3, nil) // Incorrect covariance dimension
	_, err := core.NewKalmanFilterND(4, 2, nil, initCov)
	assert.Error(t, err)
}

func TestKF_StateRetrieval(t *testing.T) {
	stateDim, measDim := 4, 2
	initState := mat.NewVecDense(stateDim, []float64{1, 2, 3, 4})
	initCov := utils.EyeMatrix(stateDim)
	kf, _ := core.NewKalmanFilterND(stateDim, measDim, initState, initCov)

	assert.Equal(t, initState, kf.State())
}

func TestKF_CovMatrixRetrieval(t *testing.T) {
	stateDim, measDim := 4, 2
	initState := mat.NewVecDense(stateDim, nil)
	initCov := utils.EyeMatrix(stateDim)
	kf, _ := core.NewKalmanFilterND(stateDim, measDim, initState, initCov)

	assert.Equal(t, initCov, kf.CovMatrix())
}

func TestKF_InnovationRetrieval(t *testing.T) {
	stateDim, measDim := 4, 2
	kf, _ := core.NewKalmanFilterND(stateDim, measDim, nil, nil)

	assert.NotNil(t, kf.Innovation())
}

func TestKF_InnovationCovRetrieval(t *testing.T) {
	stateDim, measDim := 4, 2
	kf, _ := core.NewKalmanFilterND(stateDim, measDim, nil, nil)

	assert.NotNil(t, kf.InnovationCov())
}

func TestKF_KalmanGainRetrieval(t *testing.T) {
	stateDim, measDim := 4, 2
	kf, _ := core.NewKalmanFilterND(stateDim, measDim, nil, nil)

	assert.NotNil(t, kf.KalmanGain())
}

func TestKF_SetQ(t *testing.T) {
	stateDim, measDim := 4, 2
	kf, _ := core.NewKalmanFilterND(stateDim, measDim, nil, nil)

	validQ := utils.EyeMatrix(stateDim)
	invalidQ := utils.EyeMatrix(stateDim - 1)

	assert.NoError(t, kf.SetQ(validQ))
	assert.Error(t, kf.SetQ(invalidQ))
}

func TestKF_SetR(t *testing.T) {
	stateDim, measDim := 4, 2
	kf, _ := core.NewKalmanFilterND(stateDim, measDim, nil, nil)

	validR := utils.EyeMatrix(measDim)
	invalidR := utils.EyeMatrix(measDim + 1)

	assert.NoError(t, kf.SetR(validR))
	assert.Error(t, kf.SetR(invalidR))
}

func TestKF_SetF(t *testing.T) {
	stateDim, measDim := 4, 2
	kf, _ := core.NewKalmanFilterND(stateDim, measDim, nil, nil)

	validF := utils.EyeMatrix(stateDim)
	invalidF := utils.EyeMatrix(stateDim - 1)

	assert.NoError(t, kf.SetF(validF))
	assert.Error(t, kf.SetF(invalidF))
}

func TestKF_SetH(t *testing.T) {
	stateDim, measDim := 4, 2
	kf, _ := core.NewKalmanFilterND(stateDim, measDim, nil, nil)

	validH := mat.NewDense(measDim, stateDim, nil)
	invalidH := mat.NewDense(measDim-1, stateDim, nil)

	assert.NoError(t, kf.SetH(validH))
	assert.Error(t, kf.SetH(invalidH))
}

func TestKF_SetStateLimiter(t *testing.T) {
	stateDim, measDim := 4, 2
	kf, _ := core.NewKalmanFilterND(stateDim, measDim, nil, nil)

	minState := []float64{-1, -1, -1, -1}
	maxState := []float64{1, 1, 1, 1}
	invalidMinState := []float64{-1, -1}

	assert.NoError(t, kf.SetStateLimiter(minState, maxState))
	assert.Error(t, kf.SetStateLimiter(invalidMinState, maxState))
}
