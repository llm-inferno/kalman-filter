package core

import (
	"errors"

	"kalman-filter/pkg/utils"

	"gonum.org/v1/gonum/mat"
)

type ExtendedKalmanFilter struct {
	Xdim         int                               // dimension of state vector
	Zdim         int                               // dimension of measurement vector
	X            *mat.VecDense                     // State vector (Xdim)
	P            *mat.Dense                        // Estimate uncertainty covariance (Xdim x Xdim)
	Q            *mat.Dense                        // Process noise (Xdim x Xdim)
	R            *mat.Dense                        // Measurement noise (Zdim x Zdim)
	f            func(*mat.VecDense) *mat.VecDense // Nonlinear state transition
	h            func(*mat.VecDense) *mat.VecDense // Nonlinear measurement function
	F            func(*mat.VecDense) *mat.Dense    // Jacobian of f
	H            func(*mat.VecDense) *mat.Dense    // Jacobian of h
	Y            *mat.VecDense                     // innovation vector (Zdim)
	S            *mat.Dense                        // innovation covariance matrix(Zdim x Zdim)
	K            *mat.Dense                        // Kalman Gain matrix (Xdim x Zdim)
	StateLimiter *utils.StateLimiter               // state limiter
}

func NewExtendedKalmanFilter(stateDim, measDim int, initState *mat.VecDense, initCov *mat.Dense) (*ExtendedKalmanFilter, error) {
	if stateDim <= 0 || measDim <= 0 {
		return nil, errors.New("state and measurement dimensions must be positive")
	}

	if initState != nil && initState.Len() != stateDim {
		return nil, errors.New("initial state vector dimension must match state dimension")
	}

	if initCov != nil {
		if initCov.RawMatrix().Rows != stateDim || initCov.RawMatrix().Cols != stateDim {
			return nil, errors.New("initial covariance matrix must be square with dimensions stateDim x stateDim")
		}
	}

	// create a default initial state and initial covariance if not supplied by the user
	if initState == nil {
		initState = mat.NewVecDense(stateDim, nil)
	}
	if initCov == nil {
		initCov = utils.EyeMatrix(stateDim)
	}

	// create an empty Process Noise matrix of stateDim x stateDim
	processNoise := mat.NewDense(stateDim, stateDim, nil)

	// create an empty Measurement Noise matrix of measDim x measDim
	measNoise := mat.NewDense(measDim, measDim, nil)

	// Default state transition function (identity function)
	stateTransitionFunc := func(x *mat.VecDense) *mat.VecDense {
		out := mat.NewVecDense(stateDim, nil)
		out.CopyVec(x)
		return out
	}

	// Default measurement function (maps state to a zero measurement vector)
	measurementFunc := func(x *mat.VecDense) *mat.VecDense {
		return mat.NewVecDense(measDim, nil)
	}

	// Default Jacobian of f (identity matrix)
	stateTransitionJacobianFunc := func(x *mat.VecDense) *mat.Dense {
		return utils.EyeMatrix(stateDim)
	}

	// Default Jacobian of h (zero matrix)
	measurementJacobianFunc := func(x *mat.VecDense) *mat.Dense {
		return mat.NewDense(measDim, stateDim, nil)
	}

	innovationVec := mat.NewVecDense(measDim, nil)
	innovationCov := mat.NewDense(measDim, measDim, nil)
	kalmanGain := mat.NewDense(stateDim, measDim, nil)

	return &ExtendedKalmanFilter{
		Xdim:         stateDim,
		Zdim:         measDim,
		X:            initState,
		P:            initCov,
		Q:            processNoise,
		R:            measNoise,
		f:            stateTransitionFunc,
		h:            measurementFunc,
		F:            stateTransitionJacobianFunc,
		H:            measurementJacobianFunc,
		Y:            innovationVec,
		S:            innovationCov,
		K:            kalmanGain,
		StateLimiter: nil,
	}, nil
}

func (ekf *ExtendedKalmanFilter) Predict(Q *mat.Dense) error {
	err := ekf.SetQ(Q)
	if err != nil {
		return errors.New("new process noise covariance could not be set in the prediction step")
	}
	//evaluate Jacobian F at the priori state estimate
	F := ekf.F(ekf.X)

	// Predicted state estimate x_k|k-1 = f(x_k-1|k-1)
	X_pred := ekf.f(ekf.X)
	// fmt.Println("Predicted value: ", X_pred)
	ekf.X.CloneFromVec(X_pred)
	if ekf.StateLimiter != nil {
		ekf.StateLimiter.Limit(ekf.X)
	}

	// Predicted covariance estimate P_k|k-1 = F_k * P_k-1|k-1 * F_k^T + Q_k-1
	P_pred := mat.NewDense(ekf.P.RawMatrix().Rows, ekf.P.RawMatrix().Cols, nil)
	P_pred.Product(F, ekf.P, F.T())
	P_pred.Add(P_pred, ekf.Q)

	ekf.P = P_pred
	return nil
}

func (ekf *ExtendedKalmanFilter) Update(z *mat.VecDense, R *mat.Dense) error {
	if z.Len() != ekf.Zdim {
		return errors.New("new measurement vector dimension does not match initial measurement vector")
	}
	if err := ekf.SetR(R); err != nil {
		return errors.New("new measurement noise covariance could not be set in the update step")
	}

	// Compute innovation y_k = z_k - h(x_k|k-1)
	hx := ekf.h(ekf.X)
	ekf.Y.SubVec(z, hx)

	// Compute measurement Jacobian H_k
	H := ekf.H(ekf.X)

	// Compute innovation covariance S_k = H_k * P_k|k-1 * H_k^T + R_k
	ekf.S.Product(H, ekf.P, H.T())
	ekf.S.Add(ekf.S, ekf.R)

	// Compute inverse of S
	S_inv := mat.NewDense(ekf.S.RawMatrix().Rows, ekf.S.RawMatrix().Cols, nil)
	if err := S_inv.Inverse(ekf.S); err != nil {
		return errors.New("singular matrix encountered during inversion")
	}

	// Compute Kalman Gain K_k = P_k|k-1 * H_k^T *  S_k^(-1)
	ekf.K.Product(ekf.P, H.T(), S_inv)

	// Update state estimate x_k|k = x_k|k-1 + K_k * y_k
	Ky := mat.NewVecDense(ekf.X.Len(), nil)
	Ky.MulVec(ekf.K, ekf.Y)
	ekf.X.AddVec(ekf.X, Ky)
	if ekf.StateLimiter != nil {
		ekf.StateLimiter.Limit(ekf.X)
	}

	// Update covariance estimate P_k|k = (I - K_k * H_k) * P_k|k-1
	I := utils.EyeMatrix(ekf.P.RawMatrix().Rows)
	KH := utils.MultiplyMatrices(ekf.K, H)
	I.Sub(I, KH)
	ekf.P.Mul(I, ekf.P)

	return nil
}

func (ekf *ExtendedKalmanFilter) State() *mat.VecDense {
	return ekf.X
}

func (ekf *ExtendedKalmanFilter) CovMatrix() *mat.Dense {
	return ekf.P
}

func (ekf *ExtendedKalmanFilter) Qmatrix() *mat.Dense {
	return ekf.Q
}

func (ekf *ExtendedKalmanFilter) Rmatrix() *mat.Dense {
	return ekf.R
}

func (ekf *ExtendedKalmanFilter) Innovation() *mat.VecDense {
	return ekf.Y
}

func (ekf *ExtendedKalmanFilter) InnovationCov() *mat.Dense {
	return ekf.S
}

func (ekf *ExtendedKalmanFilter) KalmanGain() *mat.Dense {
	return ekf.K
}

func (ekf *ExtendedKalmanFilter) StateJacobian() *mat.Dense {
	return ekf.F(ekf.X)
}

func (ekf *ExtendedKalmanFilter) MeasJacobian() *mat.Dense {
	return ekf.H(ekf.X)
}

func (ekf *ExtendedKalmanFilter) SetQ(Q *mat.Dense) error {
	if Q.RawMatrix().Rows != ekf.Xdim || Q.RawMatrix().Cols != ekf.Xdim {
		return errors.New("process noise matrix Q must have dimensions Xdim x Xdim")
	}
	ekf.Q = Q
	return nil
}

func (ekf *ExtendedKalmanFilter) SetR(R *mat.Dense) error {
	if R.RawMatrix().Rows != ekf.Zdim || R.RawMatrix().Cols != ekf.Zdim {
		return errors.New("measurement noise matrix R must have dimensions Zdim x Zdim")
	}
	ekf.R = R
	return nil
}

func (ekf *ExtendedKalmanFilter) SetfF(f func(*mat.VecDense) *mat.VecDense) error {
	if f == nil {
		return errors.New("state transition function cannot be nil")
	}

	// Test input-output consistency
	testOutput := f(ekf.X)
	if testOutput.Len() != ekf.Xdim {
		return errors.New("invalid state transition function: output dimension must match state dimension")
	}
	ekf.f = f

	// Automatically set F as a function that computes the Jacobian dynamically
	ekf.F = func(x *mat.VecDense) *mat.Dense {
		return utils.NumericalJacobian(f, x, x.Len())
	}
	return nil
}

func (ekf *ExtendedKalmanFilter) SethH(h func(*mat.VecDense) *mat.VecDense) error {
	if h == nil {
		return errors.New("measurement function cannot be nil")
	}

	// Test input-output consistency
	testOutput := h(ekf.X)
	if testOutput.Len() != ekf.Zdim {
		return errors.New("invalid measurement function: output dimension must match measurement dimension")
	}
	ekf.h = h

	// Automatically set H as a function that computes the Jacobian dynamically
	ekf.H = func(x *mat.VecDense) *mat.Dense {
		return utils.NumericalJacobian(h, x, ekf.Zdim)
	}
	return nil
}

func (ekf *ExtendedKalmanFilter) SetStateLimiter(minStateValue, maxStateValue []float64) error {
	if minStateValue == nil && maxStateValue == nil {
		ekf.StateLimiter = nil
		return nil
	}

	if (minStateValue != nil && len(minStateValue) != ekf.Xdim) ||
		(maxStateValue != nil && len(maxStateValue) != ekf.Xdim) {
		return errors.New("minStateValue/maxStateValue length mismatch with state vector")
	}

	// set state limiter and limit initial state
	stateLimiter, err := utils.NewStateLimiter(minStateValue, maxStateValue)
	if err != nil {
		return err
	}
	ekf.StateLimiter = stateLimiter
	ekf.StateLimiter.Limit(ekf.X)

	return nil
}
