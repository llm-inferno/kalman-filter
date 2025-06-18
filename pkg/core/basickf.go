package core

import (
	"errors"

	"github.com/llm-inferno/kalman-filter/pkg/utils"

	"gonum.org/v1/gonum/mat"
)

type KalmanFilterND struct {
	Xdim         int                 // dimension of state vector
	Zdim         int                 // dimension of measurement vector
	X            *mat.VecDense       // State vector (Xdim)
	P            *mat.Dense          // Estimate uncertainty covariance (Xdim x Xdim)
	F            *mat.Dense          // State transition matrix (Xdim x Xdim)
	H            *mat.Dense          // Measurement matrix (Zdim x Xdim)
	Q            *mat.Dense          // Process noise (Xdim x Xdim)
	R            *mat.Dense          // Measurement noise (Zdim x Zdim)
	Y            *mat.VecDense       // innovation vector (Zdim)
	S            *mat.Dense          // innovation covariance matrix(Zdim x Zdim)
	K            *mat.Dense          // Kalman Gain matrix (Xdim x Zdim)
	StateLimiter *utils.StateLimiter // state limiter
}

func NewKalmanFilterND(stateDim, measDim int, initState *mat.VecDense, initCov *mat.Dense) (*KalmanFilterND, error) {
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

	// default state transition matrix F (identity matrix)
	stateTransitionMatrix := utils.EyeMatrix(stateDim)

	// Default measurement matrix (zero matrix) of measDim x stateDim
	measurementMatrix := mat.NewDense(measDim, stateDim, nil)

	innovationVec := mat.NewVecDense(measDim, nil)
	innovationCov := mat.NewDense(measDim, measDim, nil)
	kalmanGain := mat.NewDense(stateDim, measDim, nil)

	return &KalmanFilterND{
		Xdim:         stateDim,
		Zdim:         measDim,
		X:            initState,
		P:            initCov,
		Q:            processNoise,
		R:            measNoise,
		F:            stateTransitionMatrix,
		H:            measurementMatrix,
		Y:            innovationVec,
		S:            innovationCov,
		K:            kalmanGain,
		StateLimiter: nil,
	}, nil
}

func (kf *KalmanFilterND) Predict(Q *mat.Dense) error {
	err := kf.SetQ(Q)
	if err != nil {
		return errors.New("new process noise covariance could not be set in the prediction step")
	}
	// Predicted state estimate x_k|k-1 = F(x_k-1|k-1)
	kf.X.MulVec(kf.F, kf.X)
	if kf.StateLimiter != nil {
		kf.StateLimiter.Limit(kf.X)
	}

	// Predicted covariance estimate P_k|k-1 = F_k * P_k-1|k-1 * F_k^T + Q_k-1
	P_pred := mat.NewDense(kf.P.RawMatrix().Rows, kf.P.RawMatrix().Cols, nil)
	P_pred.Product(kf.F, kf.P, kf.F.T())
	P_pred.Add(P_pred, kf.Q)

	kf.P = P_pred
	return nil
}

func (kf *KalmanFilterND) Update(z *mat.VecDense, R *mat.Dense) error {
	if z.Len() != kf.Zdim {
		return errors.New("new measurement vector dimension does not match initial measurement vector")
	}
	if err := kf.SetR(R); err != nil {
		return errors.New("new measurement noise covariance could not be set in the update step")
	}

	// Compute innovation y_k = z_k - H(x_k|k-1)
	Hx := mat.NewVecDense(kf.H.RawMatrix().Rows, nil)
	Hx.MulVec(kf.H, kf.X)
	kf.Y.SubVec(z, Hx)

	// Compute innovation covariance S_k = H_k * P_k|k-1 * H_k^T + R_k
	kf.S.Product(kf.H, kf.P, kf.H.T())
	kf.S.Add(kf.S, kf.R)

	// Compute inverse of S
	S_inv := mat.NewDense(kf.S.RawMatrix().Rows, kf.S.RawMatrix().Cols, nil)
	if err := S_inv.Inverse(kf.S); err != nil {
		return errors.New("singular matrix encountered during inversion")
	}

	// Compute Kalman Gain K_k = P_k|k-1 * H_k^T *  S_k^(-1)
	kf.K.Product(kf.P, kf.H.T(), S_inv)

	// Update state estimate x_k|k = x_k|k-1 + K_k * y_k
	Ky := mat.NewVecDense(kf.K.RawMatrix().Rows, nil)
	Ky.MulVec(kf.K, kf.Y)
	kf.X.AddVec(kf.X, Ky)
	if kf.StateLimiter != nil {
		kf.StateLimiter.Limit(kf.X)
	}

	// Update covariance estimate P_k|k = (I - K_k * H_k) * P_k|k-1
	I := utils.EyeMatrix(kf.P.RawMatrix().Rows)
	KH := utils.MultiplyMatrices(kf.K, kf.H)
	I.Sub(I, KH)
	kf.P.Mul(I, kf.P)

	return nil
}

func (kf *KalmanFilterND) State() *mat.VecDense {
	return kf.X
}

func (kf *KalmanFilterND) CovMatrix() *mat.Dense {
	return kf.P
}

func (kf *KalmanFilterND) Innovation() *mat.VecDense {
	return kf.Y
}

func (kf *KalmanFilterND) InnovationCov() *mat.Dense {
	return kf.S
}

func (kf *KalmanFilterND) KalmanGain() *mat.Dense {
	return kf.K
}

func (kf *KalmanFilterND) SetQ(Q *mat.Dense) error {
	if Q.RawMatrix().Rows != kf.Xdim || Q.RawMatrix().Cols != kf.Xdim {
		return errors.New("process noise matrix Q must have dimensions Xdim x Xdim")
	}
	kf.Q = Q
	return nil
}

func (kf *KalmanFilterND) SetR(R *mat.Dense) error {
	if R.RawMatrix().Rows != kf.Zdim || R.RawMatrix().Cols != kf.Zdim {
		return errors.New("measurement noise matrix R must have dimensions Zdim x Zdim")
	}
	kf.R = R
	return nil
}

func (kf *KalmanFilterND) SetF(F *mat.Dense) error {
	if F.RawMatrix().Rows != kf.Xdim || F.RawMatrix().Cols != kf.Xdim {
		return errors.New("the state transition matrix F doesn't match the state dimensions")
	}
	kf.F = F
	return nil
}

func (kf *KalmanFilterND) SetH(H *mat.Dense) error {
	if H.RawMatrix().Rows != kf.Zdim || H.RawMatrix().Cols != kf.Xdim {
		return errors.New("the measurement matrix H has incorrect dimensions")
	}
	kf.H = H
	return nil
}

func (kf *KalmanFilterND) SetStateLimiter(minStateValue, maxStateValue []float64) error {
	if minStateValue == nil && maxStateValue == nil {
		kf.StateLimiter = nil
		return nil
	}

	if (minStateValue != nil && len(minStateValue) != kf.Xdim) ||
		(maxStateValue != nil && len(maxStateValue) != kf.Xdim) {
		return errors.New("minStateValue/maxStateValue length mismatch with state vector")
	}

	// set state limiter and limit initial state
	stateLimiter, err := utils.NewStateLimiter(minStateValue, maxStateValue)
	if err != nil {
		return err
	}
	kf.StateLimiter = stateLimiter
	kf.StateLimiter.Limit(kf.X)

	return nil
}
