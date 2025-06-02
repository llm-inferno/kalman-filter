package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/llm-inferno/kalman-filter/pkg/core"

	"gonum.org/v1/gonum/mat"
)

// Time step (assumed constant for simplicity)
var dt = 0.2

// state transition function f. Accepts current state x(k-1) and outputs next state x(k)
func stateTransitionFunc(x *mat.VecDense) *mat.VecDense {
	F := mat.NewDense(4, 4, []float64{
		1, dt, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, dt,
		0, 0, 0, 1,
	})

	xNext := mat.NewVecDense(4, nil)
	xNext.MulVec(F, x)
	return xNext
}

// measurement function h. Accepts current state x(k) and outputs observation z(k)
func measurementFunc(x *mat.VecDense) *mat.VecDense {
	px := x.AtVec(0)
	py := x.AtVec(2)

	angle := math.Atan2(py, px)
	rangeVal := math.Hypot(px, py)

	return mat.NewVecDense(2, []float64{angle, rangeVal})
}

func main() {
	// Simulation parameters
	simTime := 20.0
	numSteps := int(simTime/dt) + 1
	stateDim := 4
	measDim := 2

	// Initial true state [x, vx, y, vy]
	trueInitialState := mat.NewVecDense(stateDim, []float64{30, 1, 40, 1})

	// Process noise covariance Q
	Q := mat.NewDense(stateDim, stateDim, []float64{
		0, 0, 0, 0,
		0, 0.01, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0.01,
	})

	// Measurement noise covariance R
	R := mat.NewDense(measDim, measDim, []float64{
		2e-6, 0,
		0, 1,
	})

	// Storage for true states and measurements
	trueStates := make([]*mat.VecDense, numSteps)
	measurements := make([]*mat.VecDense, numSteps)

	// Initialize first true state
	trueStates[0] = trueInitialState

	// Compute element-wise square root of diagonal noise covariance matrices
	Qsqrt := mat.NewDense(stateDim, stateDim, []float64{
		0, 0, 0, 0,
		0, math.Sqrt(Q.At(1, 1)), 0, 0,
		0, 0, 0, 0,
		0, 0, 0, math.Sqrt(Q.At(3, 3)),
	})

	Rsqrt := mat.NewDense(measDim, measDim, []float64{
		math.Sqrt(R.At(0, 0)), 0,
		0, math.Sqrt(R.At(1, 1)),
	})

	// Generate true states and noisy measurements
	for i := 1; i < numSteps; i++ {
		// Generate standard normal noise
		pNoiseVec := mat.NewVecDense(stateDim, []float64{
			rand.NormFloat64(), rand.NormFloat64(), rand.NormFloat64(), rand.NormFloat64(),
		})

		// Compute process noise: sqrt(Q) * noiseVec
		processNoise := mat.NewVecDense(stateDim, nil)
		processNoise.MulVec(Qsqrt, pNoiseVec)

		// Compute true state: x_k = F * x_k-1 + processNoise
		trueState := stateTransitionFunc(trueStates[i-1])
		trueState.AddVec(trueState, processNoise)
		trueStates[i] = trueState

		// Generate measurement noise
		mNoiseVec := mat.NewVecDense(measDim, []float64{
			rand.NormFloat64(),
			rand.NormFloat64(),
		})
		measurementNoise := mat.NewVecDense(measDim, nil)
		measurementNoise.MulVec(Rsqrt, mNoiseVec)

		// Compute measurement: z_k = H * x_k + measurementNoise
		measurement := measurementFunc(trueState)
		measurement.AddVec(measurement, measurementNoise)
		measurements[i] = measurement
	}

	// Initial estimated state [x, vx, y, vy]
	initState := mat.NewVecDense(stateDim, []float64{40, 0, 160, 0})

	// Initial covariance matrix P (identity matrix)
	initCov := mat.NewDense(4, 4, []float64{
		100, 0, 0, 0,
		0, 1000, 0, 0,
		0, 0, 100, 0,
		0, 0, 0, 1000,
	})

	// Create Extended Kalman Filter
	ekf, err := core.NewExtendedKalmanFilter(stateDim, measDim, initState, initCov)
	if err != nil {
		fmt.Println("Failed to initialize Extended Kalmam Filter:", err)
	}

	// Set initial F, H, Q, R
	if err := ekf.SetQ(Q); err != nil {
		fmt.Println("Error setting process noise covariance.", err)
	}
	if err := ekf.SetR(R); err != nil {
		fmt.Println("Error setting measurement noise covariance.", err)
	}
	if err := ekf.SetfF(stateTransitionFunc); err != nil {
		fmt.Println("Error setting state transition function and Jacobian.", err)
	}
	if err := ekf.SethH(measurementFunc); err != nil {
		fmt.Println("Error setting measurement function and Jacobian.", err)
	}

	// Storage for estimated states
	estimatedStates := make([]*mat.VecDense, numSteps)
	estimatedStates[0] = initState

	// Print header for simulation results
	fmt.Println("Time Step |  True x  |  True y  | Measured Angle | Measured Range | Estimated x  | Estimated y")
	fmt.Println("---------------------------------------------------------------------------------------------------")

	// Print first state
	fmt.Printf("%9d | %7.2f | %7.2f | %11.2f | %11.2f\n",
		0, trueStates[0].AtVec(0), trueStates[0].AtVec(2),
		estimatedStates[0].AtVec(0), estimatedStates[0].AtVec(2))

	// Run Extended Kalman Filter loop
	for i := 1; i < numSteps; i++ {
		// Prediction step
		// fmt.Println("Before Prediction: ", ekf.State())
		if err := ekf.Predict(Q); err != nil {
			fmt.Printf("Error in prediction step %d: %s", i, err)
		}

		// Update step
		if err := ekf.Update(measurements[i], R); err != nil {
			fmt.Printf("Error in update step %d: %s", i, err)
		}

		// Store estimated state
		estimatedStates[i] = ekf.State()

		// Print values for verification
		fmt.Printf("%8d | %7.2f | %7.2f | %14.5f | %14.5f | %12.2f | %12.2f\n",
			i, trueStates[i].AtVec(0), trueStates[i].AtVec(2),
			measurements[i].AtVec(0), measurements[i].AtVec(1),
			estimatedStates[i].AtVec(0), estimatedStates[i].AtVec(2))
	}
}
