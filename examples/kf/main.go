package main

import (
	"fmt"
	"math"
	"math/rand"

	"kalman-filter/pkg/core"
	"kalman-filter/pkg/utils"

	"gonum.org/v1/gonum/mat"
)

func main() {
	// Simulation parameters
	dt := 0.2
	simTime := 20.0
	numSteps := int(simTime/dt) + 1
	stateDim := 4
	measDim := 2

	// Initial true state [x, vx, y, vy]
	trueInitialState := mat.NewVecDense(stateDim, []float64{30, 2, 40, 2})

	// Process noise covariance Q
	Q := mat.NewDense(stateDim, stateDim, []float64{
		0, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 1,
	})

	// Measurement noise covariance R
	R := mat.NewDense(measDim, measDim, []float64{
		4, 0,
		0, 4,
	})

	// State transition matrix F
	F := mat.NewDense(stateDim, stateDim, []float64{
		1, dt, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, dt,
		0, 0, 0, 1,
	})

	// Measurement matrix H
	H := mat.NewDense(measDim, stateDim, []float64{
		1, 0, 0, 0,
		0, 0, 1, 0,
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
		noiseVec := mat.NewVecDense(stateDim, []float64{
			rand.NormFloat64(), rand.NormFloat64(), rand.NormFloat64(), rand.NormFloat64(),
		})

		// Compute process noise: sqrt(Q) * noiseVec
		processNoise := mat.NewVecDense(stateDim, nil)
		processNoise.MulVec(Qsqrt, noiseVec)

		// Compute true state: x_k = F * x_k-1 + processNoise
		trueState := mat.NewVecDense(stateDim, nil)
		trueState.MulVec(F, trueStates[i-1])
		trueState.AddVec(trueState, processNoise)
		trueStates[i] = trueState

		// Generate measurement noise
		noiseMeas := mat.NewVecDense(measDim, []float64{
			rand.NormFloat64(),
			rand.NormFloat64(),
		})
		measurementNoise := mat.NewVecDense(measDim, nil)
		measurementNoise.MulVec(Rsqrt, noiseMeas)

		// Compute measurement: z_k = H * x_k + measurementNoise
		measurement := mat.NewVecDense(measDim, nil)
		measurement.MulVec(H, trueState)
		measurement.AddVec(measurement, measurementNoise)
		measurements[i] = measurement
	}

	// Initial estimated state [x, vx, y, vy]
	initState := mat.NewVecDense(stateDim, []float64{40, 0, 160, 0})

	// Initial covariance matrix P (identity matrix)
	initCov := utils.EyeMatrix(stateDim)

	// Create Kalman Filter
	kf, err := core.NewKalmanFilterND(stateDim, measDim, initState, initCov)
	if err != nil {
		fmt.Println("Failed to initialize Kalman Filter:", err)
	}

	// Set initial F, H, Q, R
	if err := kf.SetF(F); err != nil {
		fmt.Println("Error setting state transition matrix.", err)
	}
	if err := kf.SetH(H); err != nil {
		fmt.Println("Error setting measurement matrix.", err)
	}
	if err := kf.SetQ(Q); err != nil {
		fmt.Println("Error setting process noise covariance.", err)
	}
	if err := kf.SetR(R); err != nil {
		fmt.Println("Error setting measurement noise covariance.", err)
	}

	// Storage for estimated states
	estimatedStates := make([]*mat.VecDense, numSteps)
	estimatedStates[0] = initState

	// Print header
	fmt.Println("Time Step |  True x  |  True y  | Estimated x  | Estimated y")
	fmt.Println("------------------------------------------------------------")

	// Print first state
	fmt.Printf("%9d | %7.2f | %7.2f | %11.2f | %11.2f\n",
		0, trueStates[0].AtVec(0), trueStates[0].AtVec(2),
		estimatedStates[0].AtVec(0), estimatedStates[0].AtVec(2))

	// Run Kalman Filter loop
	for i := 1; i < numSteps; i++ {
		// Prediction step
		if err := kf.Predict(Q); err != nil {
			fmt.Printf("Error in prediction step %d: %s", i, err)
		}

		// Update step
		if err := kf.Update(measurements[i], R); err != nil {
			fmt.Printf("Error in update step %d: %s", i, err)
		}

		// Store estimated state
		estimatedStates[i] = kf.State()

		// Print results
		fmt.Printf("%9d | %7.2f | %7.2f | %11.2f | %11.2f\n",
			i, trueStates[i].AtVec(0), trueStates[i].AtVec(2),
			estimatedStates[i].AtVec(0), estimatedStates[i].AtVec(2))
	}
}
