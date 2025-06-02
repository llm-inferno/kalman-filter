package tests

import (
	"testing"

	"github.com/llm-inferno/kalman-filter/pkg/utils"

	"gonum.org/v1/gonum/mat"
)

func TestStateLimiter_Limit(t *testing.T) {
	tests := []struct {
		name           string
		minStateValue  []float64
		maxStateValue  []float64
		input          []float64
		expectedOutput []float64
	}{
		{
			name:           "Within Limits",
			minStateValue:  []float64{-1, 0, -1},
			maxStateValue:  []float64{10, 10, 10},
			input:          []float64{1, 2, 3},
			expectedOutput: []float64{1, 2, 3},
		},
		{
			name:           "Test upper limit",
			minStateValue:  []float64{-1, 0, -1},
			maxStateValue:  []float64{10, 10, 10},
			input:          []float64{15, 2, 11},
			expectedOutput: []float64{10, 2, 10},
		},
		{
			name:           "Test lower limit",
			minStateValue:  []float64{-1, 0, -1},
			maxStateValue:  []float64{10, 10, 10},
			input:          []float64{1, -5, -1},
			expectedOutput: []float64{1, 0, -1},
		},
		{
			name:           "Test both limits",
			minStateValue:  []float64{-1, 0, -1},
			maxStateValue:  []float64{10, 10, 10},
			input:          []float64{-5, 12, 7},
			expectedOutput: []float64{-1, 10, 7},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stateLimiter, err := utils.NewStateLimiter(tt.minStateValue, tt.maxStateValue)
			if err != nil {
				t.Errorf("Error in creating the state limiter %s", err)
			}

			x := mat.NewVecDense(len(tt.input), tt.input)
			stateLimiter.Limit(x)
			for i := 0; i < len(tt.input); i++ {
				if x.AtVec(i) != tt.expectedOutput[i] {
					t.Errorf("Expected %v, got %v", tt.expectedOutput, x.RawVector().Data)
					break
				}
			}

		})
	}
}

func TestStateLimiter_InvalidMinMax(t *testing.T) {
	minStateValue := []float64{-1, 12, -1}
	maxStateValue := []float64{10, 10, 10}

	_, err := utils.NewStateLimiter(minStateValue, maxStateValue)

	if err == nil {
		t.Errorf("Expected an error in initializing state limiter, but got nil")
	}
}
