package utils

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/mat"
)

type StateLimiter struct {
	minStateValue []float64
	maxStateValue []float64
}

func NewStateLimiter(minStateValue, maxStateValue []float64) (*StateLimiter, error) {
	// Ensure valid min/max range
	for i := range minStateValue {
		if minStateValue[i] > maxStateValue[i] {
			return nil, errors.New("invalid min and max state vectors as min state value is greater than max state value")
		}
	}
	return &StateLimiter{
		minStateValue: minStateValue,
		maxStateValue: maxStateValue,
	}, nil
}

func (s *StateLimiter) Limit(x *mat.VecDense) {
	for i := 0; i < x.Len(); i++ {
		value := x.AtVec(i)
		if s.minStateValue != nil {
			value = math.Max(s.minStateValue[i], value)
		}
		if s.maxStateValue != nil {
			value = math.Min(s.maxStateValue[i], value)
		}
		x.SetVec(i, value)
	}
}
