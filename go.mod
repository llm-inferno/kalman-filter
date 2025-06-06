module github.com/llm-inferno/kalman-filter

go 1.23.2

replace kalman-filter => ./

require (
	github.com/stretchr/testify v1.10.0
	gonum.org/v1/gonum v0.15.1
	kalman-filter v0.0.0-00010101000000-000000000000
)

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
