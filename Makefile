.PHONY: all build test clean profile analyze

BINARY_NAME=squava
ITERATIONS=100000

all: build

build:
	go build -o $(BINARY_NAME) .

test:
	go test -v .

clean:
	go clean
	rm -f $(BINARY_NAME) squava_opt *.prof

profile: build
	./$(BINARY_NAME) -p1 mcts -p2 mcts -p3 mcts -iterations $(ITERATIONS) -cpuprofile cpu.prof
	go tool pprof -top cpu.prof

analyze:
	go tool pprof -list RunSimulation cpu.prof
