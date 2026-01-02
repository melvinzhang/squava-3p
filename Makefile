.PHONY: all build test clean profile analyze fuzz benchmark

BINARY_NAME=squava
ITERATIONS=1000000
FUZZ_ITERS=100000

all: build

build:
	go build -o $(BINARY_NAME) .

test:
	go test -v .

fuzz:
	go test -v . -args -fuzz_iters=$(FUZZ_ITERS)

repro_game_%.log:
	./squava -p1 mcts -p2 mcts -p3 mcts -iterations 1000000 -seed $* | tee $@

benchmark: build
	@echo "Starting benchmark: 100 games with 1M iterations..."
	@rm -f log_1M
	@for i in $$(seq 100); do \
		echo "Running game $$i/100..." ; \
		./$(BINARY_NAME) -p1 mcts -p2 mcts -p3 mcts -iterations $(ITERATIONS) >> log_1M 2>&1 ; \
	done
	@echo "Benchmark complete. Results saved to log_1M"

clean:
	go clean
	rm -f $(BINARY_NAME) squava_opt *.prof

profile: build
	./$(BINARY_NAME) -p1 mcts -p2 mcts -p3 mcts -iterations $(ITERATIONS) -cpuprofile cpu.prof
	go tool pprof -top cpu.prof

analyze:
	go tool pprof -list RunSimulation cpu.prof
	go tool pprof -list Select cpu.prof
