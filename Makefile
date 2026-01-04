.PHONY: all build test clean profile analyze fuzz benchmark

BINARY_NAME=squava
ITERATIONS=1000000
FUZZ_ITERS=100000
REPRO_SEED=641728870

all: build

build:
	go build -o $(BINARY_NAME) .

test:
	go test -v .

fuzz:
	go test -v . -args -fuzz_iters=$(FUZZ_ITERS)

repro_game_%.log: build
	./$(BINARY_NAME) -p1 mcts -p2 mcts -p3 mcts -iterations 1000000 -seed $* | tee $@

benchmark: build
	@echo "Starting benchmark: 100 games with 1M iterations..."
	@mkdir -p logs
	@for i in $$(seq 100); do \
		seed=$$(od -An -N4 -tu4 /dev/urandom | tr -d ' \n'); \
		echo "Running game $$i/100 with seed $$seed..." ; \
		./$(BINARY_NAME) -p1 mcts -p2 mcts -p3 mcts -iterations $(ITERATIONS) -seed $$seed > logs/game_$$seed.log 2>&1 ; \
	done
	@echo "Benchmark complete. Results saved to logs/"

clean:
	go clean
	rm -f $(BINARY_NAME) squava_opt *.prof

profile: build
	./$(BINARY_NAME) -p1 mcts -p2 mcts -p3 mcts -iterations $(ITERATIONS) -seed $(REPRO_SEED) -cpuprofile cpu.prof | tee repro_game_$(REPRO_SEED).log
	go tool pprof -top cpu.prof

pprof:
	go tool pprof -list RunSimulation cpu.prof
	go tool pprof -list Select cpu.prof

analyze:
	python3 analyze_log.py logs/*
