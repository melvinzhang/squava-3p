.PHONY: all build test clean profile analyze fuzz benchmark wasm serve zip

BINARY_NAME=squava
ITERATIONS=1000000
REPRO_SEED=641728870
GO=/usr/lib/go-1.25/bin/go

all: build

build:
	GOEXPERIMENT=greenteagc $(GO) build -o $(BINARY_NAME) .

wasm:
	mkdir -p web/public
	cp /usr/share/go-1.25/lib/wasm/wasm_exec.js web/public/
	GOOS=js GOARCH=wasm $(GO) build -o web/public/squava.wasm .

zip: wasm
	rm -f game.zip
	zip -j game.zip web/public/*

serve: wasm
	@echo "Serving at http://localhost:8080"
	python3 -m http.server 8080 --directory web/public

test:
	$(GO) test -v .

fuzz:
	@for f in $$(go test -list Fuzz . | grep ^Fuzz); do \
		echo "Running $$f..."; \
		go test -v -fuzz=$$f -fuzztime=5s . || exit 1; \
	done

repro_game_%.log: build
	./$(BINARY_NAME) -p1 mcts -p2 mcts -p3 mcts -iterations 1000000 -seed $* | tee $@

benchmark: build
	@echo "Starting benchmark: 100 games with 1M iterations..."
	@mkdir -p logs
	@for i in $$(seq 100); do \
		seed=$$(od -An -N4 -tu4 /dev/urandom | tr -d ' \n'); \
		echo "Running game $$i/100 with seed $$seed..." ; \
		./$(BINARY_NAME) -p1 mcts -p2 mcts -p3 mcts -iterations $(ITERATIONS) -seed $$seed > logs/game_$$seed.tmp 2>&1 && \
		mv logs/game_$$seed.tmp logs/game_$$seed.log ; \
	done
	@echo "Benchmark complete. Results saved to logs/"

clean:
	go clean
	rm -f $(BINARY_NAME) squava_opt *.prof game.zip

profile: build
	./$(BINARY_NAME) -p1 mcts -p2 mcts -p3 mcts -iterations $(ITERATIONS) -seed $(REPRO_SEED) -cpuprofile cpu.prof | tee repro_game_$(REPRO_SEED).log
	go tool pprof -top cpu.prof

pprof:
	go tool pprof -list RunSimulation cpu.prof
	go tool pprof -list selectBestEdge cpu.prof

analyze:
	python3 analyze_log.py logs/*.log
