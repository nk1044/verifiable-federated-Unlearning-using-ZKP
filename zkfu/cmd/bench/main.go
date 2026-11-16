// cmd/bench: proving / verification micro-benchmark on one client (spec §10).
//
//	go run ./cmd/bench -keys keys -runs 20 [-dataset sim/work/data/client_000.gob] -json results/bench.json
//
// Without -dataset it uses a random dataset at realistic feature scale.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"runtime"
	"time"

	"zkfu/commit"
	"zkfu/params"
	"zkfu/protocol"
)

func stats(xs []float64) (mean, std, min, max float64) {
	min, max = math.Inf(1), math.Inf(-1)
	for _, x := range xs {
		mean += x
		min, max = math.Min(min, x), math.Max(max, x)
	}
	mean /= float64(len(xs))
	for _, x := range xs {
		std += (x - mean) * (x - mean)
	}
	return mean, math.Sqrt(std / float64(len(xs))), min, max
}

func main() {
	keys := flag.String("keys", "keys", "key directory")
	runs := flag.Int("runs", 20, "number of proofs")
	dsPath := flag.String("dataset", "", "optional client dataset (.gob)")
	out := flag.String("json", "", "write results JSON here")
	flag.Parse()
	r := rand.New(rand.NewPCG(1, 1))

	var ds *commit.Dataset
	var err error
	if *dsPath != "" {
		if ds, err = commit.LoadDataset(*dsPath); err != nil {
			log.Fatal(err)
		}
	} else {
		f := make([][]int64, 1000)
		y := make([]int64, 1000)
		for i := range f {
			f[i] = make([]int64, params.D)
			for d := range f[i] {
				f[i][d] = r.Int64N(1<<16) - 1<<15 // ~ +-0.5 in real units
			}
			y[i] = r.Int64N(params.C)
		}
		if ds, err = commit.NewDataset(0, f, y); err != nil {
			log.Fatal(err)
		}
	}
	W := make([][]int64, params.C)
	b := make([]int64, params.C)
	for c := range W {
		W[c] = make([]int64, params.D)
		for d := range W[c] {
			W[c][d] = r.Int64N(1<<15) - 1<<14
		}
	}

	res := map[string]any{"runs": *runs, "cpus": runtime.NumCPU(), "goos": runtime.GOOS, "goarch": runtime.GOARCH}
	t := time.Now()
	ccs, err := protocol.Compile()
	if err != nil {
		log.Fatal(err)
	}
	res["constraints"], res["compile_s"] = ccs.GetNbConstraints(), time.Since(t).Seconds()
	t = time.Now()
	pk, err := protocol.LoadPK(*keys)
	if err != nil {
		log.Fatal(err)
	}
	res["pk_load_s"] = time.Since(t).Seconds()
	vk, err := protocol.LoadVK(*keys)
	if err != nil {
		log.Fatal(err)
	}
	t = time.Now()
	tree, err := ds.Tree()
	if err != nil {
		log.Fatal(err)
	}
	res["merkle_tree_build_ms"] = float64(time.Since(t).Microseconds()) / 1000

	var native, witness, prove, verify []float64
	var proofLen int
	var peak uint64
	for i := 0; i < *runs; i++ {
		forgot := map[int]bool{}
		if i%2 == 1 { // alternate FL-style and FU-style steps
			for k := 0; k < ds.NReal/50; k++ {
				forgot[r.IntN(ds.NReal)] = true
			}
		}
		t = time.Now()
		st, err := protocol.HonestStep(ds, W, b, uint64(i), forgot)
		if err != nil {
			log.Fatal(err)
		}
		native = append(native, float64(time.Since(t).Microseconds())/1000)
		t = time.Now()
		a, err := st.Assignment(tree)
		if err != nil {
			log.Fatal(err)
		}
		witness = append(witness, float64(time.Since(t).Microseconds())/1000)
		t = time.Now()
		proof, err := protocol.Prove(ccs, pk, a)
		if err != nil {
			log.Fatal(err)
		}
		prove = append(prove, time.Since(t).Seconds())
		proofLen = len(proof)
		pub, err := protocol.ServerPublic(tree.Root(), ds.NReal, W, b, st.W2, st.B2, uint64(i), uint64(ds.ClientID), forgot)
		if err != nil {
			log.Fatal(err)
		}
		t = time.Now()
		if err := protocol.Verify(vk, proof, pub); err != nil {
			log.Fatalf("verify failed: %v", err)
		}
		verify = append(verify, float64(time.Since(t).Microseconds())/1000)
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		if m.Sys > peak {
			peak = m.Sys
		}
		W, b = st.W2, st.B2 // continue training so every run is a fresh, realistic step
	}
	pm, ps, pmin, pmax := stats(prove)
	vm, vs, _, _ := stats(verify)
	nm, _, _, _ := stats(native)
	wm, _, _, _ := stats(witness)
	res["prove_s_mean"], res["prove_s_std"], res["prove_s_min"], res["prove_s_max"] = pm, ps, pmin, pmax
	res["verify_ms_mean"], res["verify_ms_std"] = vm, vs
	res["native_step_ms_mean"], res["witness_build_ms_mean"] = nm, wm
	res["proof_bytes"] = proofLen
	res["go_runtime_sys_peak_mb"] = float64(peak) / 1e6
	for name, p := range map[string]string{"pk_bytes": "pk.bin", "vk_bytes": "vk.bin"} {
		if st, err := os.Stat(*keys + "/" + p); err == nil {
			res[name] = st.Size()
		}
	}
	js, _ := json.MarshalIndent(res, "", "  ")
	fmt.Println(string(js))
	if *out != "" {
		if err := os.WriteFile(*out, js, 0o644); err != nil {
			log.Fatal(err)
		}
	}
}
