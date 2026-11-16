package tests

import (
	"math/rand/v2"
	"os"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"

	"zkfu/commit"
	"zkfu/protocol"
)

// Milestone 3a: native output equals the circuit output, bit for bit, on 1000 random steps.
// For each step we set HWNew = H(native W', b') and require the compiled circuit to be satisfied;
// since H is collision resistant, the circuit's W', b' equal the native ones. A few steps also
// check that the native result +1 in one coordinate is NOT accepted.
// ZKFU_EQUIV_STEPS overrides the count (default 1000).
func TestNativeEqualsCircuit1000Steps(t *testing.T) {
	steps := 1000
	if v, err := strconv.Atoi(os.Getenv("ZKFU_EQUIV_STEPS")); err == nil {
		steps = v
	}
	getCCS(t)
	r := rand.New(rand.NewPCG(7, 8))
	// a handful of committed datasets with different feature scales (trees are the slow part)
	type dsT struct {
		ds   *commit.Dataset
		tree *commit.Tree
	}
	var dss []dsT
	for i, sc := range []int64{1 << 10, 1 << 14, 1 << 16, 1 << 17} {
		ds := randDataset(t, r, 100+i, 600+400*i, sc)
		tr, err := ds.Tree()
		if err != nil {
			t.Fatal(err)
		}
		dss = append(dss, dsT{ds, tr})
	}

	type job struct {
		k    int
		step *protocol.ClientStep
		tree *commit.Tree
	}
	jobs := make(chan job)
	var fails, done int64
	var wg sync.WaitGroup
	for w := 0; w < runtime.NumCPU(); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for jb := range jobs {
				a, err := jb.step.Assignment(jb.tree)
				if err != nil {
					t.Error(err)
					atomic.AddInt64(&fails, 1)
					continue
				}
				if err := solve(t, a); err != nil {
					t.Errorf("step %d: circuit != native: %v", jb.k, err)
					atomic.AddInt64(&fails, 1)
				}
				if jb.k%100 == 0 { // negative control: native output +1 is not a valid output
					jb.step.W2[jb.k%10][jb.k%64]++
					bad, err := jb.step.Assignment(jb.tree)
					if err == nil && solve(t, bad) == nil {
						t.Errorf("step %d: circuit accepted a perturbed output", jb.k)
						atomic.AddInt64(&fails, 1)
					}
				}
				atomic.AddInt64(&done, 1)
			}
		}()
	}
	for k := 0; k < steps; k++ {
		d := dss[k%len(dss)]
		wScale := []int64{1 << 12, 1 << 15, 1 << 16, 1 << 17}[r.IntN(4)]
		W, b := randHead(r, wScale)
		var forgot map[int]bool
		if k%2 == 1 { // FU-style steps: forget a random 20% of real positions
			forgot = map[int]bool{}
			for i := 0; i < d.ds.NReal; i++ {
				if r.IntN(5) == 0 {
					forgot[i] = true
				}
			}
		}
		st, err := protocol.HonestStep(d.ds, W, b, uint64(k), forgot)
		if err != nil {
			t.Fatalf("step %d: native rejected in-range inputs: %v", k, err)
		}
		jobs <- job{k, st, d.tree}
	}
	close(jobs)
	wg.Wait()
	t.Logf("%d/%d steps bit-identical, %d failures", done, steps, fails)
}
