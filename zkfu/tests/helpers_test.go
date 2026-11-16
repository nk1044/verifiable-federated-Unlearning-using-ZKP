package tests

import (
	"math/big"
	"math/rand/v2"
	"sync"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/frontend"

	"zkfu/circuit"
	"zkfu/commit"
	"zkfu/params"
	"zkfu/protocol"
)

var (
	ccsOnce sync.Once
	ccs     constraint.ConstraintSystem
	keyOnce sync.Once
	pk      groth16.ProvingKey
	vk      groth16.VerifyingKey
)

func getCCS(t testing.TB) constraint.ConstraintSystem {
	ccsOnce.Do(func() {
		var err error
		if ccs, err = protocol.Compile(); err != nil {
			t.Fatal(err)
		}
	})
	return ccs
}

func getKeys(t testing.TB) (groth16.ProvingKey, groth16.VerifyingKey) {
	cs := getCCS(t)
	keyOnce.Do(func() {
		var err error
		if pk, vk, err = groth16.Setup(cs); err != nil {
			t.Fatal(err)
		}
	})
	return pk, vk
}

// solve reports whether a full assignment satisfies the compiled circuit (fast, no proof).
func solve(t testing.TB, a *circuit.ZkStepCircuit) error {
	w, err := frontend.NewWitness(a, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatal(err)
	}
	return getCCS(t).IsSolved(w)
}

func randVal(r *rand.Rand, maxAbs int64) int64 { return r.Int64N(2*maxAbs) - maxAbs }

func randDataset(t testing.TB, r *rand.Rand, cid, nReal int, maxAbs int64) *commit.Dataset {
	f := make([][]int64, nReal)
	y := make([]int64, nReal)
	for i := range f {
		f[i] = make([]int64, params.D)
		for d := range f[i] {
			f[i][d] = randVal(r, maxAbs)
		}
		y[i] = r.Int64N(params.C)
	}
	ds, err := commit.NewDataset(cid, f, y)
	if err != nil {
		t.Fatal(err)
	}
	return ds
}

func randHead(r *rand.Rand, maxAbs int64) ([][]int64, []int64) {
	W := make([][]int64, params.C)
	b := make([]int64, params.C)
	for c := range W {
		W[c] = make([]int64, params.D)
		for d := range W[c] {
			W[c][d] = randVal(r, maxAbs)
		}
		b[c] = randVal(r, maxAbs)
	}
	return W, b
}

func cloneHead(W [][]int64, b []int64) ([][]int64, []int64) {
	W2 := make([][]int64, len(W))
	for c := range W {
		W2[c] = append([]int64(nil), W[c]...)
	}
	return W2, append([]int64(nil), b...)
}

// ---- unchecked encoders, to build commitments to out-of-range values (attacker-side) ----
func packRaw(vals []int64) []fr.Element {
	out := make([]fr.Element, params.NumPacked(len(vals)))
	for k := range out {
		acc := new(big.Int)
		for i := 0; i < params.PackPer && k*params.PackPer+i < len(vals); i++ {
			u := big.NewInt(vals[k*params.PackPer+i] + params.ValOffset)
			acc.Add(acc, u.Lsh(u, uint(params.ValBits*i)))
		}
		out[k].SetBigInt(acc) // reduced mod p if negative / huge
	}
	return out
}

func leafRaw(salt fr.Element, h []int64, y int64) fr.Element {
	in := append([]fr.Element{salt}, packRaw(h)...)
	return commit.H(append(in, commit.FromInt(y))...)
}

func treeRaw(t testing.TB, ds *commit.Dataset) *commit.Tree {
	leaves := make([]fr.Element, params.N)
	for i := range leaves {
		leaves[i] = leafRaw(ds.Salt(i), ds.Features[i], ds.Labels[i])
	}
	tr, err := commit.BuildTree(leaves)
	if err != nil {
		t.Fatal(err)
	}
	return tr
}
