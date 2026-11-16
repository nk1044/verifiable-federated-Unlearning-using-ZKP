package tests

import (
	"math/big"
	mrand "math/rand"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/test"

	"zkfu/circuit"
	"zkfu/native"
	"zkfu/params"
)

// Milestone 1: the truncation gadget agrees with native truncation (and with floor division)
// on random, zero, negative and boundary inputs, for all three widths used by the circuit.
func TestTruncGadgetMatchesNative(t *testing.T) {
	r := mrand.New(mrand.NewSource(12345))
	for _, tc := range []struct{ t, n int }{{params.Frac, params.NFwd}, {params.Frac + params.S, params.NGW}, {params.S, params.NGb}} {
		lim := new(big.Int).Lsh(big.NewInt(1), uint(tc.n-1)) // valid |x| < 2^(n-1)  => x in [-lim, lim)
		p2t := new(big.Int).Lsh(big.NewInt(1), uint(tc.t))
		xs := []*big.Int{
			big.NewInt(0), big.NewInt(1), big.NewInt(-1),
			new(big.Int).Neg(lim), new(big.Int).Sub(lim, big.NewInt(1)), // extreme boundaries
			new(big.Int).Set(p2t), new(big.Int).Neg(p2t),
			new(big.Int).Sub(p2t, big.NewInt(1)), new(big.Int).Add(new(big.Int).Neg(p2t), big.NewInt(1)),
			new(big.Int).Add(p2t, big.NewInt(1)), new(big.Int).Sub(new(big.Int).Neg(p2t), big.NewInt(1)),
		}
		for i := 0; i < 200; i++ { // random over the full width, both signs
			x := new(big.Int).Rand(r, lim)
			if i%2 == 1 {
				x.Neg(x)
			}
			xs = append(xs, x)
		}
		outs := make([]frontend.Variable, len(xs))
		ins := make([]frontend.Variable, len(xs))
		for i, x := range xs {
			got, err := native.Trunc(x, uint(tc.t), uint(tc.n))
			if err != nil {
				t.Fatalf("native overflow on valid x=%s", x)
			}
			// reference: floor division
			want := new(big.Int).Div(x, p2t) // Go's Div is Euclidean: floor for positive divisor
			if got.Cmp(want) != 0 {
				t.Fatalf("native trunc(%s,%d)=%s want %s", x, tc.t, got, want)
			}
			ins[i], outs[i] = x, got
		}
		shape := &circuit.TruncCircuit{X: make([]frontend.Variable, len(xs)), Out: make([]frontend.Variable, len(xs)), T: tc.t, N: tc.n}
		ok := &circuit.TruncCircuit{X: ins, Out: outs, T: tc.t, N: tc.n}
		if err := test.IsSolved(shape, ok, ecc.BN254.ScalarField()); err != nil {
			t.Fatalf("gadget disagrees with native (t=%d n=%d): %v", tc.t, tc.n, err)
		}
		// off-by-one outputs must be rejected
		bad := append([]frontend.Variable(nil), outs...)
		bad[3] = new(big.Int).Add(outs[3].(*big.Int), big.NewInt(1))
		if test.IsSolved(shape, &circuit.TruncCircuit{X: ins, Out: bad, T: tc.t, N: tc.n}, ecc.BN254.ScalarField()) == nil {
			t.Fatal("gadget accepted a wrong truncation")
		}
		// out-of-width input: native reports overflow, gadget has no satisfying witness
		for _, x := range []*big.Int{new(big.Int).Set(lim), new(big.Int).Sub(new(big.Int).Neg(lim), big.NewInt(1))} {
			if _, err := native.Trunc(x, uint(tc.t), uint(tc.n)); err == nil {
				t.Fatalf("native accepted out-of-width x=%s", x)
			}
			one := &circuit.TruncCircuit{X: make([]frontend.Variable, 1), Out: make([]frontend.Variable, 1), T: tc.t, N: tc.n}
			q := new(big.Int).Div(x, p2t)
			if test.IsSolved(one, &circuit.TruncCircuit{X: []frontend.Variable{x}, Out: []frontend.Variable{q}, T: tc.t, N: tc.n}, ecc.BN254.ScalarField()) == nil {
				t.Fatalf("gadget accepted out-of-width x=%s", x)
			}
		}
	}
}
