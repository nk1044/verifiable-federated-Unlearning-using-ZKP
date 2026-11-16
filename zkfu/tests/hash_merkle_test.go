package tests

import (
	"math/rand/v2"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/poseidon2"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/test"

	"zkfu/circuit"
	"zkfu/commit"
	"zkfu/params"
)

// Milestone 2a: our native H equals gnark-crypto's Poseidon2 Merkle-Damgard hasher, and equals
// the in-circuit H, for every input length the protocol uses.
func TestHashNativeMatchesLibraryAndCircuit(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	for _, n := range []int{1, 2, 3, 10, params.NumPacked(params.C*params.D + params.C)} {
		xs := make([]fr.Element, n)
		vs := make([]frontend.Variable, n)
		for i := range xs {
			xs[i].SetUint64(r.Uint64())
			if i%3 == 0 {
				xs[i].SetRandom()
			}
			vs[i] = xs[i]
		}
		got := commit.H(xs...)
		lib := poseidon2.NewHasher()
		lib.WriteElements(xs...)
		if want := lib.SumDigest(); !got.Equal(&want) {
			t.Fatalf("n=%d: native H differs from gnark-crypto Poseidon2 hasher", n)
		}
		shape := &circuit.HashCircuit{In: make([]frontend.Variable, n)}
		if err := test.IsSolved(shape, &circuit.HashCircuit{In: vs, Out: got}, ecc.BN254.ScalarField()); err != nil {
			t.Fatalf("n=%d: in-circuit H differs from native: %v", n, err)
		}
		var wrong fr.Element
		wrong.Add(&got, new(fr.Element).SetOne())
		if test.IsSolved(shape, &circuit.HashCircuit{In: vs, Out: wrong}, ecc.BN254.ScalarField()) == nil {
			t.Fatal("in-circuit H accepted a wrong digest")
		}
	}
}

// Milestone 2b: native commitment (leaf + Merkle path) agrees with the circuit; wrong index,
// wrong sibling, modified feature/label/salt are all rejected.
func TestMerkleNativeMatchesCircuit(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	ds := randDataset(t, r, 7, 1500, 1<<20)
	tree, err := ds.Tree()
	if err != nil {
		t.Fatal(err)
	}
	root := tree.Root()
	shape := circuit.NewLeafPath(1)

	assign := func(i, dirIdx int, mut func(a *circuit.LeafPathCircuit)) *circuit.LeafPathCircuit {
		a := circuit.NewLeafPath(1)
		a.Root = root
		a.Salt[0] = ds.Salt(i)
		for d := 0; d < params.D; d++ {
			a.H[0][d] = ds.Features[i][d]
		}
		a.Y[0] = ds.Labels[i]
		p := tree.Path(i)
		for l := 0; l < params.Depth; l++ {
			a.Path[0][l] = p[l]
			a.Dir[0][l] = (dirIdx >> l) & 1
		}
		if mut != nil {
			mut(a)
		}
		return a
	}
	for _, i := range []int{0, 1, 2, 1023, 1499, 1500, params.N - 1} { // incl. padding leaves
		leaf, _ := commit.Leaf(ds.Salt(i), ds.Features[i], ds.Labels[i])
		if got := commit.VerifyPath(leaf, i, tree.Path(i)); !got.Equal(&root) {
			t.Fatalf("native path for %d does not reach the root", i)
		}
		if err := test.IsSolved(shape, assign(i, i, nil), ecc.BN254.ScalarField()); err != nil {
			t.Fatalf("circuit rejects honest leaf %d: %v", i, err)
		}
	}
	bad := map[string]*circuit.LeafPathCircuit{
		"wrong index":      assign(5, 6, nil),
		"modified feature": assign(5, 5, func(a *circuit.LeafPathCircuit) { a.H[0][3] = ds.Features[5][3] + 1 }),
		"modified label":   assign(5, 5, func(a *circuit.LeafPathCircuit) { a.Y[0] = (ds.Labels[5] + 1) % params.C }),
		"modified salt":    assign(5, 5, func(a *circuit.LeafPathCircuit) { a.Salt[0] = ds.Salt(6) }),
		"wrong sibling":    assign(5, 5, func(a *circuit.LeafPathCircuit) { a.Path[0][4] = ds.Salt(9) }),
	}
	for name, a := range bad {
		if test.IsSolved(shape, a, ecc.BN254.ScalarField()) == nil {
			t.Fatalf("circuit accepted %s", name)
		}
	}
}
