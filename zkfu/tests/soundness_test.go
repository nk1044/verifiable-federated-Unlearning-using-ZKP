package tests

import (
	"math/big"
	"math/rand/v2"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"

	"zkfu/circuit"
	"zkfu/commit"
	"zkfu/native"
	"zkfu/params"
	"zkfu/protocol"
)

// fixture: two committed clients and a realistic-scale global head.
type fixture struct {
	dsA, dsB     *commit.Dataset
	treeA, treeB *commit.Tree
	W, b         [][]int64
	bb           []int64
}

func newFixture(t *testing.T) *fixture {
	r := rand.New(rand.NewPCG(11, 12))
	f := &fixture{}
	f.dsA = randDataset(t, r, 3, 1200, 1<<16)
	f.dsB = randDataset(t, r, 4, 1000, 1<<16)
	var err error
	if f.treeA, err = f.dsA.Tree(); err != nil {
		t.Fatal(err)
	}
	if f.treeB, err = f.dsB.Tree(); err != nil {
		t.Fatal(err)
	}
	f.W, f.bb = randHead(r, 1<<15)
	return f
}

// prove honestly and return the proof plus the server-side public inputs for the received head.
func proveHonest(t *testing.T, ds *commit.Dataset, tree *commit.Tree, W [][]int64, b []int64, round uint64, forgot map[int]bool) ([]byte, *protocol.ClientStep) {
	pk, _ := getKeys(t)
	st, err := protocol.HonestStep(ds, W, b, round, forgot)
	if err != nil {
		t.Fatal(err)
	}
	a, err := st.Assignment(tree)
	if err != nil {
		t.Fatal(err)
	}
	proof, err := protocol.Prove(getCCS(t), pk, a)
	if err != nil {
		t.Fatalf("honest proving failed: %v", err)
	}
	return proof, st
}

func serverPub(t *testing.T, ds *commit.Dataset, root fr.Element, W [][]int64, b []int64, W2 [][]int64, b2 []int64, round uint64, forgot map[int]bool) protocol.Public {
	p, err := protocol.ServerPublic(root, ds.NReal, W, b, W2, b2, round, uint64(ds.ClientID), forgot)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func mustVerify(t *testing.T, proof []byte, p protocol.Public, name string) {
	_, vk := getKeys(t)
	if err := protocol.Verify(vk, proof, p); err != nil {
		t.Fatalf("%s: honest proof rejected: %v", name, err)
	}
}

func mustReject(t *testing.T, proof []byte, p protocol.Public, name string) {
	_, vk := getKeys(t)
	if protocol.Verify(vk, proof, p) == nil {
		t.Fatalf("SOUNDNESS: %s was accepted", name)
	}
}

func mustNotSolve(t *testing.T, a *circuit.ZkStepCircuit, name string) {
	if solve(t, a) == nil {
		t.Fatalf("SOUNDNESS: %s has a satisfying witness", name)
	}
}

func TestCompletenessAndSoundness(t *testing.T) {
	f := newFixture(t)
	rootA, rootB := f.treeA.Root(), f.treeB.Root()
	const round = 5

	// ---------------- completeness: honest FL step and honest FU step verify ----------------
	proofFL, stFL := proveHonest(t, f.dsA, f.treeA, f.W, f.bb, round, nil)
	pubFL := serverPub(t, f.dsA, rootA, f.W, f.bb, stFL.W2, stFL.B2, round, nil)
	mustVerify(t, proofFL, pubFL, "FL step")

	// FU: forget a few real positions, guaranteed to include the first sampled index
	slot := -1 // first batch slot holding a real sample; forgetting it guarantees a 0 mask bit
	for j, i := range stFL.Batch.Idx {
		if i < f.dsA.NReal {
			slot = j
			break
		}
	}
	if slot < 0 {
		t.Fatal("batch has no real position")
	}
	forgot := map[int]bool{stFL.Batch.Idx[slot]: true, 17: true, 400: true}
	proofFU, stFU := proveHonest(t, f.dsA, f.treeA, f.W, f.bb, round, forgot)
	pubFU := serverPub(t, f.dsA, rootA, f.W, f.bb, stFU.W2, stFU.B2, round, forgot)
	mustVerify(t, proofFU, pubFU, "FU step")
	if stFU.Batch.Mask[slot] != 0 {
		t.Fatal("forgotten sample not masked")
	}

	// ---------------- soundness ----------------
	t.Run("backdoor patch: one weight +1", func(t *testing.T) {
		W2, b2 := cloneHead(stFU.W2, stFU.B2)
		W2[3][7]++
		mustReject(t, proofFU, serverPub(t, f.dsA, rootA, f.W, f.bb, W2, b2, round, forgot), "patched head with honest proof")
		// the client cannot produce a proof for the patched head either
		a, _ := stFU.Assignment(f.treeA)
		hw, _ := commit.HeadHash(W2, b2)
		a.HWNew = hw
		mustNotSolve(t, a, "patched head")
	})

	t.Run("wrong starting model", func(t *testing.T) {
		Wo, bo := cloneHead(f.W, f.bb)
		Wo[0][0] += 1000
		// client trains from Wo but the server's h_w is H(W): no witness ...
		st, err := protocol.HonestStep(f.dsA, Wo, bo, round, forgot)
		if err != nil {
			t.Fatal(err)
		}
		a, _ := st.Assignment(f.treeA)
		hw, _ := commit.HeadHash(f.W, f.bb)
		a.HW = hw
		mustNotSolve(t, a, "training from a different model")
		// ... and an honest proof made from Wo does not verify against the server's view
		proofO, stO := proveHonest(t, f.dsA, f.treeA, Wo, bo, round, forgot)
		mustReject(t, proofO, serverPub(t, f.dsA, rootA, f.W, f.bb, stO.W2, stO.B2, round, forgot), "proof from wrong model")
	})

	base := func() *circuit.ZkStepCircuit { a, _ := stFU.Assignment(f.treeA); return a }
	t.Run("sample not in committed dataset", func(t *testing.T) {
		a := base() // fresh sample
		var s fr.Element
		s.SetRandom()
		a.Salt[1] = s
		for d := 0; d < params.D; d++ {
			a.H[1][d] = int64(d * 100)
		}
		mustNotSolve(t, a, "fresh sample")
		a = base() // modified feature
		a.H[2][5] = f.dsA.Features[stFU.Batch.Idx[2]][5] + 1
		mustNotSolve(t, a, "modified feature")
		a = base() // modified label
		a.Y[2] = (f.dsA.Labels[stFU.Batch.Idx[2]] + 1) % params.C
		mustNotSolve(t, a, "modified label")
	})

	t.Run("correct sample at wrong index", func(t *testing.T) {
		a := base()
		other := (stFU.Batch.Idx[4] + 1) % f.dsA.NReal // a genuine committed sample, other position
		a.Salt[4] = f.dsA.Salt(other)
		for d := 0; d < params.D; d++ {
			a.H[4][d] = f.dsA.Features[other][d]
		}
		a.Y[4] = f.dsA.Labels[other]
		p := f.treeA.Path(other)
		for l := 0; l < params.Depth; l++ {
			a.Path[4][l] = p[l]
		}
		mustNotSolve(t, a, "sample moved to another index")
	})

	t.Run("mask bit flipped 0->1 for a forgotten sample", func(t *testing.T) {
		// client proves as if it had not forgotten anything (mask bit 1 at the forgotten slot) ...
		proofX, stX := proveHonest(t, f.dsA, f.treeA, f.W, f.bb, round, nil)
		// ... the server checks it against its stored mask, where that bit is 0
		mustReject(t, proofX, serverPub(t, f.dsA, rootA, f.W, f.bb, stX.W2, stX.B2, round, forgot), "flipped mask")
		// and a witness with the server's mask but the unmasked update does not exist
		a := base()
		hw, _ := commit.HeadHash(stX.W2, stX.B2)
		a.HWNew = hw
		mustNotSolve(t, a, "unmasked update under the server mask")
	})

	t.Run("public inputs swapped between clients or rounds", func(t *testing.T) {
		proofB, stB := proveHonest(t, f.dsB, f.treeB, f.W, f.bb, round, nil)
		pubB := serverPub(t, f.dsB, rootB, f.W, f.bb, stB.W2, stB.B2, round, nil)
		mustVerify(t, proofB, pubB, "client B")
		mustReject(t, proofFL, pubB, "A's proof under B's public inputs")
		mustReject(t, proofB, pubFL, "B's proof under A's public inputs")
		proofR6, stR6 := proveHonest(t, f.dsA, f.treeA, f.W, f.bb, round+1, nil)
		mustVerify(t, proofR6, serverPub(t, f.dsA, rootA, f.W, f.bb, stR6.W2, stR6.B2, round+1, nil), "round 6")
		mustReject(t, proofR6, pubFL, "round-6 proof under round-5 public inputs")
		// only the binding fields changed (everything else identical): still rejected
		p := pubFL
		p.Round = round + 1
		mustReject(t, proofFL, p, "Round changed")
		p = pubFL
		p.ClientID = uint64(f.dsB.ClientID)
		mustReject(t, proofFL, p, "ClientID changed")
	})

	t.Run("out-of-range witness values", func(t *testing.T) {
		// A malicious client commits a sample containing an out-of-range feature (using an unchecked
		// encoder), places it in a MASKED slot so that W', b' are unaffected, and builds an otherwise
		// fully consistent witness. Only the range check can reject it. Control: same construction
		// with an in-range value is accepted, so the rejection is due to the range check.
		r := rand.New(rand.NewPCG(21, 22))
		for _, tc := range []struct {
			v      int64
			accept bool
		}{{params.ValOffset - 1, true}, {-params.ValOffset, true}, {params.ValOffset, false}, {-params.ValOffset - 1, false}, {1 << 40, false}} {
			ds := randDataset(t, r, 9, 1200, 1<<16)
			Wh, bh := randHead(r, 1<<15)
			rb, _ := protocol.DeriveBatch(Wh, bh, 1, 9, ds.NReal, nil)
			target := -1
			for _, i := range rb.Idx { // first batch slot that is a real (non-padding) position
				if i < ds.NReal {
					target = i
					break
				}
			}
			if target < 0 {
				t.Fatal("batch has no real position")
			}
			forg := map[int]bool{target: true} // masked slot
			ds.Features[target][0] = tc.v
			tree := treeRaw(t, ds)
			// native update with the masked bad feature zeroed (it contributes nothing in-circuit either)
			clean := *ds
			clean.Features = append([][]int64(nil), ds.Features...)
			clean.Features[target] = make([]int64, params.D)
			st, err := protocol.HonestStep(&clean, Wh, bh, 1, forg)
			if err != nil {
				t.Fatal(err)
			}
			st.DS = ds
			a, err := st.Assignment(tree)
			if err != nil {
				t.Fatal(err)
			}
			for j, i := range st.Batch.Idx {
				if i == target {
					a.H[j][0] = tc.v
				}
			}
			ok := solve(t, a) == nil
			if ok != tc.accept {
				t.Fatalf("feature value %d: accepted=%v, want %v", tc.v, ok, tc.accept)
			}
		}
		// out-of-range weight with a consistent (unchecked) head hash
		a := base()
		a.W[0][0] = int64(params.ValOffset)
		Wbad, _ := cloneHead(f.W, f.bb)
		Wbad[0][0] = params.ValOffset
		a.HW = commit.H(packRaw(commit.FlattenHead(Wbad, f.bb))...)
		mustNotSolve(t, a, "out-of-range weight")
		// field wraparound: a "negative" feature encoded as p - 2^40
		a = base()
		var wrap fr.Element
		wrap.SetBigInt(new(big.Int).Neg(new(big.Int).Lsh(big.NewInt(1), 40)))
		a.H[3][0] = wrap
		mustNotSolve(t, a, "wrapped feature")
		// label outside [0, C-1]
		a = base()
		a.Y[0] = params.C
		mustNotSolve(t, a, "label = C")
	})

	_ = native.ErrOverflow
}
