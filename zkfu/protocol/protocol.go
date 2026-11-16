// Package protocol is the server/client logic outside the circuit: per-round seed, batch indices,
// mask bits, public inputs, witness assembly, and Groth16 key/proof I/O.
package protocol

import (
	"fmt"
	"math/big"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"

	"zkfu/circuit"
	"zkfu/commit"
	"zkfu/native"
	"zkfu/params"
)

// Seed = H(h_w, r, client_id). Derived from server-controlled values only.
func Seed(hw fr.Element, round, cid uint64) fr.Element {
	return commit.H(hw, commit.FromUint(round), commit.FromUint(cid))
}

// Indices: idx_j = low 11 bits of H(seed, j). Duplicates are allowed (sampling with replacement).
func Indices(seed fr.Element) [params.B]int {
	var idx [params.B]int
	for j := 0; j < params.B; j++ {
		e := commit.H(seed, commit.FromUint(uint64(j)))
		var bi big.Int
		e.BigInt(&bi)
		idx[j] = int(bi.Uint64() & (params.N - 1))
	}
	return idx
}

// MaskBits: 1 unless the position is padding (>= nReal) or forgotten. FL rounds pass forgotten=nil.
func MaskBits(idx [params.B]int, nReal int, forgotten map[int]bool) [params.B]uint8 {
	var m [params.B]uint8
	for j, i := range idx {
		if i < nReal && !forgotten[i] {
			m[j] = 1
		}
	}
	return m
}

func PackIdx(idx [params.B]int) [params.NIdxElems]*big.Int {
	var out [params.NIdxElems]*big.Int
	for k := range out {
		acc := new(big.Int)
		for jj := 0; jj < params.IdxPerElem; jj++ {
			v := big.NewInt(int64(idx[k*params.IdxPerElem+jj]))
			acc.Add(acc, v.Lsh(v, uint(jj*params.IdxBits)))
		}
		out[k] = acc
	}
	return out
}

func PackMask(m [params.B]uint8) *big.Int {
	acc := new(big.Int)
	for j := params.B - 1; j >= 0; j-- {
		acc.Lsh(acc, 1)
		acc.Or(acc, big.NewInt(int64(m[j])))
	}
	return acc
}

// Public is everything the verifier needs; the server computes it from its own state.
type Public struct {
	RootA, HW, HWNew fr.Element
	Idx              [params.NIdxElems]*big.Int
	Mask             *big.Int
	Round, ClientID  uint64
}

// RoundBatch is what both sides derive for (global head, round, client): indices and mask bits.
type RoundBatch struct {
	HW   fr.Element
	Idx  [params.B]int
	Mask [params.B]uint8
}

func DeriveBatch(W [][]int64, b []int64, round, cid uint64, nReal int, forgotten map[int]bool) (RoundBatch, error) {
	hw, err := commit.HeadHash(W, b)
	if err != nil {
		return RoundBatch{}, err
	}
	idx := Indices(Seed(hw, round, cid))
	return RoundBatch{HW: hw, Idx: idx, Mask: MaskBits(idx, nReal, forgotten)}, nil
}

// ServerPublic recomputes all public inputs on the server side from: the client's stored
// commitment, the head it sent (W, b), the head it received back (W2, b2), round, client id and
// the stored forget mask. Fails if the received head is out of range.
func ServerPublic(root fr.Element, nReal int, W [][]int64, b []int64, W2 [][]int64, b2 []int64,
	round, cid uint64, forgotten map[int]bool) (Public, error) {
	rb, err := DeriveBatch(W, b, round, cid, nReal, forgotten)
	if err != nil {
		return Public{}, err
	}
	hwNew, err := commit.HeadHash(W2, b2)
	if err != nil {
		return Public{}, fmt.Errorf("received head rejected: %w", err)
	}
	return Public{RootA: root, HW: rb.HW, HWNew: hwNew, Idx: PackIdx(rb.Idx), Mask: PackMask(rb.Mask),
		Round: round, ClientID: cid}, nil
}

// AssignPublic writes the public part of an assignment.
func (p Public) AssignPublic(a *circuit.ZkStepCircuit) {
	a.RootA, a.HW, a.HWNew = p.RootA, p.HW, p.HWNew
	for k := range p.Idx {
		a.IdxPacked[k] = p.Idx[k]
	}
	a.MaskPacked = p.Mask
	a.Round, a.ClientID = p.Round, p.ClientID
}

// ClientStep is what an honest client computes for one round: the batch, the native update and
// (lazily) the full witness. Deterministic in (dataset, W, b, round, mask).
type ClientStep struct {
	Batch  RoundBatch
	W, W2  [][]int64
	B, B2  []int64
	H      [][]int64
	Y      []int64
	Round  uint64
	DS     *commit.Dataset
	Forgot map[int]bool
}

func HonestStep(ds *commit.Dataset, W [][]int64, b []int64, round uint64, forgotten map[int]bool) (*ClientStep, error) {
	rb, err := DeriveBatch(W, b, round, uint64(ds.ClientID), ds.NReal, forgotten)
	if err != nil {
		return nil, err
	}
	H := make([][]int64, params.B)
	Y := make([]int64, params.B)
	for j, i := range rb.Idx {
		H[j], Y[j] = ds.Features[i], ds.Labels[i]
	}
	W2, b2, err := native.Step(W, b, H, Y, rb.Mask[:])
	if err != nil {
		return nil, err
	}
	return &ClientStep{Batch: rb, W: W, B: b, W2: W2, B2: b2, H: H, Y: Y, Round: round, DS: ds, Forgot: forgotten}, nil
}

// Assignment builds the full witness (private + public) for this step, using tree for paths.
func (s *ClientStep) Assignment(tree *commit.Tree) (*circuit.ZkStepCircuit, error) {
	hwNew, err := commit.HeadHash(s.W2, s.B2)
	if err != nil {
		return nil, err
	}
	pub := Public{RootA: tree.Root(), HW: s.Batch.HW, HWNew: hwNew, Idx: PackIdx(s.Batch.Idx),
		Mask: PackMask(s.Batch.Mask), Round: s.Round, ClientID: uint64(s.DS.ClientID)}
	a := &circuit.ZkStepCircuit{}
	pub.AssignPublic(a)
	FillPrivate(a, s.W, s.B, s.DS, tree, s.Batch.Idx)
	return a, nil
}

// FillPrivate sets the private witness: the head and, for each batch slot, the sample at idx.
func FillPrivate(a *circuit.ZkStepCircuit, W [][]int64, b []int64, ds *commit.Dataset, tree *commit.Tree, idx [params.B]int) {
	for c := 0; c < params.C; c++ {
		for d := 0; d < params.D; d++ {
			a.W[c][d] = W[c][d]
		}
		a.Bias[c] = b[c]
	}
	for j, i := range idx {
		a.Salt[j] = ds.Salt(i)
		for d := 0; d < params.D; d++ {
			a.H[j][d] = ds.Features[i][d]
		}
		a.Y[j] = ds.Labels[i]
		p := tree.Path(i)
		for l := 0; l < params.Depth; l++ {
			a.Path[j][l] = p[l]
		}
	}
}
