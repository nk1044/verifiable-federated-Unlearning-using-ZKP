// Package commit implements the native hash H, value packing, leaf hashing, the head hash and
// the binary Merkle tree. The in-circuit counterparts live in package circuit and are tested to
// agree bit-for-bit (tests/hash_merkle_test.go).
package commit

import (
	"fmt"
	"math/big"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/poseidon2"

	"zkfu/params"
)

// comp is the Poseidon2 (width 2, default BN254 parameters) compression function. It is the
// same permutation gnark's std/hash/poseidon2 uses in-circuit.
var comp = poseidon2.NewCompressor(poseidon2.NewDefaultPermutation())

// H is Merkle-Damgard over Poseidon2 with zero IV: s = 0; for x in xs: s = Compress(s, x).
// Identical to poseidon2.NewHasher() natively and std/hash/poseidon2.New(api) in-circuit.
func H(xs ...fr.Element) fr.Element {
	var s fr.Element
	for i := range xs {
		comp.CompressField(&s, &s, &xs[i])
	}
	return s
}

// FromInt maps a signed integer to the field (negative v -> p - |v|).
func FromInt(v int64) fr.Element {
	var e fr.Element
	e.SetInt64(v)
	return e
}

// FromUint maps an unsigned integer to the field.
func FromUint(v uint64) fr.Element {
	var e fr.Element
	e.SetUint64(v)
	return e
}

// Pack encodes range-checked signed values as field elements: u = v + 2^24 in [0, 2^25),
// PackPer values per element, little-endian in 25-bit limbs. Injective on in-range inputs.
func Pack(vals []int64) ([]fr.Element, error) {
	out := make([]fr.Element, params.NumPacked(len(vals)))
	for k := range out {
		acc := new(big.Int)
		for i := 0; i < params.PackPer; i++ {
			idx := k*params.PackPer + i
			if idx >= len(vals) {
				break
			}
			if !params.InRange(vals[idx]) {
				return nil, fmt.Errorf("pack: value %d out of range at %d", vals[idx], idx)
			}
			u := big.NewInt(vals[idx] + params.ValOffset)
			acc.Add(acc, u.Lsh(u, uint(params.ValBits*i)))
		}
		out[k].SetBigInt(acc)
	}
	return out, nil
}

// Leaf = H(salt, Pack(h[0..D-1])..., y).
func Leaf(salt fr.Element, h []int64, y int64) (fr.Element, error) {
	p, err := Pack(h)
	if err != nil {
		return fr.Element{}, err
	}
	in := append([]fr.Element{salt}, p...)
	in = append(in, FromInt(y))
	return H(in...), nil
}

// FlattenHead returns W row-major followed by b (C*D + C values).
func FlattenHead(W [][]int64, b []int64) []int64 {
	out := make([]int64, 0, params.C*params.D+params.C)
	for c := 0; c < params.C; c++ {
		out = append(out, W[c]...)
	}
	return append(out, b...)
}

// HeadHash = H(Pack(W row-major, b)). Fails if any value is out of range.
func HeadHash(W [][]int64, b []int64) (fr.Element, error) {
	if len(W) != params.C || len(b) != params.C {
		return fr.Element{}, fmt.Errorf("head: wrong shape")
	}
	for c := range W {
		if len(W[c]) != params.D {
			return fr.Element{}, fmt.Errorf("head: wrong row length")
		}
	}
	p, err := Pack(FlattenHead(W, b))
	if err != nil {
		return fr.Element{}, err
	}
	return H(p...), nil
}
