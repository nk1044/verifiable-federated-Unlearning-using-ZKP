// Package circuit holds the gnark circuit ZkStepCircuit and its fixed-point / hashing gadgets.
package circuit

import (
	"fmt"
	"math/big"

	"github.com/consensys/gnark/constraint/solver"
	"github.com/consensys/gnark/frontend"
	p2 "github.com/consensys/gnark/std/permutation/poseidon2"
	"github.com/consensys/gnark/std/rangecheck"

	"zkfu/params"
)

func init() { solver.RegisterHint(TruncHint) }

func pow2(n int) *big.Int { return new(big.Int).Lsh(big.NewInt(1), uint(n)) }

// TruncHint: inputs (x', t) with x' the canonical integer of the field element;
// outputs q = x' >> t and r = x' mod 2^t. Both outputs are fully constrained by Trunc below.
func TruncHint(_ *big.Int, in []*big.Int, out []*big.Int) error {
	if len(in) != 2 || len(out) != 2 {
		return fmt.Errorf("TruncHint: bad arity")
	}
	t := uint(in[1].Uint64())
	out[0].Rsh(in[0], t)
	out[1].And(in[0], new(big.Int).Sub(pow2(int(t)), big.NewInt(1)))
	return nil
}

// Gadgets bundles the API, the range checker and the Poseidon2 permutation.
type Gadgets struct {
	api  frontend.API
	rc   frontend.Rangechecker
	perm *p2.Permutation
}

func NewGadgets(api frontend.API) (*Gadgets, error) {
	perm, err := p2.NewPoseidon2(api)
	if err != nil {
		return nil, err
	}
	return &Gadgets{api: api, rc: rangecheck.New(api), perm: perm}, nil
}

// Trunc returns floor(x / 2^t) for signed x with |x| < 2^(n-1).
// Constraints: x' = x + 2^(n-1);  x' == q*2^t + r;  0 <= r < 2^t;  0 <= q < 2^(n-t).
// Because q*2^t + r < 2^n << p, the decomposition is unique, so the hint is fully constrained;
// if x were out of width, no (q, r) would satisfy the equation and proving fails.
func (g *Gadgets) Trunc(x frontend.Variable, t, n int) frontend.Variable {
	xp := g.api.Add(x, pow2(n-1))
	res, err := g.api.Compiler().NewHint(TruncHint, 2, xp, t)
	if err != nil {
		panic(err)
	}
	q, r := res[0], res[1]
	g.rc.Check(r, t)
	g.rc.Check(q, n-t)
	g.api.AssertIsEqual(xp, g.api.Add(g.api.Mul(q, pow2(t)), r))
	return g.api.Sub(q, pow2(n-1-t))
}

// CheckValue asserts v in [-2^24, 2^24), i.e. v + 2^24 in [0, 2^25).
func (g *Gadgets) CheckValue(v frontend.Variable) {
	g.rc.Check(g.api.Add(v, params.ValOffset), params.ValBits)
}

// Pack mirrors commit.Pack. Callers MUST have range-checked every value with CheckValue,
// otherwise the 25-bit limbs could overlap and the encoding would not be injective.
func (g *Gadgets) Pack(vals []frontend.Variable) []frontend.Variable {
	out := make([]frontend.Variable, params.NumPacked(len(vals)))
	for k := range out {
		var acc frontend.Variable = 0
		for i := 0; i < params.PackPer; i++ {
			idx := k*params.PackPer + i
			if idx >= len(vals) {
				break
			}
			u := g.api.Add(vals[idx], params.ValOffset)
			acc = g.api.Add(acc, g.api.Mul(u, pow2(params.ValBits*i)))
		}
		out[k] = acc
	}
	return out
}

// H mirrors commit.H: Merkle-Damgard over the Poseidon2 compression with zero IV.
func (g *Gadgets) H(xs ...frontend.Variable) frontend.Variable {
	var s frontend.Variable = 0
	for _, x := range xs {
		s = g.perm.Compress(s, x)
	}
	return s
}

// Leaf mirrors commit.Leaf (h must already be range-checked).
func (g *Gadgets) Leaf(salt frontend.Variable, h []frontend.Variable, y frontend.Variable) frontend.Variable {
	in := append([]frontend.Variable{salt}, g.Pack(h)...)
	return g.H(append(in, y)...)
}

// MerkleRoot recomputes the root from a leaf; dirBits[l] = 1 means the running node is the
// right child at level l. The direction bits are the bits of the public index.
func (g *Gadgets) MerkleRoot(leaf frontend.Variable, dirBits []frontend.Variable, path []frontend.Variable) frontend.Variable {
	cur := leaf
	for l := range path {
		left := g.api.Select(dirBits[l], path[l], cur)
		right := g.api.Select(dirBits[l], cur, path[l])
		cur = g.H(left, right)
	}
	return cur
}

// HeadHash mirrors commit.HeadHash (values must already be range-checked).
func (g *Gadgets) HeadHash(W [params.C][params.D]frontend.Variable, b [params.C]frontend.Variable) frontend.Variable {
	flat := make([]frontend.Variable, 0, params.C*params.D+params.C)
	for c := 0; c < params.C; c++ {
		flat = append(flat, W[c][:]...)
	}
	flat = append(flat, b[:]...)
	return g.H(g.Pack(flat)...)
}
