package circuit

import (
	"github.com/consensys/gnark/frontend"

	"zkfu/params"
)

// Small component circuits, compiled only to report a per-component constraint breakdown.

// LeafPathCircuit: M samples, each = range-check D features + label one-hot + leaf + Merkle path.
type LeafPathCircuit struct {
	Root frontend.Variable `gnark:",public"`
	Dir  [][params.Depth]frontend.Variable
	Salt []frontend.Variable
	H    [][params.D]frontend.Variable
	Y    []frontend.Variable
	Path [][params.Depth]frontend.Variable
}

func NewLeafPath(m int) *LeafPathCircuit {
	return &LeafPathCircuit{Dir: make([][params.Depth]frontend.Variable, m), Salt: make([]frontend.Variable, m),
		H: make([][params.D]frontend.Variable, m), Y: make([]frontend.Variable, m), Path: make([][params.Depth]frontend.Variable, m)}
}

func (c *LeafPathCircuit) Define(api frontend.API) error {
	g, err := NewGadgets(api)
	if err != nil {
		return err
	}
	for s := range c.Salt {
		for d := 0; d < params.D; d++ {
			g.CheckValue(c.H[s][d])
		}
		var sum frontend.Variable = 0
		for cc := 0; cc < params.C; cc++ {
			sum = api.Add(sum, api.IsZero(api.Sub(c.Y[s], cc)))
		}
		api.AssertIsEqual(sum, 1)
		for l := 0; l < params.Depth; l++ {
			api.AssertIsBoolean(c.Dir[s][l])
		}
		leaf := g.Leaf(c.Salt[s], c.H[s][:], c.Y[s])
		api.AssertIsEqual(g.MerkleRoot(leaf, c.Dir[s][:], c.Path[s][:]), c.Root)
	}
	return nil
}

// HeadHashCircuit: M heads, each range-checked (C*D + C values) and hashed.
type HeadHashCircuit struct {
	Out  []frontend.Variable `gnark:",public"`
	W    [][params.C][params.D]frontend.Variable
	Bias [][params.C]frontend.Variable
}

func NewHeadHash(m int) *HeadHashCircuit {
	return &HeadHashCircuit{Out: make([]frontend.Variable, m), W: make([][params.C][params.D]frontend.Variable, m),
		Bias: make([][params.C]frontend.Variable, m)}
}

func (c *HeadHashCircuit) Define(api frontend.API) error {
	g, err := NewGadgets(api)
	if err != nil {
		return err
	}
	for s := range c.Out {
		for cc := 0; cc < params.C; cc++ {
			g.CheckValue(c.Bias[s][cc])
			for d := 0; d < params.D; d++ {
				g.CheckValue(c.W[s][cc][d])
			}
		}
		api.AssertIsEqual(g.HeadHash(c.W[s], c.Bias[s]), c.Out[s])
	}
	return nil
}

// TruncCircuit applies the truncation gadget to a batch of inputs and checks the outputs.
// Used both for the breakdown and by the truncation equivalence tests.
type TruncCircuit struct {
	X   []frontend.Variable
	Out []frontend.Variable `gnark:",public"`
	T   int                 `gnark:"-"`
	N   int                 `gnark:"-"`
}

func (c *TruncCircuit) Define(api frontend.API) error {
	g, err := NewGadgets(api)
	if err != nil {
		return err
	}
	for i := range c.X {
		api.AssertIsEqual(g.Trunc(c.X[i], c.T, c.N), c.Out[i])
	}
	return nil
}

// HashCircuit computes H over a fixed number of inputs (hash agreement tests).
type HashCircuit struct {
	In  []frontend.Variable
	Out frontend.Variable `gnark:",public"`
}

func (c *HashCircuit) Define(api frontend.API) error {
	g, err := NewGadgets(api)
	if err != nil {
		return err
	}
	api.AssertIsEqual(g.H(c.In...), c.Out)
	return nil
}
