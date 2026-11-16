package circuit

import (
	"github.com/consensys/gnark/frontend"

	"zkfu/params"
)

// ZkStepCircuit proves: "W', b' (hashed to HWNew) is exactly one fixed-point SGD step of the head
// whose hash is HW, on the B samples at the public indices of the dataset committed in RootA,
// with the public per-sample mask bits applied."
type ZkStepCircuit struct {
	// ---- public inputs ----
	RootA      frontend.Variable                   `gnark:",public"` // Com(A)
	HW         frontend.Variable                   `gnark:",public"` // H(W, b) sent by the server
	HWNew      frontend.Variable                   `gnark:",public"` // H(W', b') sent back by the client
	IdxPacked  [params.NIdxElems]frontend.Variable `gnark:",public"` // B indices, 11 bits each
	MaskPacked frontend.Variable                   `gnark:",public"` // B mask bits
	Round      frontend.Variable                   `gnark:",public"` // replay binding only
	ClientID   frontend.Variable                   `gnark:",public"` // replay binding only

	// ---- private witness ----
	W    [params.C][params.D]frontend.Variable
	Bias [params.C]frontend.Variable
	Salt [params.B]frontend.Variable
	H    [params.B][params.D]frontend.Variable
	Y    [params.B]frontend.Variable
	Path [params.B][params.Depth]frontend.Variable
}

func (c *ZkStepCircuit) Define(api frontend.API) error {
	g, err := NewGadgets(api)
	if err != nil {
		return err
	}

	// Bind Round and ClientID into a real constraint so their Groth16 verification-key terms are
	// non-trivial (a public input used by no constraint would be malleable).
	api.Mul(api.Add(c.Round, 1), api.Add(c.ClientID, 1))

	// (1) unpack indices (direction bits) and mask bits; ToBinary constrains booleanity + recomposition.
	var dir [params.B][]frontend.Variable
	for k := 0; k < params.NIdxElems; k++ {
		bits := api.ToBinary(c.IdxPacked[k], params.IdxPerElem*params.IdxBits)
		for jj := 0; jj < params.IdxPerElem; jj++ {
			dir[k*params.IdxPerElem+jj] = bits[jj*params.IdxBits : (jj+1)*params.IdxBits]
		}
	}
	mask := api.ToBinary(c.MaskPacked, params.B)

	// (5) range-check the head, (2) bind it to HW.
	for cc := 0; cc < params.C; cc++ {
		g.CheckValue(c.Bias[cc])
		for d := 0; d < params.D; d++ {
			g.CheckValue(c.W[cc][d])
		}
	}
	api.AssertIsEqual(g.HeadHash(c.W, c.Bias), c.HW)

	// (3)+(4)+(5) per sample: range-check features, label one-hot, leaf, Merkle path to RootA.
	var onehot [params.B][params.C]frontend.Variable
	for j := 0; j < params.B; j++ {
		for d := 0; d < params.D; d++ {
			g.CheckValue(c.H[j][d])
		}
		var sum frontend.Variable = 0
		for cc := 0; cc < params.C; cc++ {
			onehot[j][cc] = api.IsZero(api.Sub(c.Y[j], cc))
			sum = api.Add(sum, onehot[j][cc])
		}
		api.AssertIsEqual(sum, 1) // y in [0, C-1]

		leaf := g.Leaf(c.Salt[j], c.H[j][:], c.Y[j])
		api.AssertIsEqual(g.MerkleRoot(leaf, dir[j], c.Path[j][:]), c.RootA)
	}

	// (6) forward, masked error, gradients, update — the SAME H[j], Y[j] that were Merkle-checked.
	var me [params.B][params.C]frontend.Variable
	for j := 0; j < params.B; j++ {
		for cc := 0; cc < params.C; cc++ {
			var acc frontend.Variable = 0
			for d := 0; d < params.D; d++ {
				acc = api.Add(acc, api.Mul(c.W[cc][d], c.H[j][d]))
			}
			z := api.Add(g.Trunc(acc, params.Frac, params.NFwd), c.Bias[cc])
			e := api.Sub(z, api.Mul(onehot[j][cc], params.One))
			me[j][cc] = api.Mul(mask[j], e)
		}
	}
	var W2 [params.C][params.D]frontend.Variable
	var b2 [params.C]frontend.Variable
	for cc := 0; cc < params.C; cc++ {
		for d := 0; d < params.D; d++ {
			var gw frontend.Variable = 0
			for j := 0; j < params.B; j++ {
				gw = api.Add(gw, api.Mul(me[j][cc], c.H[j][d]))
			}
			W2[cc][d] = api.Sub(c.W[cc][d], g.Trunc(gw, params.Frac+params.S, params.NGW))
			g.CheckValue(W2[cc][d]) // needed for injectivity of the packed head hash
		}
		var gb frontend.Variable = 0
		for j := 0; j < params.B; j++ {
			gb = api.Add(gb, me[j][cc])
		}
		b2[cc] = api.Sub(c.Bias[cc], g.Trunc(gb, params.S, params.NGb))
		g.CheckValue(b2[cc])
	}

	// (7) bind the result to HWNew.
	api.AssertIsEqual(g.HeadHash(W2, b2), c.HWNew)
	return nil
}
