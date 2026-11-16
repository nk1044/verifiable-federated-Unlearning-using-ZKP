// cmd/hashcost: marginal R1CS cost of one 2-to-1 hash for the candidate hash functions, and the
// projected size of ZkStepCircuit if each were used instead of Poseidon2 (justifies the choice).
//
//	go run ./cmd/hashcost -json results/hashcost.json
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/logger"
	"github.com/consensys/gnark/std/hash/mimc"
	"github.com/consensys/gnark/std/hash/sha2"
	"github.com/consensys/gnark/std/math/uints"

	"zkfu/circuit"
	"zkfu/protocol"
)

// MiMCCircuit hashes M pairs of field elements with MiMC (Merkle-Damgard, 2 absorbs each).
type MiMCCircuit struct{ In [][2]frontend.Variable }

func (c *MiMCCircuit) Define(api frontend.API) error {
	for i := range c.In {
		h, err := mimc.New(api)
		if err != nil {
			return err
		}
		h.Write(c.In[i][0], c.In[i][1])
		api.AssertIsDifferent(h.Sum(), 0)
	}
	return nil
}

// SHACircuit hashes M 64-byte inputs (two 32-byte children) with SHA-256.
type SHACircuit struct{ In [][64]uints.U8 }

func (c *SHACircuit) Define(api frontend.API) error {
	for i := range c.In {
		h, err := sha2.New(api)
		if err != nil {
			return err
		}
		h.Write(c.In[i][:])
		d := h.Sum()
		api.AssertIsDifferent(d[0].Val, 256)
	}
	return nil
}

func count(c frontend.Circuit) int {
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, c)
	if err != nil {
		log.Fatal(err)
	}
	return cs.GetNbConstraints()
}

func main() {
	out := flag.String("json", "", "write JSON here")
	flag.Parse()
	logger.Disable()
	marg := func(mk func(int) frontend.Circuit) int { return count(mk(4)) - count(mk(3)) }
	p2 := marg(func(m int) frontend.Circuit { return &circuit.HashCircuit{In: make([]frontend.Variable, m)} }) * 2
	mi := marg(func(m int) frontend.Circuit { return &MiMCCircuit{In: make([][2]frontend.Variable, m)} })
	sh := marg(func(m int) frontend.Circuit { return &SHACircuit{In: make([][64]uints.U8, m)} })
	ccs, err := protocol.Compile()
	if err != nil {
		log.Fatal(err)
	}
	total := ccs.GetNbConstraints()
	// ZkStepCircuit performs 1170 compressions: 32 leaves x 10 absorbs + 32 x 11 nodes x 2 + 2 heads x 73.
	const absorbs = 32*10 + 32*11*2 + 2*73
	perAbsorbP2 := p2 / 2
	res := map[string]any{
		"two_to_one_hash": map[string]int{"poseidon2": p2, "mimc": mi, "sha256_64B": sh},
		"absorbs_in_circuit": absorbs, "circuit_total_poseidon2": total,
		"projected_total": map[string]int{
			"poseidon2": total,
			"mimc":      total - absorbs*perAbsorbP2 + absorbs*mi/2,
			"sha256":    total - absorbs*perAbsorbP2 + absorbs*sh/2, // one 64-byte block per 2 absorbs (optimistic)
		},
	}
	b, _ := json.MarshalIndent(res, "", "  ")
	fmt.Println(string(b))
	if *out != "" {
		if err := os.WriteFile(*out, b, 0o644); err != nil {
			log.Fatal(err)
		}
	}
}
