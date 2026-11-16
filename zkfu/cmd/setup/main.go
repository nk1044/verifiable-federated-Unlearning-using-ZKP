// cmd/setup: compile ZkStepCircuit, run the Groth16 setup, write pk/vk, print sizes and counts.
//
//	go run ./cmd/setup -keys keys            # compile + setup + write keys
//	go run ./cmd/setup -compile-only -breakdown
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"

	"zkfu/circuit"
	"zkfu/params"
	"zkfu/protocol"
)

func count(c frontend.Circuit) int {
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, c)
	if err != nil {
		log.Fatal(err)
	}
	return cs.GetNbConstraints()
}

func main() {
	keys := flag.String("keys", "keys", "output directory for pk.bin / vk.bin")
	compileOnly := flag.Bool("compile-only", false, "only compile and count constraints")
	breakdown := flag.Bool("breakdown", false, "also report per-component constraint counts")
	out := flag.String("json", "", "write a JSON summary here")
	flag.Parse()

	res := map[string]any{"D": params.D, "C": params.C, "B": params.B, "N": params.N, "FRAC": params.Frac, "S": params.S, "K": params.K}
	t0 := time.Now()
	ccs, err := protocol.Compile()
	if err != nil {
		log.Fatal(err)
	}
	res["constraints"] = ccs.GetNbConstraints()
	res["public_inputs"] = ccs.GetNbPublicVariables() - 1
	res["compile_s"] = time.Since(t0).Seconds()
	fmt.Printf("ZkStepCircuit: %d constraints, %d public inputs, compiled in %.2fs\n",
		ccs.GetNbConstraints(), ccs.GetNbPublicVariables()-1, time.Since(t0).Seconds())

	if *breakdown {
		// marginal costs: count(2 copies) - count(1 copy) removes the lookup-table fixed overhead
		marg := func(mk func(int) frontend.Circuit) int { return count(mk(2)) - count(mk(1)) }
		leaf := marg(func(m int) frontend.Circuit { return circuit.NewLeafPath(m) })
		head := marg(func(m int) frontend.Circuit { return circuit.NewHeadHash(m) })
		tr := func(t, n int) int {
			return marg(func(m int) frontend.Circuit {
				return &circuit.TruncCircuit{X: make([]frontend.Variable, m), Out: make([]frontend.Variable, m), T: t, N: n}
			})
		}
		tf, tw, tb := tr(params.Frac, params.NFwd), tr(params.Frac+params.S, params.NGW), tr(params.S, params.NGb)
		hc := marg(func(m int) frontend.Circuit { return &circuit.HashCircuit{In: make([]frontend.Variable, m)} })
		samples := leaf * params.B
		heads := 2 * head
		truncs := params.B*params.C*tf + params.C*params.D*tw + params.C*tb
		mults := params.B*params.C*params.D*2 + params.B*params.C
		bd := map[string]any{
			"poseidon2_compression":             hc,
			"per_sample_leaf_rangecheck_merkle": leaf,
			"all_samples_B":                     samples,
			"head_rangecheck_and_hash_each":     head,
			"head_hashes_x2":                    heads,
			"trunc_fwd_each":                    tf, "trunc_gw_each": tw, "trunc_gb_each": tb,
			"truncations_total":        truncs,
			"multiplications_fwd_grad": mults,
			"remainder_overhead":       ccs.GetNbConstraints() - samples - heads - truncs - mults,
		}
		res["breakdown"] = bd
		b, _ := json.MarshalIndent(bd, "", "  ")
		fmt.Println("breakdown (approx, small circuits carry a little fixed overhead):", string(b))
	}

	if !*compileOnly {
		t1 := time.Now()
		pk, vk, err := groth16.Setup(ccs)
		if err != nil {
			log.Fatal(err)
		}
		res["setup_s"] = time.Since(t1).Seconds()
		fmt.Printf("Groth16 setup (BN254): %.2fs\n", time.Since(t1).Seconds())
		if err := protocol.WriteKeys(*keys, pk, vk); err != nil {
			log.Fatal(err)
		}
		pkp, vkp := protocol.KeyPaths(*keys)
		for name, p := range map[string]string{"pk_bytes": pkp, "vk_bytes": vkp} {
			if st, err := os.Stat(p); err == nil {
				res[name] = st.Size()
				fmt.Printf("%s: %s (%.1f MB)\n", name, p, float64(st.Size())/1e6)
			}
		}
	}
	if *out != "" {
		b, _ := json.MarshalIndent(res, "", "  ")
		if err := os.WriteFile(*out, b, 0o644); err != nil {
			log.Fatal(err)
		}
	}
}
