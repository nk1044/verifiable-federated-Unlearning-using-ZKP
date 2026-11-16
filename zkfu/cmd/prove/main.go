// cmd/prove (client side): one honest fixed-point SGD step per job, optionally with a Groth16 proof.
// Batched so one process (one proving-key load) serves all clients of a round.
//
// input  {"keys": "keys", "jobs": [{"client_id": 0, "dataset": "...gob", "round": 3,
//
//	"W": [[...]], "b": [...], "forgotten": [12, 40], "prove": true}]}
//
// output {"results": [{"client_id": 0, "round": 3, "W2": ..., "b2": ..., "proof": "<base64>",
//
//	"native_ms": .., "witness_ms": .., "prove_ms": .., "error": ""}], "pk_load_ms": .., "total_ms": ..}
//
// "prove": false runs only the native step (used for un-audited rounds; the client keeps the
// round's inputs and can prove later if the server audits that round).
package main

import (
	"encoding/base64"
	"flag"
	"log"
	"time"

	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/constraint"

	"github.com/consensys/gnark/logger"
	"zkfu/commit"

	"zkfu/protocol"
)

type job struct {
	ClientID  int       `json:"client_id"`
	Dataset   string    `json:"dataset"`
	Round     uint64    `json:"round"`
	W         [][]int64 `json:"W"`
	B         []int64   `json:"b"`
	Forgotten []int     `json:"forgotten"`
	Prove     bool      `json:"prove"`
}

type result struct {
	ClientID  int       `json:"client_id"`
	Round     uint64    `json:"round"`
	W2        [][]int64 `json:"W2"`
	B2        []int64   `json:"b2"`
	Proof     string    `json:"proof,omitempty"`
	ProofLen  int       `json:"proof_bytes,omitempty"`
	NativeMs  float64   `json:"native_ms"`
	WitnessMs float64   `json:"witness_ms,omitempty"`
	ProveMs   float64   `json:"prove_ms,omitempty"`
	Error     string    `json:"error,omitempty"`
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

func main() {
	in := flag.String("in", "-", "input JSON")
	out := flag.String("out", "-", "output JSON")
	flag.Parse()
	logger.Disable() // keep stdout clean for JSON
	var req struct {
		Keys string `json:"keys"`
		Jobs []job  `json:"jobs"`
	}
	if err := protocol.ReadJSON(*in, &req); err != nil {
		log.Fatal(err)
	}
	t0 := time.Now()
	var (
		ccs constraint.ConstraintSystem
		pk  groth16.ProvingKey
		err error
	)
	var resp struct {
		Results   []result `json:"results"`
		PkLoadMs  float64  `json:"pk_load_ms"`
		CompileMs float64  `json:"compile_ms"`
		TotalMs   float64  `json:"total_ms"`
	}
	for _, j := range req.Jobs {
		if j.Prove && pk == nil {
			tc := time.Now()
			if ccs, err = protocol.Compile(); err != nil {
				log.Fatal(err)
			}
			resp.CompileMs = ms(time.Since(tc))
			tl := time.Now()
			if pk, err = protocol.LoadPK(req.Keys); err != nil {
				log.Fatal(err)
			}
			resp.PkLoadMs = ms(time.Since(tl))
			break
		}
	}
	cache := map[string]*commit.Dataset{}
	for _, j := range req.Jobs {
		r := result{ClientID: j.ClientID, Round: j.Round}
		ds, ok := cache[j.Dataset]
		if !ok {
			if ds, err = commit.LoadDataset(j.Dataset); err != nil {
				r.Error = err.Error()
				resp.Results = append(resp.Results, r)
				continue
			}
			cache[j.Dataset] = ds
		}
		tn := time.Now()
		st, err := protocol.HonestStep(ds, j.W, j.B, j.Round, protocol.ForgotSet(j.Forgotten))
		r.NativeMs = ms(time.Since(tn))
		if err != nil {
			r.Error = err.Error()
			resp.Results = append(resp.Results, r)
			continue
		}
		r.W2, r.B2 = st.W2, st.B2
		if j.Prove {
			if err := proveJob(ccs, pk, ds, st, &r); err != nil {
				r.Error = err.Error()
			}
		}
		resp.Results = append(resp.Results, r)
	}
	resp.TotalMs = ms(time.Since(t0))
	if err := protocol.WriteJSON(*out, resp); err != nil {
		log.Fatal(err)
	}
}

// proveJob builds the witness (Merkle tree + paths) and the Groth16 proof for one step.
func proveJob(ccs constraint.ConstraintSystem, pk groth16.ProvingKey, ds *commit.Dataset, st *protocol.ClientStep, r *result) error {
	tw := time.Now()
	tree, err := ds.Tree()
	if err != nil {
		return err
	}
	a, err := st.Assignment(tree)
	if err != nil {
		return err
	}
	r.WitnessMs = ms(time.Since(tw))
	tp := time.Now()
	proof, err := protocol.Prove(ccs, pk, a)
	r.ProveMs = ms(time.Since(tp))
	if err != nil {
		return err
	}
	r.Proof = base64.StdEncoding.EncodeToString(proof)
	r.ProofLen = len(proof)
	return nil
}
