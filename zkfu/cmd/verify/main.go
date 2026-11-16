// cmd/verify (server side): recompute every public input from the server's own state and verify.
// The client never supplies public inputs; it only supplies the head it sent back and the proof.
//
// input  {"keys": "keys", "checks": [{"client_id": 0, "round": 3, "root": "<hex>", "n_real": 1000,
//
//	"W": [[...]], "b": [...], "W2": [[...]], "b2": [...], "forgotten": [...], "proof": "<base64>"}]}
//
// output {"results": [{"client_id": 0, "ok": true, "error": "", "verify_ms": 1.2}], "vk_load_ms": ..}
package main

import (
	"encoding/base64"
	"flag"
	"log"
	"time"

	"github.com/consensys/gnark/logger"

	"zkfu/protocol"
)

type check struct {
	ClientID  int       `json:"client_id"`
	Round     uint64    `json:"round"`
	Root      string    `json:"root"`
	NReal     int       `json:"n_real"`
	W         [][]int64 `json:"W"`
	B         []int64   `json:"b"`
	W2        [][]int64 `json:"W2"`
	B2        []int64   `json:"b2"`
	Forgotten []int     `json:"forgotten"`
	Proof     string    `json:"proof"`
}

type result struct {
	ClientID int     `json:"client_id"`
	OK       bool    `json:"ok"`
	Error    string  `json:"error,omitempty"`
	VerifyMs float64 `json:"verify_ms"`
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

func main() {
	in := flag.String("in", "-", "input JSON")
	out := flag.String("out", "-", "output JSON")
	flag.Parse()
	logger.Disable() // keep stdout clean for JSON
	var req struct {
		Keys   string  `json:"keys"`
		Checks []check `json:"checks"`
	}
	if err := protocol.ReadJSON(*in, &req); err != nil {
		log.Fatal(err)
	}
	var resp struct {
		Results  []result `json:"results"`
		VkLoadMs float64  `json:"vk_load_ms"`
		TotalMs  float64  `json:"total_ms"`
	}
	t0 := time.Now()
	vk, err := protocol.LoadVK(req.Keys)
	if err != nil {
		log.Fatal(err)
	}
	resp.VkLoadMs = ms(time.Since(t0))
	for _, c := range req.Checks {
		r := result{ClientID: c.ClientID}
		tv := time.Now()
		err := func() error {
			root, err := protocol.FrFromHex(c.Root)
			if err != nil {
				return err
			}
			pub, err := protocol.ServerPublic(root, c.NReal, c.W, c.B, c.W2, c.B2, c.Round, uint64(c.ClientID),
				protocol.ForgotSet(c.Forgotten))
			if err != nil {
				return err
			}
			proof, err := base64.StdEncoding.DecodeString(c.Proof)
			if err != nil {
				return err
			}
			return protocol.Verify(vk, proof, pub)
		}()
		r.VerifyMs = ms(time.Since(tv))
		r.OK = err == nil
		if err != nil {
			r.Error = err.Error()
		}
		resp.Results = append(resp.Results, r)
	}
	resp.TotalMs = ms(time.Since(t0))
	if err := protocol.WriteJSON(*out, resp); err != nil {
		log.Fatal(err)
	}
}
