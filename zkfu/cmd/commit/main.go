// cmd/commit (client side): quantised features + labels -> Com(A) = root_A, plus a PRIVATE
// dataset file (salts, padded data) that the client keeps for proving.
//
// input  {"out_dir": "...", "clients": [{"client_id": 0, "features": [[...D ints]...], "labels": [...]}]}
// output {"clients": [{"client_id": 0, "root": "<hex>", "n_real": 1000, "dataset": "<path>", "ms": 12.3}]}
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/consensys/gnark/logger"
	"zkfu/commit"

	"zkfu/protocol"
)

type inClient struct {
	ClientID int       `json:"client_id"`
	Features [][]int64 `json:"features"`
	Labels   []int64   `json:"labels"`
}

type outClient struct {
	ClientID int     `json:"client_id"`
	Root     string  `json:"root"`
	NReal    int     `json:"n_real"`
	Dataset  string  `json:"dataset"`
	Ms       float64 `json:"ms"`
}

func main() {
	in := flag.String("in", "-", "input JSON (- = stdin)")
	out := flag.String("out", "-", "output JSON (- = stdout)")
	flag.Parse()
	logger.Disable() // keep stdout clean for JSON
	var req struct {
		OutDir  string     `json:"out_dir"`
		Clients []inClient `json:"clients"`
	}
	if err := protocol.ReadJSON(*in, &req); err != nil {
		log.Fatal(err)
	}
	if err := os.MkdirAll(req.OutDir, 0o755); err != nil {
		log.Fatal(err)
	}
	var res struct {
		Clients []outClient `json:"clients"`
	}
	for _, c := range req.Clients {
		t0 := time.Now()
		ds, err := commit.NewDataset(c.ClientID, c.Features, c.Labels)
		if err != nil {
			log.Fatalf("client %d: %v", c.ClientID, err)
		}
		tree, err := ds.Tree()
		if err != nil {
			log.Fatalf("client %d: %v", c.ClientID, err)
		}
		p := filepath.Join(req.OutDir, fmt.Sprintf("client_%03d.gob", c.ClientID))
		if err := ds.Save(p); err != nil {
			log.Fatal(err)
		}
		res.Clients = append(res.Clients, outClient{c.ClientID, protocol.FrHex(tree.Root()), ds.NReal, p,
			float64(time.Since(t0).Microseconds()) / 1000})
	}
	if err := protocol.WriteJSON(*out, res); err != nil {
		log.Fatal(err)
	}
}
