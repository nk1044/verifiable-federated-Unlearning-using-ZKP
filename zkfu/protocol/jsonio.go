package protocol

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

func FrHex(e fr.Element) string { b := e.Bytes(); return hex.EncodeToString(b[:]) }

func FrFromHex(s string) (fr.Element, error) {
	var e fr.Element
	b, err := hex.DecodeString(s)
	if err != nil || len(b) != fr.Bytes {
		return e, fmt.Errorf("bad field element hex %q", s)
	}
	if err := e.SetBytesCanonical(b); err != nil {
		return e, err
	}
	return e, nil
}

func ForgotSet(xs []int) map[int]bool {
	m := make(map[int]bool, len(xs))
	for _, x := range xs {
		m[x] = true
	}
	return m
}

// ReadJSON reads a JSON document from a path ("-" = stdin).
func ReadJSON(path string, v any) error {
	var r io.Reader = os.Stdin
	if path != "-" {
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		defer f.Close()
		r = f
	}
	return json.NewDecoder(r).Decode(v)
}

// WriteJSON writes v to a path ("-" = stdout).
func WriteJSON(path string, v any) error {
	var w io.Writer = os.Stdout
	if path != "-" {
		f, err := os.Create(path)
		if err != nil {
			return err
		}
		defer f.Close()
		w = f
	}
	return json.NewEncoder(w).Encode(v)
}
