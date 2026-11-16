package protocol

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"path/filepath"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"

	"zkfu/circuit"
)

// Compile builds the R1CS of ZkStepCircuit (deterministic; cheaper than (de)serialising it).
func Compile() (constraint.ConstraintSystem, error) {
	return frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit.ZkStepCircuit{})
}

func KeyPaths(dir string) (pk, vk string) {
	return filepath.Join(dir, "pk.bin"), filepath.Join(dir, "vk.bin")
}

func WriteKeys(dir string, pk groth16.ProvingKey, vk groth16.VerifyingKey) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	pkp, vkp := KeyPaths(dir)
	f, err := os.Create(pkp)
	if err != nil {
		return err
	}
	bw := bufio.NewWriterSize(f, 1<<20)
	if _, err := pk.WriteRawTo(bw); err != nil {
		return err
	}
	if err := bw.Flush(); err != nil {
		return err
	}
	f.Close()
	g, err := os.Create(vkp)
	if err != nil {
		return err
	}
	defer g.Close()
	_, err = vk.WriteRawTo(g)
	return err
}

func LoadPK(dir string) (groth16.ProvingKey, error) {
	pkp, _ := KeyPaths(dir)
	f, err := os.Open(pkp)
	if err != nil {
		return nil, fmt.Errorf("open pk (run cmd/setup first): %w", err)
	}
	defer f.Close()
	pk := groth16.NewProvingKey(ecc.BN254)
	_, err = pk.UnsafeReadFrom(bufio.NewReaderSize(f, 1<<20))
	return pk, err
}

func LoadVK(dir string) (groth16.VerifyingKey, error) {
	_, vkp := KeyPaths(dir)
	f, err := os.Open(vkp)
	if err != nil {
		return nil, fmt.Errorf("open vk (run cmd/setup first): %w", err)
	}
	defer f.Close()
	vk := groth16.NewVerifyingKey(ecc.BN254)
	_, err = vk.UnsafeReadFrom(f)
	return vk, err
}

// Prove creates a Groth16 proof for a full assignment and returns its serialised bytes.
func Prove(ccs constraint.ConstraintSystem, pk groth16.ProvingKey, a *circuit.ZkStepCircuit) ([]byte, error) {
	w, err := frontend.NewWitness(a, ecc.BN254.ScalarField())
	if err != nil {
		return nil, err
	}
	proof, err := groth16.Prove(ccs, pk, w)
	if err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	if _, err := proof.WriteTo(&buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// Verify checks a serialised proof against server-computed public inputs.
func Verify(vk groth16.VerifyingKey, proofBytes []byte, pub Public) error {
	proof := groth16.NewProof(ecc.BN254)
	if _, err := proof.ReadFrom(bytes.NewReader(proofBytes)); err != nil {
		return fmt.Errorf("malformed proof: %w", err)
	}
	a := &circuit.ZkStepCircuit{}
	pub.AssignPublic(a)
	w, err := frontend.NewWitness(a, ecc.BN254.ScalarField(), frontend.PublicOnly())
	if err != nil {
		return err
	}
	return groth16.Verify(proof, vk, w)
}
