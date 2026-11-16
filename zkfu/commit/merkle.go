package commit

import (
	"encoding/gob"
	"fmt"
	"os"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"

	"zkfu/params"
)

// Tree is a binary Merkle tree over exactly N leaves; node = H(left, right).
type Tree struct{ levels [][]fr.Element } // levels[0] = leaves, levels[Depth] = [root]

func BuildTree(leaves []fr.Element) (*Tree, error) {
	if len(leaves) != params.N {
		return nil, fmt.Errorf("merkle: need %d leaves, got %d", params.N, len(leaves))
	}
	t := &Tree{levels: [][]fr.Element{leaves}}
	cur := leaves
	for l := 0; l < params.Depth; l++ {
		next := make([]fr.Element, len(cur)/2)
		for i := range next {
			next[i] = H(cur[2*i], cur[2*i+1])
		}
		t.levels = append(t.levels, next)
		cur = next
	}
	return t, nil
}

func (t *Tree) Root() fr.Element { return t.levels[params.Depth][0] }

// Path returns the siblings of leaf idx from the bottom up. Bit l of idx (LSB first) is 1 when
// the running node is a right child at level l: this is exactly what the circuit uses.
func (t *Tree) Path(idx int) [params.Depth]fr.Element {
	var p [params.Depth]fr.Element
	for l := 0; l < params.Depth; l++ {
		p[l] = t.levels[l][idx^1]
		idx >>= 1
	}
	return p
}

// VerifyPath recomputes the root from a leaf, its index and its path (native check).
func VerifyPath(leaf fr.Element, idx int, path [params.Depth]fr.Element) fr.Element {
	cur := leaf
	for l := 0; l < params.Depth; l++ {
		if (idx>>l)&1 == 1 {
			cur = H(path[l], cur)
		} else {
			cur = H(cur, path[l])
		}
	}
	return cur
}

// Dataset is a client's PRIVATE committed dataset A (padded to N). Never sent to the server.
type Dataset struct {
	ClientID int
	NReal    int        // positions [NReal, N) are padding: features 0, label 0, mask forced 0
	Salts    [][32]byte // salt_i, a random field element in canonical big-endian bytes
	Features [][]int64  // N x D quantised features (value * 2^Frac)
	Labels   []int64
}

// NewDataset pads real samples to N and draws fresh secret salts.
func NewDataset(cid int, feats [][]int64, labels []int64) (*Dataset, error) {
	n := len(feats)
	if n == 0 || n > params.N || len(labels) != n {
		return nil, fmt.Errorf("dataset: need 1..%d samples with labels, got %d/%d", params.N, n, len(labels))
	}
	ds := &Dataset{ClientID: cid, NReal: n, Salts: make([][32]byte, params.N),
		Features: make([][]int64, params.N), Labels: make([]int64, params.N)}
	for i := 0; i < params.N; i++ {
		var s fr.Element
		if _, err := s.SetRandom(); err != nil {
			return nil, err
		}
		ds.Salts[i] = s.Bytes()
		if i < n {
			if len(feats[i]) != params.D {
				return nil, fmt.Errorf("dataset: sample %d has %d features", i, len(feats[i]))
			}
			ds.Features[i] = feats[i]
			ds.Labels[i] = labels[i]
		} else {
			ds.Features[i] = make([]int64, params.D) // padding sample: zero features, label 0
		}
	}
	return ds, nil
}

func (ds *Dataset) Salt(i int) fr.Element {
	var e fr.Element
	e.SetBytes(ds.Salts[i][:])
	return e
}

func (ds *Dataset) Tree() (*Tree, error) {
	leaves := make([]fr.Element, params.N)
	for i := 0; i < params.N; i++ {
		l, err := Leaf(ds.Salt(i), ds.Features[i], ds.Labels[i])
		if err != nil {
			return nil, fmt.Errorf("leaf %d: %w", i, err)
		}
		leaves[i] = l
	}
	return BuildTree(leaves)
}

func (ds *Dataset) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(ds)
}

func LoadDataset(path string) (*Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var ds Dataset
	if err := gob.NewDecoder(f).Decode(&ds); err != nil {
		return nil, err
	}
	return &ds, nil
}
