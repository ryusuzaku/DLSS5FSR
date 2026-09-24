#!/usr/bin/env python3
"""Synthetic recovery mechanics tests; these do NOT recover the model's maps."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import recover_head70_maps as R


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        scratch=R.ROOT/'build'
        scratch.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='synthetic-map-test-',dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.assets=self.root/'native-game-tiled-assets'
        self.assets.mkdir()
        records={x['name']:x for x in json.loads((R.ROOT/'dlss5-analysis/model.resolved.json').read_text())['tensors']}
        self.truth={}
        rng=np.random.default_rng(244)
        for block in R.TRAIN+R.HOLDOUT:
            body,original=R.raw_body(block,records)
            raw=R.regions(body)
            values={}
            for key,a in raw.items():
                if key not in self.truth:
                    n=len(a);index=np.arange(n,dtype=np.int32)
                    bits=rng.permutation(n.bit_length()-1)
                    self.truth[key]=sum(((index>>int(s))&1)<<d for d,s in enumerate(bits))
                    if key=='g1':self.truth[key]=np.asarray(R.HW.ORDER)
                    if key in ('k','v'):self.truth[key]=self.truth['q']
                values[key]=np.empty_like(a)
                values[key][self.truth[key]]=a
            stem='post70' if block==70 else f'block{block}'
            np.concatenate([np.zeros(512),values['w1'],values['w2'],values['g1']]).astype('<f2').tofile(self.assets/(stem+'-ffn.f16'))
            np.concatenate([values[k] for k in ('q','k','v','p','bias')]+
                           [np.frombuffer(body,'<f4',1,19552),values['g2']]).astype('<f4').tofile(self.assets/(stem+'-attention.f32'))
            if block==70:
                stage=R.raw_body(1,records)[0]
                _,sm,ss,coeff,_=R.HW.extract(original,stage[R.HW.PAD_AT:R.HW.PAD_AT+16])
                np.concatenate((sm,ss)).astype('<f2').tofile(self.assets/'post70-scales.f16')
                coeff[[0,2,4]].astype('<f2').tofile(self.assets/'post70-head.f16')

    def recover(self,path,out):
        assets=R.Assets(path)
        try:return R.recover(assets,out)
        finally:assets.close()

    def test_directory_and_zip_exact(self):
        archive=self.root/'synthetic.zip'
        with zipfile.ZipFile(archive,'w') as z:
            for p in self.assets.iterdir():z.write(p,'test/native-game-tiled-assets/'+p.name)
        for i,path in enumerate((self.assets,archive)):
            out=self.root/f'recovered{i}'
            report=self.recover(path,out)
            self.assertEqual(len(report['output_hashes']),3)
            for key,evidence in report['regions'].items():
                index=np.arange(len(self.truth[key]),dtype=np.int32)
                actual=sum(((index>>s)&1)<<d for d,s in enumerate(evidence['source_bits_for_destination']))
                np.testing.assert_array_equal(actual,self.truth[key])
            with self.assertRaises(FileExistsError):self.recover(path,out)

    def test_heldout_corruption_rejected(self):
        p=self.assets/'block69-attention.f32'
        data=np.fromfile(p,'<f4');data[0]+=1;data.tofile(p)
        out=self.root/'invalid'
        with self.assertRaisesRegex(ValueError,'held-out block69 q mismatch'):
            self.recover(self.assets,out)
        self.assertFalse(out.exists())

    def test_consumer_failure_never_publishes(self):
        out=self.root/'invalid'
        with patch.object(R.N,'load_recovered',side_effect=ValueError('consumer rejection')):
            with self.assertRaisesRegex(ValueError,'consumer rejection'):
                self.recover(self.assets,out)
        self.assertFalse(out.exists())
        self.assertFalse(list(self.root.glob('head70-map-validation-*')))

    def test_ambiguous_and_non_bit_permutations_rejected(self):
        with self.assertRaisesRegex(ValueError,'not enough unique'):
            R.infer_permutation(np.zeros((5,32)),np.zeros((5,32)))
        a=np.arange(32,dtype=np.float32)[None,:]
        b=a.copy();b[0,[0,1]]=b[0,[1,0]]
        with self.assertRaisesRegex(ValueError,'non-permutation'):
            R.infer_permutation(a,b)


if __name__=='__main__':unittest.main(verbosity=2)
