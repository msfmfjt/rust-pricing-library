from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backtesting'))
from raw_variance_basis import matrix_document,load_matrix,coupled_fixture_basis
from custom_basis_cases import basis_for
from multi_step_cases import models
from rust_pricing_backtest.domain import BacktestError
from rust_pricing_backtest.multi_step_extension import MultiStepExtensionCalibrator as Calibrator


class RawVarianceBasisTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'basis.json';self.payload=models()[0]['request']
        self.basis=coupled_fixture_basis(self.payload);self.doc=matrix_document(self.payload,self.basis)

    def write(self,doc):self.path.write_text(json.dumps(doc))

    def test_roundtrip_preserves_signed_matrix_and_explicit_metric(self):
        self.write(self.doc);b,a=load_matrix(self.path,self.payload)
        np.testing.assert_array_equal(b,self.basis);self.assertLess(b.min(),0.)
        self.assertTrue(a['diagnostic_only'])
        with self.assertRaises(ValueError):b[0,0]=0.
        self.doc['values'][0][0]=99.;self.assertNotEqual(b[0,0],99.)

    def test_request_axes_metric_and_format_are_bound(self):
        for change in [dict(schema_version=True),dict(source_request_hash='other'),dict(variance_coordinate_order='other'),
                       dict(parameter_metric='normalized'),dict(time_nodes=[0.,1.]),dict(extra='ignored'),dict(values=[[1.]])]:
            self.write({**self.doc,**change})
            with self.subTest(change=next(iter(change))),self.assertRaises(BacktestError):load_matrix(self.path,self.payload)
        self.write(self.doc);changed=deepcopy(self.payload);changed['market']['spot']+=1.
        with self.assertRaisesRegex(BacktestError,'basis_input_binding'):load_matrix(self.path,changed)
        for v in (float('inf'),float('nan'),'1',True,10**400):
            doc=deepcopy(self.doc);doc['values'][0][0]=v;self.write(doc)
            with self.subTest(v=str(v)[:20]),self.assertRaises(BacktestError):load_matrix(self.path,self.payload)
        self.path.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaisesRegex(BacktestError,'basis_input_fields'):load_matrix(self.path,self.payload)

    def test_nonseparable_metric_changes_tangent_but_orthogonal_parameters_do_not(self):
        for case in models():
            p=case['request'];b=coupled_fixture_basis(p);base,_=basis_for(p,'row_hats')
            n=case['intervals'];m=33;metric=b@b.T
            s=np.linalg.svd(metric.reshape(n,m,n,m).transpose(0,2,1,3).reshape(n*n,m*m),compute_uv=False)
            self.assertGreater(s[1]/s[0],.01)
            c=Calibrator(p,extension='custom',variance_basis=b);t,a=c.all_maturity_iv_tangent()
            previous,_=Calibrator(p,extension='custom',variance_basis=base).all_maturity_iv_tangent()
            self.assertGreater(np.max(abs(t-previous)),.01)
            q,_=np.linalg.qr(np.random.default_rng(731).normal(size=(b.shape[1],b.shape[1])))
            rotated=Calibrator(p,extension='custom',variance_basis=b@q);tr,_=rotated.all_maturity_iv_tangent()
            np.testing.assert_allclose(t,tr,rtol=0,atol=1e-10)
            shift=np.linspace(-.0002,.0003,c.iv.size).reshape(c.iv.shape)
            fitted,_=c.calibrate_all_maturities(shift);other,_=rotated.calibrate_all_maturities(shift)
            np.testing.assert_allclose(fitted['model']['local_variance_grid']['values'],other['model']['local_variance_grid']['values'],rtol=0,atol=1e-12)

    def test_unspanned_quotes_are_rejected(self):
        for b in (np.zeros_like(self.basis),np.ones((len(self.basis),1))):
            self.write(matrix_document(self.payload,b));matrix,_=load_matrix(self.path,self.payload)
            with self.assertRaisesRegex(BacktestError,'extension_rank'):
                Calibrator(self.payload,extension='custom',variance_basis=matrix).all_maturity_iv_tangent()


if __name__=='__main__':unittest.main()
