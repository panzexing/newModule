import unittest
from types import SimpleNamespace

import numpy as np
import torch

from nc_model.graph import graph_from_data
from nc_model.marginal_diffusion import build_model, combine, ranking_loss


class MarginalDiffusionTests(unittest.TestCase):
    def test_public_encoder_on_compact_graph(self):
        graph = graph_from_data(SimpleNamespace(
            x=np.arange(20).reshape(5,4)/20, ppi_edges=np.array([[0,1],[1,2],[2,3]]),
            herb_members=[[0],[1,2],[2,4]], symptom_members=[[0,3],[1]]))
        pairs = torch.cartesian_prod(torch.arange(3), torch.arange(2))
        labels = torch.tensor([[True, False], [False, True], [False, True]])
        prior = labels.sum(1).float().log1p()
        for diffuse in (False, True):
            torch.manual_seed(3)
            model = build_model(dict(hidden_dim=8,num_layers=2,seed=3,dropout=0.),4,diffusion=diffuse)
            logits = model.decode(model.encode(graph), pairs).reshape(3,2)
            loss = ranking_loss(logits, labels, prior)
            expected = torch.stack([-torch.log_softmax(logits[:,i]+prior,0)[labels[:,i]].mean()
                                    for i in range(2)]).mean()
            torch.testing.assert_close(loss, expected)
            loss.backward()
            if diffuse:
                gradient = model.encoder.layers[0].ppi.hop_logits.grad
                self.assertEqual(tuple(gradient.shape), (4,))
                self.assertGreater(float(gradient.abs().sum()),0.)

    def test_calibration_and_constant_columns(self):
        a = np.array([[1.,1.],[1.,3.]])
        prior = np.array([0.,2.])
        np.testing.assert_allclose(combine(a,prior,.5,"original"),[[0.,-.5],[2.,2.5]])
        np.testing.assert_allclose(combine(a,prior,.5,"calibrated"),[[1.,1.],[2.,4.]])


if __name__=="__main__":
    unittest.main()
