"""Testes sintéticos de contrato: exigem as dependências em requirements.txt."""
import unittest
from unittest.mock import patch
import numpy as np
from separacao_final import segmentar_fatia

class TestContratoPipeline(unittest.TestCase):
    def test_mascara_vazia(self):
        imagem = np.zeros((16, 16), dtype=np.uint8)
        saida = segmentar_fatia(imagem, (1.0, 1.0))
        self.assertEqual(saida['status'], 'mascara_vazia')
        self.assertFalse(saida['mascara_combinada'].any())

    def test_contrato_duas_mascaras(self):
        imagem = np.zeros((8, 8), dtype=np.uint8)
        base = np.zeros((8, 8), dtype=bool)
        base[1:3, 1:3] = True
        base[5:7, 5:7] = True
        e = np.zeros_like(base); e[1:3, 1:3] = True
        d = np.zeros_like(base); d[5:7, 5:7] = True
        with patch('separacao_final.preparar_mascara_base', return_value=base), \
             patch('separacao_final.separar_pulmoes', return_value=(e, d, None)):
            r = segmentar_fatia(imagem, (1., 1.))
        self.assertEqual(r['status'], 'componentes_preservados_heuristica')
        self.assertTrue(np.array_equal(r['mascara_combinada'], base))
        self.assertFalse((r['pulmao_esquerdo'] & r['pulmao_direito']).any())

    def test_forma_invalida(self):
        with self.assertRaises(ValueError):
            segmentar_fatia(np.zeros((2, 2, 2)), (1., 1.))

    def test_impede_sobreposicao(self):
        base = np.ones((8, 8), dtype=bool)
        with patch('separacao_final.preparar_mascara_base', return_value=base), \
             patch('separacao_final.separar_pulmoes', return_value=(base, base, None)):
            with self.assertRaises(ValueError):
                segmentar_fatia(np.zeros((8, 8)), (1., 1.))

if __name__ == '__main__':
    unittest.main()
