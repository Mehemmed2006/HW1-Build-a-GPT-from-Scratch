
import unittest

import torch

from model import GPT, GPTConfig


class TestGPTModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = GPTConfig()
        self.model = GPT(self.config)

    def test_output_shape(self):
        idx = torch.randint(0, 1024, (2, 10))
        logits, loss = self.model(idx)

        self.assertEqual(tuple(logits.shape), (2, 10, 1024))
        self.assertIsNone(loss)

    def test_loss_is_finite(self):
        idx = torch.randint(0, 1024, (2, 10))
        targets = torch.randint(0, 1024, (2, 10))

        _, loss = self.model(idx, targets)

        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss).item())

    def test_parameter_count(self):
        expected = (
            self.config.vocab_size * self.config.n_embd
            + self.config.block_size * self.config.n_embd
            + self.config.n_layer
            * (
                12 * self.config.n_embd**2
                + 4 * self.config.n_embd
            )
            + 2 * self.config.n_embd
        )

        self.assertEqual(self.model.num_params(), expected)
        self.assertEqual(self.model.num_params(), 936192)

    def test_shared_weights(self):
        self.assertIs(
            self.model.lm_head.weight,
            self.model.token_embedding.weight,
        )

    def test_causal_attention(self):
        self.model.eval()

        x1 = torch.tensor([[10, 20, 30, 40]])
        x2 = torch.tensor([[10, 20, 99, 100]])

        with torch.no_grad():
            y1, _ = self.model(x1)
            y2, _ = self.model(x2)

        self.assertTrue(
            torch.allclose(y1[:, :2, :], y2[:, :2, :])
        )

    def test_cpu_forward_and_backward(self):
        model = GPT(self.config).cpu()
        idx = torch.randint(0, 1024, (2, 10))
        targets = torch.randint(0, 1024, (2, 10))

        logits, loss = model(idx, targets)
        loss.backward()

        self.assertEqual(logits.device.type, "cpu")
        self.assertTrue(torch.isfinite(loss).item())

        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(
                    torch.isfinite(parameter.grad).all().item()
                )

    def test_invalid_sequence_length(self):
        idx = torch.randint(
            0,
            self.config.vocab_size,
            (1, self.config.block_size + 1),
        )

        with self.assertRaises(ValueError):
            self.model(idx)

    def test_invalid_input_dimensions(self):
        idx = torch.randint(0, 1024, (10,))

        with self.assertRaises(ValueError):
            self.model(idx)

    def test_invalid_target_shape(self):
        idx = torch.randint(0, 1024, (2, 10))
        targets = torch.randint(0, 1024, (2, 9))

        with self.assertRaises(ValueError):
            self.model(idx, targets)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            GPTConfig(n_embd=127, n_head=4)

        with self.assertRaises(ValueError):
            GPTConfig(dropout=1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
