"""Contextual-embedding models."""

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from .static import StaticBertModel
from .utils import ArrayFloat


class SimpleContextualBertModel(StaticBertModel):
    """BERT contextual-embedding model (outputs)."""

    def _embeddings(self, context: str) -> ArrayFloat:
        return (
            self.model(**self.tokenizer(context, return_tensors="pt"))[0][0]
            .detach()
            .numpy()
        )

    def _embedding(self, word: str, context: str, word_context: str) -> ArrayFloat:
        if self.context_window_operation == "none" or self.context_window_size == 0:
            return self._embeddings(context)[self._find(word_context, context)]

        start, end = self._context_window(word, context, word_context)
        return self._compose(self._embeddings(context)[start:end])


class PooledContextualBertModel(StaticBertModel):
    """BERT contextual-embedding model (sum of last four hidden-states)."""

    def _embeddings(self, context: str) -> ArrayFloat:
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.model(
            **self.tokenizer(context, return_tensors="pt")
        )

        assert outputs.hidden_states is not None

        hidden_states0 = outputs.hidden_states[-1][0].detach().numpy()
        hidden_states1 = outputs.hidden_states[-2][0].detach().numpy()  # type: ignore
        hidden_states2 = outputs.hidden_states[-3][0].detach().numpy()  # type: ignore
        hidden_states3 = outputs.hidden_states[-4][0].detach().numpy()  # type: ignore

        return hidden_states0 + hidden_states1 + hidden_states2 + hidden_states3

    def _embedding(self, word: str, context: str, word_context: str) -> ArrayFloat:
        if self.context_window_operation == "none" or self.context_window_size == 0:
            return self._embeddings(context)[self._find(word_context, context)]

        start, end = self._context_window(word, context, word_context)
        return self._compose(self._embeddings(context)[start:end])
