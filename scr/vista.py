# Encoder training
import torch
from safetensors.torch import load_file
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
import os
import random
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
from sacrebleu import corpus_bleu, corpus_ter
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.encoder_decoder.modeling_encoder_decoder import shift_tokens_right

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

class MTDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, padding_value, targets=['eng_Latn']):
        self.padding_value = padding_value
        self.samples = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        self.samples = [f for f in self.samples if os.path.isdir(f)]
        self.targets = targets
    
    def __len__(self):
        return len(self.samples) * len(self.targets)
    
    def __getitem__(self, idx):
        sample_path = self.samples[idx % len(self.samples)]
        img_path = os.path.join(sample_path, "image.safetensors")
        image = load_file(img_path)
        lang = self.targets[idx // len(self.samples)]
        target = load_file(os.path.join(sample_path, f'{lang}_txt_tok_ids.safetensors'))
        source = load_file(os.path.join(sample_path, f'src_txt_tok_ids.safetensors'))
        #eos_marks = torch.zeros(sample['tgt_txt_lat'].shape[0]+1, dtype=torch.float32)
        #eos_marks[-1] = 1.0

        return {
            'pixel_values': image['pixel_values'],
            'image_grid_thw': image['image_grid_thw'],
            'tgt': target['tgt'],
            'src_txt': source['tgt'],
            'tgt_lang': lang,
        }

    def collate_fn(self, batch):
        tgt = pad_sequence([t['tgt'] for t in batch], batch_first=True, padding_value=self.padding_value)
        src_txt = pad_sequence([st['src_txt'] for st in batch], batch_first=True, padding_value=self.padding_value)

        pixel_values_list = []
        grid_thw_list = []

        for b in batch:
            p_val = b['pixel_values'].squeeze() if b['pixel_values'].ndim > 4 else b['pixel_values']
            if p_val.ndim == 3:
                p_val = p_val.unsqueeze(0)
            pixel_values_list.append(p_val)

            g_thw = b['image_grid_thw'].squeeze()
            grid_thw_list.append(g_thw)

        grid_thw = torch.stack(grid_thw_list, dim=0)

        # Concatenate all images' patches into one sequence, then add the
        # required dummy batch dim of size 1 (model only supports batch_size==1
        # and instead uses grid_thw to slice per-image patches out of this
        # single flattened sequence — see PaddleOCRVisionEmbeddings.forward).
        pixel_values = torch.cat(pixel_values_list, dim=0).unsqueeze(0)  # [1, total_patches, C, ph, pw]

        patches_per_image = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
        cu_seqlens = torch.zeros(patches_per_image.shape[0] + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(patches_per_image, dim=0)

        return {
            'pixel_values': pixel_values,      # [1, total_patches, C, ph, pw]
            'image_grid_thw': grid_thw,        # [B, 3]
            'cu_seqlens': cu_seqlens,
            'tgt': tgt,
            'src_txt': src_txt,
            'tgt_langs': [b['tgt_lang'] for b in batch],
        }

class MTDataModule(pl.LightningDataModule):
    def __init__(self, 
                 train_data_dir, 
                 val_data_dir, 
                 pad_token_id,
                 train_batch_size=32, 
                 val_batch_size=32,
                 num_workers=4,
                 tgt_langs=['eng_Latn']):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pad_token_id = pad_token_id
        self.tgt_langs = tgt_langs
    
    def setup(self, stage=None):
        self.train_dataset = MTDataset(
            self.train_data_dir,
            padding_value=self.pad_token_id,
            targets=self.tgt_langs
        )
        self.val_dataset = MTDataset(
            self.val_data_dir,
            padding_value=self.pad_token_id,
            targets=self.tgt_langs
        )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            prefetch_factor=2 if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.train_dataset.collate_fn
        )
 
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            prefetch_factor=2 if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.val_dataset.collate_fn
        )


class FlashCrossAttention(nn.Module):
    """Cross-attention using Flash Attention 2 via torch.nn.functional.scaled_dot_product_attention.
    Expects inputs in (B, T, E) format (batch_first=True semantics).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: torch.Tensor,           # (B, Tq, E)
        key: torch.Tensor,             # (B, Tk, E)
        value: torch.Tensor,           # (B, Tk, E)
        key_padding_mask: torch.Tensor = None,  # (B, Tk), True = ignore
    ) -> torch.Tensor:
        B, Tq, _ = query.shape
        Tk = key.shape[1]

        # Project and reshape to (B, heads, T, head_dim)
        q = self.q_proj(query).view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)

        # Convert key_padding_mask (B, Tk) -> attention bias (B, 1, 1, Tk)
        attn_bias = None
        if key_padding_mask is not None:
            # True positions should be masked; fill with -inf
            attn_bias = torch.zeros(B, 1, 1, Tk, device=query.device, dtype=query.dtype)
            attn_bias = attn_bias.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        # Flash Attention 2 via SDPA (uses FA2 kernel when available on CUDA)
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=dropout_p,
        )  # (B, heads, Tq, head_dim)

        # Reshape back to (B, Tq, E)
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.embed_dim)
        return self.out_proj(out)


class FlashSelfAttentionLayer(nn.Module):
    """Transformer encoder layer using Flash Attention 2 for self-attention."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = FlashCrossAttention(d_model, nhead, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x)))
        x = x + self.ff(self.norm2(x))
        return x

class RotaryPositionalEmbeddings(nn.Module):

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: int = 10000,
    ):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError("dim must be even")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float() / dim)
        )

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        input_pos: torch.Tensor = None,
    ):
        """
        x:        [B, N, D]
        input_pos: [N]  or  [B, N]
        """
        B, N, D = x.shape

        if D != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {D}")

        device = x.device

        if input_pos is None:
            positions = torch.arange(N, device=device, dtype=self.inv_freq.dtype)
            positions = positions.unsqueeze(0).expand(B, N)          # [B, N]
        else:
            positions = input_pos.to(device=device, dtype=self.inv_freq.dtype)
            if positions.dim() == 1:
                positions = positions.unsqueeze(0).expand(B, N)      # [B, N]
            # else: already [B, N]

        # [B, N, D/2]  (batched outer product)
        freqs = positions.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)

        cos = freqs.cos()   # [B, N, D/2]
        sin = freqs.sin()   # [B, N, D/2]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        x_out = torch.stack([x_rot_even, x_rot_odd], dim=-1)
        x_out = x_out.flatten(-2)

        return x_out.type_as(x)

class Rotary2D(nn.Module):

    def __init__(
        self,
        dim,
        max_height=256,
        max_width=256,
        base=10000,
    ):
        super().__init__()

        if dim % 4 != 0:
            raise ValueError(
                "Embedding dimension must be divisible by 4"
            )

        half_dim = dim // 2

        self.rope_y = RotaryPositionalEmbeddings(
            dim=half_dim,
            max_seq_len=max_height,
            base=base,
        )

        self.rope_x = RotaryPositionalEmbeddings(
            dim=half_dim,
            max_seq_len=max_width,
            base=base,
        )

    def _build_position_ids_batched(self, grid_hw, seq_len, device):
        """
        grid_hw: [B, 2] real (h, w) per image
        seq_len: padded sequence length (S_max)
        returns row_ids, col_ids of shape [B, seq_len] (padded slots get 0, ignored via attn_mask)
        """
        B = grid_hw.shape[0]
        row_ids = torch.zeros(B, seq_len, dtype=torch.long, device=device)
        col_ids = torch.zeros(B, seq_len, dtype=torch.long, device=device)
        for i in range(B):
            h, w = int(grid_hw[i, 0]), int(grid_hw[i, 1])
            n = h * w
            r = torch.arange(h, device=device).repeat_interleave(w)
            c = torch.arange(w, device=device).repeat(h)
            row_ids[i, :n] = r
            col_ids[i, :n] = c
        return row_ids, col_ids

    def forward(self, x, grid_hw):
        """
        x: [B, N, D]  (N = padded seq_len)
        grid_hw: [B, 2] real (h, w) per image, h*w <= N
        """
        B, N, D = x.shape
        row_ids, col_ids = self._build_position_ids_batched(grid_hw, N, x.device)

        x_y, x_x = x.chunk(2, dim=-1)
        x_y = self.rope_y(x_y, row_ids)
        x_x = self.rope_x(x_x, col_ids)
        x = torch.cat([x_y, x_x], dim=-1)
        return x

class PerceiverResampler(nn.Module):
    """Perceiver-style resampler that turns dense vision tokens into a small
    sequence of latent tokens suitable for a language encoder.

    - Cross-attends learned latents to vision tokens (queries=latents, keys/values=vision)
    - Applies a small stack of Transformer-style self-attention layers on latents
    - Projects latents to `text_hidden_size` for compatibility with the LM encoder

    Args:
        visual_hidden_size: dimensionality of input vision tokens
        text_hidden_size: desired output embedding size (LM embedding dim)
        padding_vector: tensor used for padding token embeddings
        num_latents: number of latent tokens to produce (M)
        latent_dim: internal latent dimensionality used for attention
        num_self_layers: number of self-attention layers applied to latents
        cross_heads: number of attention heads for cross-attention
        self_heads: number of heads for latent self-attention
        dropout: dropout probability
        use_flash_attn: if True, use Flash Attention 2 via
            torch.nn.functional.scaled_dot_product_attention (requires PyTorch >= 2.0
            and a CUDA device with FA2 support for full speed benefit; falls back
            gracefully to a standard SDPA kernel on CPU/older GPUs).
            If False, uses nn.MultiheadAttention and nn.TransformerEncoderLayer
            (original behaviour).
    """

    def __init__(
        self,
        input_hidden_size: int,
        output_hidden_size: int,
        num_latents: int = 128,
        latent_dim: int = 512,
        num_self_layers: int = 2,
        cross_heads: int = 8,
        self_heads: int = 8,
        dropout: float = 0.1,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.use_flash_attn = use_flash_attn

        # Learned latent queries
        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim) * 0.02)
        self.positions = nn.Parameter(torch.randn(1,num_latents, output_hidden_size))
        self.rope = Rotary2D(latent_dim, max_height=16, max_width=16)

        # Project vision tokens to key/value dimension if needed
        if input_hidden_size != latent_dim:
            self.kv_proj = nn.Linear(input_hidden_size, latent_dim)
        else:
            self.kv_proj = nn.Identity()

        # ── Cross-attention: latents ← vision ────────────────────────────────
        if use_flash_attn:
            self.cross_attn = FlashCrossAttention(latent_dim, cross_heads, dropout=dropout)
        else:
            self.cross_attn = nn.MultiheadAttention(
                latent_dim, cross_heads, dropout=dropout, batch_first=True
            )
        self.cross_norm = nn.LayerNorm(latent_dim)

        # ── Self-attention stack over latents ────────────────────────────────
        if use_flash_attn:
            self.self_layers = nn.ModuleList([
                FlashSelfAttentionLayer(
                    d_model=latent_dim,
                    nhead=self_heads,
                    dim_feedforward=latent_dim * 4,
                    dropout=dropout,
                )
                for _ in range(num_self_layers)
            ])
        else:
            self.self_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=self_heads,
                    dim_feedforward=latent_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_self_layers)
            ])
        self.self_norm = nn.LayerNorm(latent_dim)

        # Output projection into text embedding dimension
        self.to_text = nn.Sequential(
            nn.Linear(latent_dim, output_hidden_size),
            nn.RMSNorm(output_hidden_size)
        )

    def forward(self, vision_outputs: torch.Tensor, attn_mask: torch.Tensor = None, grid_hw: torch.Tensor = None):
        """Forward pass.

        Args:
            vision_outputs: (B, N, D_v)
            attn_mask: (B, N) with 1 for valid tokens, 0 for padding

        Returns:
            (B, M, output_hidden_size)
        """
        bsz, seq_len, _ = vision_outputs.shape
        attn_mask = attn_mask if attn_mask is not None else torch.ones(bsz, seq_len, device=vision_outputs.device, dtype=torch.bool)

        kv = self.kv_proj(vision_outputs)
        kv = self.rope(kv, grid_hw)   # grid_hw: [B, 2], no more sqrt(seq_len) guessing

        # Expand learned latents to batch
        latents = self.latents.expand(bsz, -1, -1)

        # key_padding_mask: True → ignore that position
        if attn_mask is None:
            key_padding_mask = None
        else:
            key_padding_mask = ~attn_mask.bool()  # (B, N)
        

        # ── Cross-attention ───────────────────────────────────────────────────
        if self.use_flash_attn:
            latents_attn = self.cross_attn(latents, kv, kv, key_padding_mask=key_padding_mask)
        else:
            latents_attn, _ = self.cross_attn(
                latents, kv, kv, key_padding_mask=key_padding_mask
            )
        latents = self.cross_norm(latents + latents_attn)

        # ── Self-attention layers on latents ─────────────────────────────────
        for layer in self.self_layers:
            latents = layer(latents)
        latents = self.self_norm(latents)
        latents = latents

        # Project to text hidden size
        return self.to_text(latents) + self.positions.expand(bsz, -1, -1)

def unflatten_to_batch(vision_outputs: torch.Tensor, cu_seqlens: torch.Tensor):
    """
    vision_outputs: [total_patches, D]  (e.g. [5384, 1152])
    cu_seqlens:     [B+1] boundaries, e.g. [0, 672, 1344, ...]
    returns:
        padded:   [B, S_max, D]
        pad_mask: [B, S_max]  (True = real token, False = padding)
    """
    seqs = [
        vision_outputs[cu_seqlens[i]:cu_seqlens[i + 1]]
        for i in range(len(cu_seqlens) - 1)
    ]  # list of [S_i, D], lengths vary per image

    lengths = torch.tensor([s.shape[0] for s in seqs], device=vision_outputs.device)

    padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)  # [B, S_max, D]

    S_max = padded.shape[1]
    pad_mask = torch.arange(S_max, device=vision_outputs.device)[None, :] < lengths[:, None]  # [B, S_max]

    return padded, pad_mask

class Vista(nn.Module):
    def __init__(
            self,
            input_hidden_size: int,
            output_hidden_size: int,
            num_latents: int = 128,
            latent_dim: int = 512,
            num_self_layers: int = 2,
            cross_heads: int = 8,
            self_heads: int = 8,
            dropout: float = 0.1,
            use_flash_attn: bool = False,
            padding_val: int = 1,
            ctc_vocab: int = 256204,
    ):
        super().__init__()
        self.resampler = PerceiverResampler(
            input_hidden_size=input_hidden_size,
            output_hidden_size=output_hidden_size,
            num_latents=num_latents,
            latent_dim=latent_dim,
            num_self_layers=num_self_layers,
            cross_heads=cross_heads,
            self_heads=self_heads,
            dropout=dropout,
            use_flash_attn=use_flash_attn
        )
        self.ocr_head = nn.Sequential(
            nn.Linear(output_hidden_size, output_hidden_size),
            nn.GELU(),
            nn.Linear(output_hidden_size, ctc_vocab)
        )
        self.ctc_loss = nn.CTCLoss()
        self.padding_val = padding_val

    def compute_ctc_loss(
                self,
                logits,
                targets,
                softmax=True,
            ):
            """
            logits: (B, T, C)
            targets: (B, S) padded label sequences
            """
    
            B, T, C = logits.shape
    
            # (T, B, C) for CTCLoss
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2) if softmax else logits
            # All input lengths are full length (assumes no time padding)
            input_lengths = torch.full(
                (B,),
                T,
                dtype=torch.long,
                device=logits.device
            )
    
            targets_list = []
            target_lengths = []
    
            for i in range(B):
                seq = targets[i]
    
                mask = seq != self.padding_val
                seq = seq[mask]
    
                length = seq.size(0)
                target_lengths.append(length)
    
                if length > 0:
                    targets_list.append(seq)
    
            # Handle empty batch safely
            if len(targets_list) == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
            targets_concat = torch.cat(targets_list).to(logits.device)
            target_lengths = torch.tensor(
                target_lengths,
                dtype=torch.long,
                device=logits.device
            )
    
            loss = self.ctc_loss(
                log_probs,
                targets_concat,
                input_lengths,
                target_lengths
            )
            return loss

    @staticmethod
    def align_loss(hypothesis, target):
            hypothesis = hypothesis.view(-1,hypothesis.size(-1)).T
            hyp_cov = torch.cov(hypothesis)
            hyp_mean = torch.mean(hypothesis, dim=1, keepdim=True)
            hyp_stats = torch.cat((hyp_cov, hyp_mean), dim=-1)
    
            target = target.view(-1, target.size(-1)).T
            tgt_cov = torch.cov(target)
            tgt_mean = torch.mean(target, dim=1, keepdim=True)
            tgt_stats = torch.cat((tgt_cov, tgt_mean), dim=-1)
    
            return nn.functional.mse_loss(hyp_stats, tgt_stats)

    def forward(
            self,
            source,
            attn_mask,
            grid_hw,
            align_target,
            ctc_target
    ):
        projection = self.resampler(source, attn_mask=attn_mask, grid_hw=grid_hw)
        align = self.align_loss(projection, align_target)
        ctc = self.compute_ctc_loss(self.ocr_head(projection), ctc_target)
        return projection, align, ctc

    def inference(
            self,
            source,
            attn_mask,
            grid_hw
    ):
        return self.resampler(source, attn_mask=attn_mask, grid_hw=grid_hw)

class Translator(pl.LightningModule):
    def __init__(self, 
                 vision_encoder,
                 mt_model,
                 tokenizer,
                 src_lang,
                 tgt_langs,
                 encoder_lr,
                 adapter_lr,
                 decoder_lr,
                 min_lr,
                 max_steps,
                 step_milestones=[-1,-1,-1],
                 mt_loss=1.0,
                 ctc_loss=1.0,
                 align_loss=1.0,
                 padding_value=1,
                 flash_attn=False,
                 adapter_args={}):
        super().__init__()
        self.save_hyperparameters(ignore=['vision_encoder','mt_model', 'tokenizer'])
        self.vision_encoder = vision_encoder
        self.mt_model = mt_model
        self.flash_attn = flash_attn
        assert len(step_milestones) == 3 , "A fixed number of 3 milestones must be set."
        self.milestones = step_milestones
        # Language tokens
        self.tokenizer = tokenizer
        src_id = self.tokenizer.convert_tokens_to_ids(src_lang)
        tgt_id = {lang_id: self.tokenizer.convert_tokens_to_ids(lang_id) for lang_id in tgt_langs}
        self.register_buffer('src_lang_tok', self.mt_model.model.shared.weight[src_id].detach().clone().unsqueeze(0).unsqueeze(0))
        self.tgt_lang_id = tgt_id
        self.vista = Vista(
            use_flash_attn = flash_attn,
            padding_val = padding_value,
            ctc_vocab = 256204,
            **adapter_args
        )

        self.ctc_loss = nn.CTCLoss()
        self.ctc_weight = ctc_loss
        self.mt_weight = mt_loss
        self.align_weight = align_loss
        self.padding_val = padding_value

        # Validation metrics
        self.hyp = []
        self.tgt = []

    def prepend_lang(self, embeddings):
        embeddings = torch.cat([self.src_lang_tok.expand(embeddings.size(0), -1, -1), embeddings], dim=1)
        return embeddings

    def forward(self, pixel_values, labels, image_grid_thw, cu_seqlens, align_targets, ctc_targets):
        """
        Returns:
            nllb outputs
            image adpter outputs
            ocr head outputs
        """
        vision_outputs = self.vision_encoder(
            pixel_values=pixel_values,
            grid_thw=image_grid_thw,
            cu_seqlens=cu_seqlens
        ).last_hidden_state

        vision_outputs, pad_mask = unflatten_to_batch(vision_outputs, cu_seqlens)
        vision_outputs = vision_outputs.to(dtype=next(self.vista.parameters()).dtype)

        input_embeds, align_loss, ctc_loss = self.vista(vision_outputs, attn_mask=pad_mask, grid_hw=image_grid_thw[:, 1:], align_target=align_targets, ctc_target=ctc_targets)
        if self.flash_attn:
            input_embeds = input_embeds.to(dtype=torch.float16)

        input_embeds = self.prepend_lang(input_embeds)

        labels = torch.concat([labels[:,1:],labels[:,:1]], dim=1)
        decoder_input_ids = shift_tokens_right(
            labels,
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.mt_model.config.decoder_start_token_id
        )
        bsz, seq_len, _ = input_embeds.shape
        encoder_attention_mask = torch.ones((bsz, seq_len), device=input_embeds.device, dtype=torch.long)
        encoder_outputs = BaseModelOutput(last_hidden_state=input_embeds)
        mt_outputs = self.mt_model(
            #inputs_embeds=input_embeds,
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            labels=decoder_input_ids,
        )
        return mt_outputs, align_loss, ctc_loss

    def inference(self, pixel_values, image_grid_thw, cu_seqlens, langs):
        vision_outputs = self.vision_encoder(
            pixel_values=pixel_values,
            grid_thw=image_grid_thw,
            cu_seqlens=cu_seqlens
        ).last_hidden_state

        vision_outputs, pad_mask = unflatten_to_batch(vision_outputs, cu_seqlens)
        vision_outputs = vision_outputs.to(dtype=next(self.vista.parameters()).dtype)

        input_embeds = self.vista.inference(vision_outputs, attn_mask=pad_mask, grid_hw=image_grid_thw[:, 1:])
        input_embeds = self.prepend_lang(input_embeds)
        if self.flash_attn:
            input_embeds = input_embeds.to(dtype=torch.float16)
        bsz, seq_len, _ = input_embeds.shape
        encoder_attention_mask = torch.ones((bsz, seq_len), device=input_embeds.device, dtype=torch.long)
        encoder_outputs = BaseModelOutput(last_hidden_state=input_embeds)

        generated_tokens = self.mt_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            forced_bos_token_id=[self.tgt_lang_id[lang] for lang in langs],
            max_length=128,
        )
        return generated_tokens
    
    def training_step(self, batch, batch_idx):
        batch_size = batch['pixel_values'].size(0)
        with torch.no_grad():
            encoder_outputs = self.mt_model.model.encoder(input_ids = batch['src_txt']).last_hidden_state
        mt_output, align_loss, ocr_loss = self(
            batch['pixel_values'], 
            batch['tgt'],
            batch['image_grid_thw'],
            batch['cu_seqlens'],
            encoder_outputs,
            batch['src_txt']
        )
        
        self.log('train_loss', mt_output.loss, prog_bar=True, batch_size=batch_size)
        self.log('train_ctc', ocr_loss, prog_bar=False, batch_size=batch_size)
        self.log('train_align', align_loss, prog_bar=False, batch_size=batch_size)

        loss = self.mt_weight * mt_output.loss + self.ctc_weight * ocr_loss + self.align_weight * align_loss
        self.log('frozen', sum(p.numel() for p in self.mt_model.model.decoder.parameters() if p.requires_grad) == 0, batch_size=batch_size)

        if self.global_step == self.milestones[0]:
            self.unfreeze_vision_encoder_top_layers(2)
        if self.global_step == self.milestones[1]:
            self.unfreeze_decoder_cross_attention()
        if self.global_step == self.milestones[2]:
            self.unfreeze_decoder_top_layers(2)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch['pixel_values'].size(0)
        outputs = self.inference(
            batch['pixel_values'],
            batch['image_grid_thw'],
            batch['cu_seqlens'],
            batch['tgt_langs'],
        )

        hyp_batch = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        del outputs
        tgt_batch = self.tokenizer.batch_decode(batch['tgt'], skip_special_tokens=True)
        self.hyp.extend(hyp_batch)
        self.tgt.extend(tgt_batch)

        with torch.no_grad():
            encoder_outputs = self.mt_model.model.encoder(input_ids = batch['src_txt']).last_hidden_state
        loss = self(
            batch['pixel_values'], 
            batch['tgt'],
            batch['image_grid_thw'],
            batch['cu_seqlens'],
            encoder_outputs,
            batch['src_txt']
        )[0].loss
        self.log('val_loss', loss, prog_bar=True, batch_size=batch_size)

        return loss

    def on_validation_epoch_start(self):
        self.hyp, self.tgt = [], []

    def on_validation_epoch_end(self):
        bleu = corpus_bleu(self.hyp, [self.tgt])
        self.log('val_BLEU', bleu.score)
        ter = corpus_ter(self.hyp, [self.tgt])
        self.log('val_TER', ter.score)
        examples = [[self.hyp[idx], self.tgt[idx]] for idx in torch.randint(0, len(self.hyp), (5,),)]
        self.logger.log_text(
            key='samples',
            columns=['Hipotesis', 'Ground truth'],
            data=examples
        )
        self.hyp, self.tgt = [], []
    
    def predict_step(self, batch):
        outputs = self.inference(
            batch['pixel_values'],
            batch['image_grid_thw'],
            batch['cu_seqlens'],
            batch['tgt_langs']
        )

        hyp_batch = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return batch['name'], hyp_batch

    def configure_optimizers(self):
        self.mt_model.train()  # Ensure the model is in training mode
        encoder_lr = self.hparams.encoder_lr
        adapter_lr = self.hparams.adapter_lr
        decoder_lr = self.hparams.decoder_lr
        min_lr = self.hparams.min_lr
        max_steps = self.hparams.max_steps
        optimizer = torch.optim.AdamW([
            {"params": self.vista.parameters(), "lr": adapter_lr},
            {"params": self.vision_encoder.parameters(), "lr": encoder_lr},
            {"params": self.mt_model.model.decoder.parameters(), "lr": decoder_lr},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=min_lr)
        return [optimizer], [{
            "scheduler": scheduler,
            "interval": "step",
        }]
    
    def unfreeze_vision_encoder_top_layers(self, n: int):
        layers = self.vision_encoder.encoder.layers
        for layer in layers[max(0, len(layers) - n):]:
            for param in layer.parameters():
                param.requires_grad = True

        for param in self.vision_encoder.post_layernorm.parameters():
            param.requires_grad = True

    def unfreeze_decoder_cross_attention(self):
        """Unfreeze the decoder's cross-attention parameters so they require grad.

        This tries to locate the decoder module under `self.mt_model.model.decoder`
        and enables gradients for any parameter whose name indicates it
        belongs to cross-attention (common names: `cross_attn`,
        `encoder_attn`, `enc_dec_attn`). If no decoder is found the method
        is a no-op.
        """
        decoder = self.mt_model.model.decoder

        changed = 0
        for name, p in decoder.named_parameters():
            lname = name.lower()
            if "encoder_attn" in lname:
                if not p.requires_grad:
                    p.requires_grad = True
                    changed += 1

        return changed

    def unfreeze_decoder_top_layers(self, n: int):
        """Unfreeze the top `n` layers of `self.mt_model.model.decoder`.

        This method attempts to locate the decoder's layer modules using a
        few common attribute names. It will unfreeze parameters in the last
        `n` layers (the layers closest to the output). Returns the number of
        parameters whose `requires_grad` flag was changed from False to True.
        """
        if n <= 0:
            return 0

        decoder = self.mt_model.model.decoder

        # Prefer the explicit ModuleList used by M2M100Decoder: `layers`
        layers = None
        if hasattr(decoder, 'layers') and isinstance(decoder.layers, (list, nn.ModuleList)):
            layers = list(decoder.layers)
        elif hasattr(decoder, 'layer') and isinstance(decoder.layer, (list, nn.ModuleList)):
            layers = list(decoder.layer)

        if not layers:
            raise Exception("Decoder layers not found in the module to unfreeze.")

        total = len(layers)
        n = min(n, total)

        # Top layers: last `n` entries
        layers_to_unfreeze = layers[-n:]

        changed = 0
        for layer in layers_to_unfreeze:
            for p in layer.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    changed += 1

        return changed
